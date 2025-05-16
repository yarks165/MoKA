# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class AdapterMLP(nn.Module):
    def __init__(self, config, kg_module=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.kg_module = kg_module

        self.moe = SharedParallelMoE(config)
        
        self.kg_dim = 4096
        self.kg_down_proj = nn.Linear(self.intermediate_size, self.kg_dim, bias=False)
        self.kg_up_proj = nn.Linear(self.kg_dim, self.intermediate_size, bias=False)
    
    def set_KG(self):
        self.kg_module = KG_infuded_module(self.config)
        self.kg_module = self.kg_module.cuda()
        
    def forward(self, x, words_ents_list=None, words_subtoken_map=None, input_ids=None):
        '''
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        '''
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            base_up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            
            if hasattr(self, 'moe'):
                expert_slice = self.moe.num_experts // self.config.pretraining_tp
                moe_outputs = []
                
                for tp_rank in range(self.config.pretraining_tp):
                    start = tp_rank * expert_slice
                    end = (tp_rank + 1) * expert_slice
                    
                    expert_outputs = []
                    for idx in range(start, end):
                        expert = self.moe.expert_adapters[idx]
                        expert_outputs.append(expert(x))
                    
                    all_expert_outputs = [torch.zeros_like(expert_outputs[0]).to(x.device) 
                                        for _ in range(self.config.pretraining_tp)]
                    torch.distributed.all_gather(all_expert_outputs, torch.cat(expert_outputs, dim=1))
                    
                    router_input = x.mean(dim=1)
                    gathered_router_input = [torch.zeros_like(router_input).to(x.device) 
                                            for _ in range(self.config.pretraining_tp)]
                    torch.distributed.all_gather(gathered_router_input, router_input)
                    full_router_input = torch.cat(gathered_router_input, dim=0)
                    
                    if tp_rank == 0:
                        router_logits = self.moe.router(full_router_input)
                        router_weights = torch.softmax(router_logits / self.moe.temperature, dim=-1)
                        topk_weights, topk_idx = torch.topk(router_weights, k=2)
                        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
                    else:
                        topk_weights = None
                        topk_idx = None
                    
                    topk_weights = broadcast(topk_weights, src=0)
                    topk_idx = broadcast(topk_idx, src=0)
                    
                    selected_experts = []
                    for i in range(2):
                        expert_idx = topk_idx[:, i]
                        mask = (expert_idx >= start) & (expert_idx < end)
                        if mask.any():
                            local_idx = expert_idx[mask] - start
                            selected = all_expert_outputs[tp_rank][:, local_idx]
                            selected *= topk_weights[mask, i].unsqueeze(-1)
                            selected_experts.append(selected)
                    
                    if len(selected_experts) > 0:
                        moe_outputs.append(torch.sum(torch.cat(selected_experts), dim=1))
                    else:
                        moe_outputs.append(torch.zeros_like(base_up_proj))
                
                moe_out = sum(moe_outputs) / self.config.pretraining_tp
                up_out = base_up_proj + moe_out
            else:
                up_out = base_up_proj

            intermediate_states = self.act_fn(gate_proj) * up_out
            if self.kg_module is not None:
                reduced_states = self.kg_down_proj(intermediate_states)
                processed_states = self.kg_module(reduced_states, words_ents_list, words_subtoken_map, input_ids)
                intermediate_states = self.kg_up_proj(processed_states)
                
            intermediate_states_split = intermediate_states.split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states_split[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            gate = self.act_fn(self.gate_proj(x))
            up_out = self.up_proj(x) + self.moe(x)
            intermediate_states = gate * up_out
            if self.kg_module is not None:
                reduced_states = self.kg_down_proj(intermediate_states)
                processed_states = self.kg_module(reduced_states, words_ents_list, words_subtoken_map, input_ids)
                intermediate_states = self.kg_up_proj(processed_states)
                
            down_proj = self.down_proj(intermediate_states)
        
        return down_proj

class SharedParallelMoE(nn.Module):
    def __init__(self, config, num_experts=4, expert_rank=16, k=2, lora_alpha=16):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = num_experts
        self.k = k
        self.expert_rank = expert_rank
        self.lora_alpha = lora_alpha  
        self.expert_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, expert_rank, bias=False),
                nn.GELU(),
                nn.Linear(expert_rank, self.intermediate_size, bias=False)
            ) for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(self.hidden_size, num_experts, bias=False)
        
        self.lb_loss_coef = 0.003
        self.register_buffer("current_routing_weight", torch.zeros(1))
        self.register_buffer("current_attention_mask", torch.zeros(1))
        self.register_buffer("last_topk_idx", None)
        self.gamma_div_balance = 1.0 
        self.gamma_div_certain = 0.4 
        
    def init_weight(self):
        for expert in self.expert_adapters:
            nn.init.kaiming_uniform_(expert[0].weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(expert[2].weight)
        nn.init.xavier_normal_(self.router.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        expert_outputs = torch.stack([
            e(x) * (self.lora_alpha / self.expert_rank)
            for e in self.expert_adapters
        ], dim=2)  # [B, S, E, D]
        
        router_logits = self.router(x)
        router_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_idx = torch.topk(router_weights, self.k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        self.last_topk_idx = topk_idx.detach()
        
        selected_experts = torch.gather(
            expert_outputs, 
            dim=2, 
            index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.intermediate_size)
        )
        
        moe_out = (selected_experts * topk_weights.unsqueeze(-1)).sum(dim=2)
        
        expert_mask = torch.zeros_like(router_weights)
        expert_mask.scatter_(2, topk_idx, 1)
        selected_counts = expert_mask.sum(dim=(0,1))
            
        self.current_routing_weight = router_weights.detach()
        self.current_attention_mask = torch.ones_like(x[..., 0], dtype=torch.float)
        return moe_out
    
    def load_balancing_loss(self):
        if not hasattr(self, 'current_routing_weight') or self.current_routing_weight.sum() == 0:
            print("current_routing_weight None")
            return torch.tensor(0.0).to(self.router.weight.device)
        
        routing_weight = self.current_routing_weight
        mask = self.current_attention_mask
        topk_idx = self.last_topk_idx  
        count = torch.zeros_like(routing_weight)
        count.scatter_(-1, topk_idx, 1)
        
        num_token = mask.sum()
        freq = (count * mask.unsqueeze(-1)).sum((0,1)) / (num_token * self.k + 1e-8)
        prop = (routing_weight * mask.unsqueeze(-1)).sum((0,1)) / (num_token + 1e-8)
        
        loss = (prop * freq).sum() * self.num_experts
        # print(f"balancing_loss: {loss}")
        return loss * self.lb_loss_coef
    
    def divergence_loss(self):
        if not hasattr(self, 'current_routing_weight') or self.current_routing_weight.sum() == 0:
            print("current_routing_weight None")
            return torch.tensor(0.0).to(self.router.weight.device)
        
        routing_weight = self.current_routing_weight  # [B, S, E]
        mask = self.current_attention_mask.to(routing_weight.dtype).unsqueeze(-1)  # [B, S, 1]
        
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=routing_weight.dtype, 
                                        device=routing_weight.device))
        
        max_entropy_m = self.gamma_div_balance * max_entropy
        min_entropy_p = self.gamma_div_certain * max_entropy
        max_div = max_entropy_m - min_entropy_p
        
        num_token = torch.sum(mask)  # scalar
        
        m = torch.sum(routing_weight * mask, dim=(0,1)) / (num_token + 1e-8)  # [E]
        entropy_m = -torch.sum(m * torch.log(m + 1e-9))  # scalar
        entropy_m = torch.clamp(entropy_m, max=max_entropy_m)
        
        entropy_p = -torch.sum(routing_weight * torch.log(routing_weight + 1e-9), dim=-1)  # [B, S]
        entropy_p = torch.clamp(entropy_p, min=min_entropy_p) * mask.squeeze(-1)  # [B, S]
        entropy_p = torch.sum(entropy_p) / (num_token + 1e-8)
        
        loss = torch.relu(max_div - (entropy_m - entropy_p)) / (max_entropy + 1e-8)
        return loss * self.lb_loss_coef

class KGMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
class KG_infuded_module(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()                    
        
        self.concept_embed = None
        self.interlayer = 100
        self.knowledge_sentinel = nn.Embedding(1, self.interlayer).cuda()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda()

        self.convert_matrix_entity = KGMLP(
            hidden_size=100,
            intermediate_size=config.kg_intermediate_size,
            output_size=4096,
            hidden_act=config.hidden_act,
        ).cuda()
        self.MLP = nn.Linear(config.hidden_size + self.interlayer, config.hidden_size).cuda()
        self.act_fn = ACT2FN[config.hidden_act]
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.dim = self.interlayer

    def init(self, embedding_path):
        self.id2concept, self.concept2id, embedding_mat, (concept_size, dim) = self.read_conceptnet_embedding(embedding_path)
        self.concept_embed = nn.Embedding.from_pretrained(torch.from_numpy(embedding_mat)).cpu()
        torch.nn.init.xavier_uniform_(self.knowledge_sentinel.weight)

    def read_concept_embedding(self, embedding_path):
        fin = open(embedding_path, encoding='utf-8')
        info = [line.strip() for line in fin]
        dim = len(info[0].split(' ')[1:])
        n_concept = len(info)
        embedding_mat = []
        id2concept, concept2id = [], {}
        for line in info:
            concept_name = line.split(' ')[0]
            embedding = [float(value_str) for value_str in line.split(' ')[1:]]
            assert len(embedding) == dim and not np.any(np.isnan(embedding))
            embedding_mat.append(embedding)
            concept2id[concept_name] = len(id2concept)
            id2concept.append(concept_name)
        embedding_mat = np.array(embedding_mat, dtype=np.float32)
        fin.close()
        return id2concept, concept2id, embedding_mat, (n_concept, dim)
    
    def read_conceptnet_embedding(self, embedding_path):
        ar_load = np.load(embedding_path)
        embedding_mat = np.array(ar_load, dtype=np.float32)
        return None, None, embedding_mat, (embedding_mat.shape[0], 100)

    def forward(self, output_hidden_states, words_ents_list, words_subtoken_map, input_ids):
        """
        Infused KG to the embeddings of output_hidden_states

        Args:
            output_hidden_states: Output of each decoder layer.
                size: [batch_size, seq_length, hidden_size_dim]
        """
        bsz, _, _ = output_hidden_states.size()
        output = None
        if output_hidden_states.size()[1] == 1:
            return output_hidden_states
        #residual = output_hidden_states
        residual = output_hidden_states
        output_hidden_states = self.input_layernorm(output_hidden_states)
        for i in range(bsz):
            hidden_state = output_hidden_states[i]
            try:
                words_ents = words_ents_list[i].long().to(hidden_state.device)
            except:
                words_ents = words_ents_list[0]
                try:
                    words_ents = words_ents.long().to(hidden_state.device)
                except:
                    pass
                # words_ents = words_ents_list[0].long().to(hidden_state.device)
            if len(words_ents) == 0:
                if output == None:
                    output = hidden_state.unsqueeze(0)
                else:
                    output = torch.cat((output, hidden_state.unsqueeze(0)), 0)
                continue
            
            pad_embed = torch.zeros_like(hidden_state[0]).unsqueeze(0)
            hidden_state = torch.cat((hidden_state, pad_embed), dim = 0)
            
            converted_words_ents = words_ents.masked_fill(words_ents == -1, 0)
            ents_embeds = self.concept_embed(converted_words_ents.cpu()).to(hidden_state.device)
            knowledge_sentinel = self.knowledge_sentinel(torch.LongTensor([0]).to(hidden_state.device)).view(1, 1, -1).repeat(ents_embeds.size()[0], 1, 1)
            try:
                ents_embeds = torch.cat((ents_embeds, knowledge_sentinel), 1)
                ent_ori_embeds = ents_embeds
            except:
                print(ents_embeds)
                print(words_ents)
                ent_ori_embeds = ents_embeds

            ents_embeds = self.convert_matrix_entity(ents_embeds)
            #words_subtoken = torch.LongTensor(words_subtoken_map[i]).to(hidden_state.device)
            try:
                words_subtoken = words_subtoken_map[i].long().to(hidden_state.device)
            except:
                words_subtoken = words_subtoken_map[0].long().to(hidden_state.device)
            # words_subtoken_embeds size: [map_num, max_subtoken_num, hidden_size]

            # Avg pooling of each word(tokens)
            sub_token_num = words_subtoken.ne(-1).sum(1)
            """
            print("words_subtoken is {}".format(words_subtoken))
            index = words_subtoken.view(-1)
            print("index is {}".format(index))
            index_fixed = index.masked_fill(index == -1, hidden_state.size()[0] - 1)
            print("index_fixed is {}".format(index_fixed))
            b = torch.index_select(hidden_state, index=index_fixed, dim = 0)
            b = b.view(words_subtoken.size()[0], words_subtoken.size()[1], -1)
            """
            index_fixed = words_subtoken.masked_fill(words_subtoken == -1, hidden_state.size()[0] - 1)
            b = hidden_state[words_subtoken]
            b = torch.sum(b, dim = 1).squeeze()
            b = torch.div(b, sub_token_num.view(-1, 1))
            #b = self.convert_matrix_token(b).unsqueeze(2)
            b = b.unsqueeze(2)
            atten_weight = torch.bmm(ents_embeds, b)

            # Compute attention mask
            atten_ones = torch.ones([words_ents.size()[0], 1]).to(hidden_state.device)
            words_ents = torch.cat([words_ents, atten_ones], -1)
            attention_mask = torch.zeros_like(words_ents).to(hidden_state.device)
            #attention_mask = words_ents.masked_fill(words_ents != -1, value=torch.tensor(0))
            attention_mask = attention_mask.masked_fill(words_ents == -1, value=torch.tensor(-1e9))
            atten_weight = atten_weight.squeeze() + attention_mask
            # atten_weight size: [map_num, top_k + 1]
            attn_weights = nn.functional.softmax(atten_weight, dim=-1, dtype=torch.float32).to(ents_embeds.dtype).unsqueeze(1)
            attn_output = torch.bmm(attn_weights, ent_ori_embeds)
            attn_output = attn_output.repeat(1, words_subtoken.size()[1], 1).view(-1, self.dim)
            index_fixed = index_fixed.flatten()
            tmp = torch.zeros([hidden_state.size()[0], self.dim]).to(hidden_state.device).to(hidden_state.dtype)
            KG_infused = tmp.index_copy(0, index_fixed, attn_output)
            KG_infused = torch.cat((hidden_state, KG_infused), -1)[: -1, :]
            KG_infused = self.MLP(KG_infused)
            KG_infused = self.act_fn(KG_infused).unsqueeze(0)
            if output == None:
                output = KG_infused
                #print(output.size())
            else:
                output = torch.cat((output, KG_infused), 0)
                #print(output.size())
        # print(output.size())
        # print(output_hidden_states.size())
        assert output.size() == output_hidden_states.size()
        #output = torch.nn.functional.dropout(output, p=0.1, training=True)
        output = output * self.alpha + residual
        return output

class LlamaDecoderLayer_1(nn.Module):
    def __init__(self, idx, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.store_value_mha = 0
        self.store_value_ffn = 0
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init(self):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        words_ents_list = None,
        words_subtoken_map = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        idx = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        pre = hidden_states[-1][-1]
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        post = hidden_states[-1][-1]
        if hidden_states.size()[1] != 1:
            self.store_value_mha += (torch.cosine_similarity(pre, post, dim=0).item())
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        pre = hidden_states[-1][-1]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        post = hidden_states[-1][-1]
        if hidden_states.size()[1] != 1:
            self.store_value_ffn += (torch.cosine_similarity(pre, post, dim=0).item())
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class LlamaDecoderLayer_2(nn.Module):
    def __init__(self, idx, config: LlamaConfig):
        super().__init__()
        self.idx = idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        
        if (idx + 1) in list(config.layer_insertion):
            self.mlp = AdapterMLP(config)
            # self.mlp = MoEAdapter(config=config, num_experts=4, top_k=2, kg_module=kg_module)
        else:
            self.mlp = LlamaMLP(config)
            
        self.store_value_mha = 0
        self.store_value_ffn = 0
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init(self):
        if (self.idx + 1) in list(self.config.layer_insertion):
            self.mlp.set_KG()
            # self.mlp.moe.init_weight()
        else:
            pass
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        words_ents_list = None,
        words_subtoken_map = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        idx = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        pre = hidden_states[-1][-1]
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        post = hidden_states[-1][-1]
        if hidden_states.size()[1] != 1:
            self.store_value_mha += (torch.cosine_similarity(pre, post, dim=0).item())
        hidden_states = residual + hidden_states

        residual = hidden_states
        pre = hidden_states[-1][-1]
        hidden_states = self.post_attention_layernorm(hidden_states)

        if (self.idx + 1) in list(self.config.layer_insertion):
            hidden_states = self.mlp(hidden_states, words_ents_list, words_subtoken_map, None)
        else:
            hidden_states = self.mlp(hidden_states)
            
        post = hidden_states[-1][-1]
        if hidden_states.size()[1] != 1:
            self.store_value_ffn += (torch.cosine_similarity(pre, post, dim=0).item())
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def forward_ffd(self, hidden_states):
        # Fully Connected
        residual = hidden_states[0]
        hidden_states[0] = self.post_attention_layernorm(hidden_states[0])
        hidden_states[0] = self.mlp(hidden_states[0])
        hidden_states[0] = residual + hidden_states[0]

        return hidden_states

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.first = True
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        print(config.iskg)
        if config.iskg:
            self.layers = nn.ModuleList([LlamaDecoderLayer_2(idx, config) for idx in range(config.num_hidden_layers)])
        else:
            self.layers = nn.ModuleList([LlamaDecoderLayer_1(idx, config) for idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def activate_KG_modules(self):
        for idx, decoder_layer in enumerate(self.layers):
            if (idx + 1) in list(self.config.layer_insertion) and decoder_layer.mlp.kg_module is not None:
                for name, param in decoder_layer.mlp.kg_module.named_parameters():
                    param.requires_grad = False
        
        for idx, decoder_layer in enumerate(self.layers):
            if (idx + 1) in list(self.config.layer_insertion) and decoder_layer.mlp.kg_module is not None:
                for name, param in decoder_layer.mlp.kg_module.named_parameters():
                    if name != 'concept_embed.weight':
                        param.requires_grad = True
                for name, param in decoder_layer.mlp.moe.named_parameters():
                     param.requires_grad = True
    

    def load_off(self):
        for i in range(len(self.layers)):
            if (i + 1) not in list(self.config.layer_insertion):
                continue
            else:
                if self.layers[i].mlp.kg_module is not None and self.layers[i].mlp.kg_module.concept_embed is not None:
                    self.layers[i].mlp.kg_module.concept_embed = self.layers[i].mlp.kg_module.concept_embed.cpu()

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        words_ents_list = None,
        words_subtoken_map = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        try:
            if self.first:
                self.load_off()
                self.first = False
        except:
            pass
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    words_ents_list = words_ents_list,
                    words_subtoken_map = words_subtoken_map,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    idx=idx
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def activate_KG_modules(self):
        self.model.activate_KG_modules()

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        words_ents_list = None,
        words_subtoken_map = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            words_ents_list = words_ents_list,
            words_subtoken_map = words_subtoken_map,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            lb_loss = 0.0
            for idx, layer in enumerate(self.model.layers):
                if (idx + 1) in self.config.layer_insertion:
                    if hasattr(layer.mlp.moe, 'divergence_loss'):
                        lb_loss += layer.mlp.moe.divergence_loss()
                    # if hasattr(layer.mlp.moe, 'load_balancing_loss'):
                    #     lb_loss += layer.mlp.moe.load_balancing_loss()
            loss += lb_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "words_ents_list": kwargs.get("words_ents_list"),
                "words_subtoken_map": kwargs.get("words_subtoken_map"),
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
            lb_loss = 0.0
            for idx, layer in enumerate(self.model.layers):
                if (idx + 1) in self.config.layer_insertion:
                    if hasattr(layer.mlp.moe, 'divergence_loss'):
                        lb_loss += layer.mlp.moe.divergence_loss()
                    # if hasattr(layer.mlp.moe, 'load_balancing_loss'):
                    #     lb_loss += layer.mlp.moe.load_balancing_loss()
            loss += lb_loss
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
