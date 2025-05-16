import re
from collections import defaultdict

def parse_results(file_path):
    """解析单个测试结果文件"""
    samples = defaultdict(dict)
    current_sample = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 匹配样本编号
            sample_match = re.match(r'样本 (\d+):', line)
            if sample_match:
                current_sample = int(sample_match.group(1))
                continue
                
            # 提取关键字段
            if current_sample is not None:
                if line.startswith('问题:'):
                    samples[current_sample]['question'] = line.split(':', 1)[1].strip()
                elif line.startswith('预测:'):
                    samples[current_sample]['pred'] = line.split(':', 1)[1].strip()
                elif line.startswith('正确答案:'):
                    samples[current_sample]['answer'] = line.split(':', 1)[1].strip()
                elif line.startswith('是否正确:'):
                    samples[current_sample]['correct'] = line.split(':', 1)[1].strip() == 'True'
    
    return samples

def generate_venn_data(your_results, baseline_results, llama_results):
    """生成Venn图所需的重叠计数"""
    sets = {
        'yours_correct': set(),
        'baseline_correct': set(),
        'llama_correct': set()
    }
    
    for sample_id in your_results:
        try:
            # 你的模型正确
            y_correct = your_results[sample_id]['correct']
            # baseline错误
            b_correct = baseline_results[sample_id]['correct']
            # llama错误
            l_correct = llama_results[sample_id]['correct']
            
            if y_correct: sets['yours_correct'].add(sample_id)
            if b_correct: sets['baseline_correct'].add(sample_id)
            if l_correct: sets['llama_correct'].add(sample_id)
        except KeyError:
            continue
    
    # 计算各区域样本数
    venn_data = {
        'your_only': len(sets['yours_correct'] - sets['baseline_correct'] - sets['llama_correct']),
        'baseline_only': len(sets['baseline_correct'] - sets['yours_correct'] - sets['llama_correct']),
        'llama_only': len(sets['llama_correct'] - sets['yours_correct'] - sets['baseline_correct']),
        'your_baseline': len(sets['yours_correct'] & sets['baseline_correct'] - sets['llama_correct']),
        'your_llama': len(sets['yours_correct'] & sets['llama_correct'] - sets['baseline_correct']),
        'baseline_llama': len(sets['baseline_correct'] & sets['llama_correct'] - sets['yours_correct']),
        'all_correct': len(sets['yours_correct'] & sets['baseline_correct'] & sets['llama_correct'])
    }
    return venn_data

def find_special_cases(your_file, baseline_file, llama_file, output_file):
    """对比三个模型的结果"""
    # 解析所有结果
    your_results = parse_results(your_file)
    baseline_results = parse_results(baseline_file)
    llama_results = parse_results(llama_file)
    
    # 查找符合条件的样本
    special_cases = []
    for sample_id in your_results:
        try:
            # 验证三个文件的样本一致性
            assert your_results[sample_id]['question'] == baseline_results[sample_id]['question'], f"样本{sample_id}问题不一致"
            assert your_results[sample_id]['question'] == llama_results[sample_id]['question'], f"样本{sample_id}问题不一致"
            
            # 判断条件：你的模型正确，其他两个错误
            if (your_results[sample_id]['correct'] and 
                baseline_results[sample_id]['correct'] and 
                not llama_results[sample_id]['correct']):
                
                special_cases.append({
                    'sample_id': sample_id,
                    'question': your_results[sample_id]['question'],
                    'your_pred': your_results[sample_id]['pred'],
                    'baseline_pred': baseline_results[sample_id]['pred'],
                    'llama_pred': llama_results[sample_id]['pred'],
                    'correct_answer': your_results[sample_id]['answer']
                })
        except KeyError:
            print(f"警告：样本{sample_id}在某个文件中缺失，已跳过")
            continue
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"找到 {len(special_cases)} 个特殊案例\n")
        f.write("="*80 + "\n")
        
        for case in special_cases:
            f.write(f"样本 {case['sample_id']}:\n")
            f.write(f"问题: {case['question']}\n")
            f.write(f"正确答案: {case['correct_answer']}\n")
            f.write(f"您的预测: {case['your_pred']}\n")
            f.write(f"Baseline预测: {case['baseline_pred']}\n")
            f.write(f"Llama预测: {case['llama_pred']}\n")
            f.write("\n")
    
    print(f"分析完成！结果已保存到 {output_file}")

# 使用示例（请替换实际文件路径）
if __name__ == "__main__":
    your_file = "results_siqa.txt"
    baseline_file = "siqa_1.txt"
    llama_file = "siqa_llama2.txt"
    output_file = "case_study.txt"
    
    your_results = parse_results(your_file)
    baseline_results = parse_results(baseline_file)
    llama_results = parse_results(llama_file)
    
    venn_data = generate_venn_data(your_results, baseline_results, llama_results)
    print(venn_data)