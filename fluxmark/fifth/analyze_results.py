#!/usr/bin/env python3
"""
OrthoFlow 实验结果分析脚本
读取 test_results.json 并生成详细报告和可视化
"""

import json
import numpy as np
import os
from datetime import datetime

output_dir = '/data/daiyina/project_flux/pic'

def load_results(experiment_name=None):
    """加载实验结果"""
    if experiment_name is None:
        # 自动找到最新的结果文件
        files = [f for f in os.listdir(output_dir) if f.endswith('_test_results.json')]
        if not files:
            print("未找到结果文件")
            return None
        latest = max(files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        experiment_name = latest.replace('_test_results.json', '')
    
    result_path = os.path.join(output_dir, f'{experiment_name}_test_results.json')
    with open(result_path, 'r') as f:
        return json.load(f), experiment_name

def print_summary_table(results):
    """打印汇总表格"""
    overall = results['overall_stats']
    baseline_mean = overall['clean']['mean']
    
    print("\n" + "="*100)
    print(f"📊 鲁棒性测试汇总: {results['experiment_name']}")
    print(f"   图像数量: {len(results['results'])}, 每攻击运行次数: {results['num_runs_per_attack']}")
    print("="*100)
    print(f"{'攻击类型':<20} {'均值':>10} {'标准差':>10} {'中位数':>10} {'Q25':>10} {'Q75':>10} {'保留率':>10} {'状态':>8}")
    print("-"*100)
    
    for attack in results['attacks']:
        name = attack['name']
        stats = overall[name]
        retention = stats['mean'] / baseline_mean * 100 if name != 'clean' else 100.0
        
        if stats['mean'] > 0.85:
            status = "✅ 强"
        elif stats['mean'] > 0.70:
            status = "🟢 良"
        elif stats['mean'] > 0.55:
            status = "🟡 中"
        elif stats['mean'] > 0.40:
            status = "🟠 弱"
        else:
            status = "🔴 差"
        
        print(f"{name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['median']:>10.4f} "
              f"{stats['q25']:>10.4f} {stats['q75']:>10.4f} {retention:>9.1f}% {status:>8}")
    
    print("="*100)

def print_per_image_analysis(results):
    """打印每张图像的详细分析"""
    print("\n📋 逐图像分析:")
    print("-"*100)
    
    for img_result in results['results']:
        img_id = img_result['id']
        clean_score = img_result['attacks']['clean']['cos_sim_mean']
        
        print(f"\n{img_id}: {img_result['prompt'][:50]}...")
        print(f"  基准分数: {clean_score:.4f}")
        
        # 找出最脆弱和最鲁棒的攻击
        attack_scores = []
        for attack_name, attack_data in img_result['attacks'].items():
            if attack_name != 'clean':
                attack_scores.append((attack_name, attack_data['cos_sim_mean']))
        
        attack_scores.sort(key=lambda x: x[1])
        
        print(f"  最脆弱攻击: {attack_scores[0][0]} ({attack_scores[0][1]:.4f})")
        print(f"  最鲁棒攻击: {attack_scores[-1][0]} ({attack_scores[-1][1]:.4f})")

def print_latex_table(results):
    """打印 LaTeX 表格"""
    overall = results['overall_stats']
    baseline_mean = overall['clean']['mean']
    
    print("\n📄 LaTeX 表格代码:")
    print("-"*100)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{OrthoFlow Robustness Evaluation (Mean Cosine Similarity)}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Attack Type & Mean & Std & Retention (\\%) & Status \\\\")
    print("\\hline")
    
    for attack in results['attacks']:
        name = attack['name'].replace('_', ' ')
        stats = overall[attack['name']]
        retention = stats['mean'] / baseline_mean * 100 if attack['name'] != 'clean' else 100.0
        
        if stats['mean'] > 0.85:
            status = "Strong"
        elif stats['mean'] > 0.70:
            status = "Good"
        elif stats['mean'] > 0.55:
            status = "Moderate"
        elif stats['mean'] > 0.40:
            status = "Weak"
        else:
            status = "Poor"
        
        print(f"{name} & {stats['mean']:.4f} & {stats['std']:.4f} & {retention:.1f} & {status} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

def print_csv_data(results):
    """打印 CSV 格式数据"""
    print("\n📊 CSV 格式数据:")
    print("-"*100)
    print("attack_type,mean,std,median,min,max,q25,q75")
    
    overall = results['overall_stats']
    for attack in results['attacks']:
        name = attack['name']
        stats = overall[name]
        print(f"{name},{stats['mean']:.6f},{stats['std']:.6f},{stats['median']:.6f},"
              f"{stats['min']:.6f},{stats['max']:.6f},{stats['q25']:.6f},{stats['q75']:.6f}")

def compare_experiments(exp_names):
    """比较多个实验"""
    print("\n🔬 多实验对比:")
    print("-"*100)
    
    all_results = {}
    for name in exp_names:
        result_path = os.path.join(output_dir, f'{name}_test_results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                all_results[name] = json.load(f)
        else:
            print(f"警告: 未找到 {name}")
    
    if not all_results:
        return
    
    # 打印对比表
    print(f"{'攻击类型':<20}", end='')
    for name in all_results:
        print(f"{name[:15]:>15}", end='')
    print()
    print("-"*100)
    
    # 获取所有攻击类型
    first_result = list(all_results.values())[0]
    for attack in first_result['attacks']:
        attack_name = attack['name']
        print(f"{attack_name:<20}", end='')
        
        for name, result in all_results.items():
            mean = result['overall_stats'][attack_name]['mean']
            print(f"{mean:>15.4f}", end='')
        print()

def main():
    import sys
    
    # 解析命令行参数
    experiment_name = None
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    
    # 加载结果
    result = load_results(experiment_name)
    if result is None:
        return
    
    results, exp_name = result
    print(f"\n📁 加载实验: {exp_name}")
    print(f"   生成时间: {results['timestamp']}")
    
    # 打印各种报告
    print_summary_table(results)
    print_per_image_analysis(results)
    print_latex_table(results)
    print_csv_data(results)
    
    # 如果有多个实验名参数，进行对比
    if len(sys.argv) > 2:
        compare_experiments(sys.argv[1:])

if __name__ == '__main__':
    main()
