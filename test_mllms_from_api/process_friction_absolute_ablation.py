#!/usr/bin/env python3
"""
Friction Absolute Ablation Results Processor
处理friction absolute ablation实验的结果并生成对比表格
"""

import os
import glob
import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np


def get_experiment_name_from_dir(dir_path: str) -> str:
    """
    从目录路径提取实验名称
    
    参数:
        dir_path: 实验结果目录路径
        
    返回:
        实验名称
    """
    dir_name = os.path.basename(dir_path)
    
    # 专门处理friction ablation实验的目录命名
    if dir_name == 'gemini_physics_analysis_results_friction_absolute_20samples':
        return 'Baseline'
    elif dir_name.startswith('gemini_physics_analysis_results_friction_absolute_20samples_'):
        suffix = dir_name.replace('gemini_physics_analysis_results_friction_absolute_20samples_', '')
        
        # 映射ablation实验的目录名到显示名称
        mapping = {
            'frameindex': '+ FrameIndex',
            'icl': '+ ICL',
            'oracle': '+ Oracle',
            'frameindex_icl': '+ FrameIndex+ICL',
            'frameindex_oracle': '+ FrameIndex+Oracle',
            'icl_oracle': '+ ICL+Oracle',
            'frameindex_icl_oracle': '+ FrameIndex+ICL+Oracle',
        }
        
        return mapping.get(suffix, f'+ {suffix.replace("_", "+").title()}')
    
    return dir_name


def extract_predictions_and_gts(result_file: str) -> List[Dict]:
    """
    从结果文件中提取预测值和GT值
    
    参数:
        result_file: 结果文件路径
        
    返回:
        包含预测值和GT值的字典列表
    """
    results = []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有的样本结果
    pattern = r'--- 样本 (\d+) ---\n.*?视频: (.*?)\n.*?预测值: (.*?)\n.*?GT值: (.*?)\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        sample_id, video, predicted_str, gt_str = match
        
        # 解析预测值
        predicted = None
        if predicted_str != 'None' and predicted_str.strip():
            try:
                predicted = float(predicted_str)
            except ValueError:
                predicted = None
        
        # 解析GT值
        gt = None
        if gt_str != 'None' and gt_str.strip():
            try:
                gt = float(gt_str)
            except ValueError:
                gt = None
        
        results.append({
            'sample_id': int(sample_id),
            'video': video.strip(),
            'predicted': predicted,
            'gt': gt
        })
    
    return results


def calculate_metrics(predictions: List[float], ground_truths: List[float]) -> Dict[str, float]:
    """
    计算评估指标
    
    参数:
        predictions: 预测值列表
        ground_truths: GT值列表
        
    返回:
        包含各种指标的字典
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'mse': None,
            'mae': None,
            'pearson_r': None,
            'pearson_p': None,
            'valid_predictions': 0,
            'total_samples': len(ground_truths)
        }
    
    # 只使用有效的预测值计算指标
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                   if p is not None and g is not None]
    
    if len(valid_pairs) == 0:
        return {
            'mse': None,
            'mae': None,
            'pearson_r': None,
            'pearson_p': None,
            'valid_predictions': 0,
            'total_samples': len(ground_truths)
        }
    
    valid_predictions, valid_gts = zip(*valid_pairs)
    valid_predictions = list(valid_predictions)
    valid_gts = list(valid_gts)
    
    # 计算指标
    mse = mean_squared_error(valid_gts, valid_predictions)
    mae = mean_absolute_error(valid_gts, valid_predictions)
    
    try:
        pearson_r, pearson_p = pearsonr(valid_predictions, valid_gts)
    except:
        pearson_r, pearson_p = None, None
    
    return {
        'mse': mse,
        'mae': mae,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'valid_predictions': len(valid_predictions),
        'total_samples': len(ground_truths)
    }


def process_single_experiment(exp_dir: str) -> Dict:
    """
    处理单个实验的结果
    
    参数:
        exp_dir: 实验目录路径
        
    返回:
        实验结果摘要
    """
    print(f"处理实验: {exp_dir}")
    
    # 查找friction absolute的结果文件
    result_files = glob.glob(os.path.join(exp_dir, "*friction*absolute*results.txt"))
    
    if not result_files:
        print(f"  警告: 未找到friction absolute结果文件")
        return None
    
    all_results = []
    test_metrics = {}
    
    # 处理每个测试集的结果
    for result_file in result_files:
        filename = os.path.basename(result_file)
        
        # 从文件名提取测试集名称
        if 'test1' in filename:
            test_name = 'Test-1'
        elif 'test2' in filename:
            test_name = 'Test-2'
        elif 'test3' in filename:
            test_name = 'Test-3'
        else:
            test_name = 'Unknown'
        
        # 提取预测值和GT值
        results = extract_predictions_and_gts(result_file)
        all_results.extend(results)
        
        if results:
            predictions = [r['predicted'] for r in results]
            ground_truths = [r['gt'] for r in results]
            
            metrics = calculate_metrics(predictions, ground_truths)
            test_metrics[test_name] = metrics
            
            print(f"  {test_name}: {metrics['valid_predictions']}/{metrics['total_samples']} 有效预测")
    
    return {
        'experiment_name': get_experiment_name_from_dir(exp_dir),
        'directory': exp_dir,
        'test_metrics': test_metrics,
        'total_results': len(all_results)
    }


def main():
    """主函数"""
    print("开始处理 Friction Absolute Ablation 实验结果...")
    
    # 查找所有相关的实验目录
    experiment_dirs = []
    
    # 查找所有可能的实验目录
    base_patterns = [
        "gemini_physics_analysis_results_friction_absolute_20samples",
        "gemini_physics_analysis_results_friction_absolute_20samples_frameindex",
        "gemini_physics_analysis_results_friction_absolute_20samples_icl",
        "gemini_physics_analysis_results_friction_absolute_20samples_oracle",
        "gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl",
        "gemini_physics_analysis_results_friction_absolute_20samples_frameindex_oracle",
        "gemini_physics_analysis_results_friction_absolute_20samples_icl_oracle",
        "gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl_oracle",
    ]
    
    for pattern in base_patterns:
        matching_dirs = glob.glob(pattern)
        experiment_dirs.extend(matching_dirs)
    
    # 去重并排序
    experiment_dirs = sorted(list(set(experiment_dirs)))
    
    if not experiment_dirs:
        print("❌ 未找到任何实验结果目录")
        print("请确保已运行 friction ablation 实验并且结果目录存在")
        return
    
    print(f"找到 {len(experiment_dirs)} 个实验目录:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir}")
    print()
    
    # 处理每个实验
    experiment_results = []
    for exp_dir in experiment_dirs:
        result = process_single_experiment(exp_dir)
        if result:
            experiment_results.append(result)
    
    if not experiment_results:
        print("❌ 没有成功处理任何实验结果")
        return
    
    # 生成对比表格
    print("生成对比表格...")
    
    # 准备数据
    table_data = []
    test_names = ['Test-1', 'Test-2', 'Test-3']
    
    for exp_result in experiment_results:
        row = {
            'Experiment': exp_result['experiment_name'],
            'Directory': os.path.basename(exp_result['directory'])
        }
        
        # 添加每个测试集的指标
        avg_metrics = {'mse': [], 'mae': [], 'pearson_r': [], 'valid_count': [], 'total_count': []}
        
        for test_name in test_names:
            if test_name in exp_result['test_metrics']:
                metrics = exp_result['test_metrics'][test_name]
                
                # MSE
                mse_val = metrics['mse']
                row[f'{test_name}_MSE'] = f"{mse_val:.6f}" if mse_val is not None else "N/A"
                if mse_val is not None:
                    avg_metrics['mse'].append(mse_val)
                
                # MAE
                mae_val = metrics['mae']
                row[f'{test_name}_MAE'] = f"{mae_val:.6f}" if mae_val is not None else "N/A"
                if mae_val is not None:
                    avg_metrics['mae'].append(mae_val)
                
                # Pearson相关系数
                pearson_val = metrics['pearson_r']
                row[f'{test_name}_Pearson'] = f"{pearson_val:.4f}" if pearson_val is not None else "N/A"
                if pearson_val is not None:
                    avg_metrics['pearson_r'].append(pearson_val)
                
                # 有效预测数
                row[f'{test_name}_Valid'] = f"{metrics['valid_predictions']}/{metrics['total_samples']}"
                avg_metrics['valid_count'].append(metrics['valid_predictions'])
                avg_metrics['total_count'].append(metrics['total_samples'])
            else:
                row[f'{test_name}_MSE'] = "N/A"
                row[f'{test_name}_MAE'] = "N/A"
                row[f'{test_name}_Pearson'] = "N/A"
                row[f'{test_name}_Valid'] = "N/A"
        
        # 计算平均值
        if avg_metrics['mse']:
            row['Avg_MSE'] = f"{np.mean(avg_metrics['mse']):.6f}"
        else:
            row['Avg_MSE'] = "N/A"
            
        if avg_metrics['mae']:
            row['Avg_MAE'] = f"{np.mean(avg_metrics['mae']):.6f}"
        else:
            row['Avg_MAE'] = "N/A"
            
        if avg_metrics['pearson_r']:
            row['Avg_Pearson'] = f"{np.mean(avg_metrics['pearson_r']):.4f}"
        else:
            row['Avg_Pearson'] = "N/A"
        
        if avg_metrics['valid_count'] and avg_metrics['total_count']:
            total_valid = sum(avg_metrics['valid_count'])
            total_all = sum(avg_metrics['total_count'])
            row['Total_Valid'] = f"{total_valid}/{total_all}"
        else:
            row['Total_Valid'] = "N/A"
        
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 按实验名称排序（Baseline在前）
    experiment_order = [
        'Baseline',
        '+ FrameIndex', 
        '+ ICL', 
        '+ Oracle',
        '+ FrameIndex+ICL', 
        '+ FrameIndex+Oracle', 
        '+ ICL+Oracle',
        '+ FrameIndex+ICL+Oracle'
    ]
    
    df['exp_order'] = df['Experiment'].map({exp: i for i, exp in enumerate(experiment_order)})
    df = df.sort_values('exp_order').drop('exp_order', axis=1)
    
    # 显示表格
    print("\n" + "="*120)
    print("Friction Absolute Ablation Study Results")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # 保存CSV文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"friction_absolute_20samples_ablation_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"\n✅ 结果已保存到: {csv_filename}")
    
    # 生成性能分析
    if 'Baseline' in df['Experiment'].values:
        print(f"\n" + "="*80)
        print("性能分析 (相对于Baseline的改进)")
        print("="*80)
        
        baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
        
        for _, row in df.iterrows():
            if row['Experiment'] == 'Baseline':
                continue
            
            print(f"\n{row['Experiment']}:")
            
            # 分析每个指标
            for metric in ['MSE', 'MAE', 'Pearson']:
                baseline_val = baseline_row[f'Avg_{metric}']
                current_val = row[f'Avg_{metric}']
                
                if baseline_val != "N/A" and current_val != "N/A":
                    baseline_num = float(baseline_val)
                    current_num = float(current_val)
                    
                    if metric in ['MSE', 'MAE']:
                        # MSE和MAE越小越好
                        improvement = (baseline_num - current_num) / baseline_num * 100
                        if improvement > 0:
                            print(f"  {metric}: {improvement:+.1f}% (更好)")
                        else:
                            print(f"  {metric}: {improvement:+.1f}% (更差)")
                    else:  # Pearson
                        # Pearson相关系数越大越好
                        if baseline_num != 0:
                            improvement = (current_num - baseline_num) / abs(baseline_num) * 100
                            if improvement > 0:
                                print(f"  {metric}: {improvement:+.1f}% (更好)")
                            else:
                                print(f"  {metric}: {improvement:+.1f}% (更差)")
                        else:
                            print(f"  {metric}: N/A (基线为0)")


if __name__ == "__main__":
    main() 