#!/usr/bin/env python3
"""
Elasticity Relative Ablation Results Processor
处理elasticity relative ablation实验的结果并生成对比表格
"""

import os
import glob
import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score
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
    
    # 专门处理elasticity relative ablation实验的目录命名
    if dir_name == 'gemini_physics_analysis_results_elasticity_relative_20samples':
        return 'Baseline'
    elif dir_name.startswith('gemini_physics_analysis_results_elasticity_relative_20samples_'):
        suffix = dir_name.replace('gemini_physics_analysis_results_elasticity_relative_20samples_', '')
        
        # 映射ablation实验的目录名到显示名称
        mapping = {
            'concat': '+ Concat',
            'icl': '+ ICL',
            'oracle': '+ Oracle',
            'frameindex': '+ FrameIndex',
            'concat_icl': '+ Concat+ICL',
            'concat_oracle': '+ Concat+Oracle',
            'concat_frameindex': '+ Concat+FrameIndex',
            'icl_oracle': '+ ICL+Oracle',
            'icl_frameindex': '+ ICL+FrameIndex',
            'oracle_frameindex': '+ Oracle+FrameIndex',
            'concat_icl_oracle': '+ Concat+ICL+Oracle',
            'concat_icl_frameindex': '+ Concat+ICL+FrameIndex',
            'concat_oracle_frameindex': '+ Concat+Oracle+FrameIndex',
            'icl_oracle_frameindex': '+ ICL+Oracle+FrameIndex',
            'concat_icl_oracle_frameindex': '+ Concat+ICL+Oracle+FrameIndex',
        }
        
        return mapping.get(suffix, f'+ {suffix.replace("_", "+").title()}')
    
    return dir_name


def extract_comparisons_and_gts(result_file: str) -> List[Dict]:
    """
    从结果文件中提取比较结果和GT值
    
    参数:
        result_file: 结果文件路径
        
    返回:
        包含预测值和GT值的字典列表
    """
    results = []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有的比较结果
    pattern = r'--- 比较 (\d+) ---\n.*?视频对: (.*?)\n.*?预测结果: (.*?)\n.*?GT结果: (.*?)\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        comparison_id, video_pair, predicted_str, gt_str = match
        
        # 解析预测结果（现在支持浮点数）
        predicted = None
        if predicted_str != 'None' and predicted_str.strip():
            try:
                predicted = float(predicted_str)  # 改为float以支持连续值
            except ValueError:
                predicted = None
        
        # 解析GT结果（仍然是整数）
        gt = None
        if gt_str != 'None' and gt_str.strip():
            try:
                gt = int(gt_str)
            except ValueError:
                gt = None
        
        results.append({
            'comparison_id': int(comparison_id),
            'video_pair': video_pair.strip(),
            'predicted': predicted,
            'gt': gt
        })
    
    return results


def calculate_auc_metrics(predictions: List[float], ground_truths: List[int]) -> Dict[str, float]:
    """
    计算AUC评估指标
    
    参数:
        predictions: 预测值列表（连续值，0-1之间）
        ground_truths: GT值列表（二分类，0或1）
        
    返回:
        包含各种指标的字典
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'auc': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'valid_predictions': 0,
            'total_samples': len(ground_truths),
            'prediction_stats': None
        }
    
    # 只使用有效的预测值计算指标
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                   if p is not None and g is not None]
    
    if len(valid_pairs) == 0:
        return {
            'auc': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'valid_predictions': 0,
            'total_samples': len(ground_truths),
            'prediction_stats': None
        }
    
    valid_predictions, valid_gts = zip(*valid_pairs)
    valid_predictions = list(valid_predictions)
    valid_gts = list(valid_gts)
    
    # 计算预测值的统计信息
    prediction_stats = {
        'min': np.min(valid_predictions),
        'max': np.max(valid_predictions),
        'mean': np.mean(valid_predictions),
        'std': np.std(valid_predictions),
        'unique_values': sorted(list(set(valid_predictions)))
    }
    
    # 计算AUC（适用于连续预测值和二分类GT）
    try:
        # 检查GT值是否只有一个类别
        unique_gts = set(valid_gts)
        if len(unique_gts) == 1:
            auc = None  # 无法计算AUC，因为只有一个类别
        else:
            auc = roc_auc_score(valid_gts, valid_predictions)
    except Exception as e:
        print(f"  警告: AUC计算失败: {e}")
        auc = None
    
    # 对于二分类指标，需要将连续预测值转换为0/1预测
    # 使用0.5作为阈值
    binary_predictions = [1 if p >= 0.5 else 0 for p in valid_predictions]
    
    # 计算准确率
    accuracy = sum(bp == g for bp, g in zip(binary_predictions, valid_gts)) / len(valid_gts)
    
    # 计算精确率、召回率和F1分数
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    try:
        precision = precision_score(valid_gts, binary_predictions, zero_division=0)
        recall = recall_score(valid_gts, binary_predictions, zero_division=0)
        f1 = f1_score(valid_gts, binary_predictions, zero_division=0)
    except Exception as e:
        print(f"  警告: 分类指标计算失败: {e}")
        precision = None
        recall = None
        f1 = None
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'valid_predictions': len(valid_predictions),
        'total_samples': len(ground_truths),
        'prediction_stats': prediction_stats
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
    
    # 查找elasticity relative的结果文件
    result_files = glob.glob(os.path.join(exp_dir, "*relative*results.txt"))
    
    if not result_files:
        print(f"  警告: 未找到elasticity relative结果文件")
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
        
        # 提取比较结果和GT值
        results = extract_comparisons_and_gts(result_file)
        all_results.extend(results)
        
        if results:
            predictions = [r['predicted'] for r in results]
            ground_truths = [r['gt'] for r in results]
            
            metrics = calculate_auc_metrics(predictions, ground_truths)
            test_metrics[test_name] = metrics
            
            # 显示详细统计信息
            valid_count = metrics['valid_predictions']
            total_count = metrics['total_samples']
            print(f"  {test_name}: {valid_count}/{total_count} 有效预测")
            
            if metrics['prediction_stats'] is not None:
                stats = metrics['prediction_stats']
                print(f"    预测值范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"    预测值均值±标准差: {stats['mean']:.3f}±{stats['std']:.3f}")
                print(f"    预测值分布: {stats['unique_values']}")
                
            if metrics['auc'] is not None:
                print(f"    AUC: {metrics['auc']:.4f}")
            if metrics['accuracy'] is not None:
                print(f"    准确率: {metrics['accuracy']:.4f}")
    
    return {
        'experiment_name': get_experiment_name_from_dir(exp_dir),
        'directory': exp_dir,
        'test_metrics': test_metrics,
        'total_results': len(all_results)
    }


def main():
    """主函数"""
    print("开始处理 Elasticity Relative Ablation 实验结果...")
    
    # 查找所有相关的实验目录
    experiment_dirs = []
    
    # 查找所有可能的实验目录
    base_patterns = [
        "gemini_physics_analysis_results_elasticity_relative_20samples",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat",
        "gemini_physics_analysis_results_elasticity_relative_20samples_icl",
        "gemini_physics_analysis_results_elasticity_relative_20samples_oracle",
        "gemini_physics_analysis_results_elasticity_relative_20samples_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle",
        "gemini_physics_analysis_results_elasticity_relative_20samples_icl_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_oracle_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle_frameindex",
        "gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle_frameindex",
    ]
    
    for pattern in base_patterns:
        matching_dirs = glob.glob(pattern)
        experiment_dirs.extend(matching_dirs)
    
    # 去重并排序
    experiment_dirs = sorted(list(set(experiment_dirs)))
    
    if not experiment_dirs:
        print("❌ 未找到任何实验结果目录")
        print("请确保已运行 elasticity relative ablation 实验并且结果目录存在")
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
        avg_metrics = {
            'auc': [], 
            'accuracy': [], 
            'precision': [],
            'recall': [],
            'f1_score': [],
            'valid_count': [], 
            'total_count': []
        }
        
        for test_name in test_names:
            if test_name in exp_result['test_metrics']:
                metrics = exp_result['test_metrics'][test_name]
                
                # AUC
                auc_val = metrics['auc']
                row[f'{test_name}_AUC'] = f"{auc_val:.4f}" if auc_val is not None else "N/A"
                if auc_val is not None:
                    avg_metrics['auc'].append(auc_val)
                
                # Accuracy
                acc_val = metrics['accuracy']
                row[f'{test_name}_Acc'] = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                if acc_val is not None:
                    avg_metrics['accuracy'].append(acc_val)
                
                # Precision
                prec_val = metrics['precision']
                row[f'{test_name}_Prec'] = f"{prec_val:.4f}" if prec_val is not None else "N/A"
                if prec_val is not None:
                    avg_metrics['precision'].append(prec_val)
                
                # Recall
                rec_val = metrics['recall']
                row[f'{test_name}_Rec'] = f"{rec_val:.4f}" if rec_val is not None else "N/A"
                if rec_val is not None:
                    avg_metrics['recall'].append(rec_val)
                
                # F1 Score
                f1_val = metrics['f1_score']
                row[f'{test_name}_F1'] = f"{f1_val:.4f}" if f1_val is not None else "N/A"
                if f1_val is not None:
                    avg_metrics['f1_score'].append(f1_val)
                
                # 有效预测数
                row[f'{test_name}_Valid'] = f"{metrics['valid_predictions']}/{metrics['total_samples']}"
                avg_metrics['valid_count'].append(metrics['valid_predictions'])
                avg_metrics['total_count'].append(metrics['total_samples'])
            else:
                row[f'{test_name}_AUC'] = "N/A"
                row[f'{test_name}_Acc'] = "N/A"
                row[f'{test_name}_Prec'] = "N/A"
                row[f'{test_name}_Rec'] = "N/A"
                row[f'{test_name}_F1'] = "N/A"
                row[f'{test_name}_Valid'] = "N/A"
        
        # 计算平均值
        for metric_name in ['auc', 'accuracy', 'precision', 'recall', 'f1_score']:
            if avg_metrics[metric_name]:
                avg_val = np.mean(avg_metrics[metric_name])
                row[f'Avg_{metric_name.replace("_score", "").title()}'] = f"{avg_val:.4f}"
            else:
                row[f'Avg_{metric_name.replace("_score", "").title()}'] = "N/A"
        
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
        # 单个功能
        '+ Concat', 
        '+ ICL', 
        '+ Oracle',
        '+ FrameIndex',
        # 两两组合
        '+ Concat+ICL',
        '+ Concat+Oracle',
        '+ Concat+FrameIndex',
        '+ ICL+Oracle',
        '+ ICL+FrameIndex',
        '+ Oracle+FrameIndex',
        # 三功能组合
        '+ Concat+ICL+Oracle',
        '+ Concat+ICL+FrameIndex',
        '+ Concat+Oracle+FrameIndex',
        '+ ICL+Oracle+FrameIndex',
        # 四功能组合
        '+ Concat+ICL+Oracle+FrameIndex'
    ]
    
    df['exp_order'] = df['Experiment'].map({exp: i for i, exp in enumerate(experiment_order)})
    df = df.sort_values('exp_order').drop('exp_order', axis=1)
    
    # 首先显示概览表格（只显示平均值和总计）
    print("\n" + "="*120)
    print("Elasticity Relative Ablation Study Results - Overview")
    print("="*120)
    
    # 创建概览表格
    overview_columns = ['Experiment', 'Avg_Auc', 'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_F1', 'Total_Valid']
    overview_df = df[overview_columns].copy()
    print(overview_df.to_string(index=False))
    print("="*120)
    
    # 然后显示详细表格（包含每个测试集的结果）
    print(f"\n详细结果 (按测试集分解):")
    print("-" * 160)
    
    # 重新排列列的顺序以便更好地阅读
    detail_columns = ['Experiment']
    for test_name in test_names:
        detail_columns.extend([
            f'{test_name}_AUC', 
            f'{test_name}_Acc', 
            f'{test_name}_Valid'
        ])
    
    detail_df = df[detail_columns].copy()
    print(detail_df.to_string(index=False))
    print("-" * 160)
    
    # 保存CSV文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"elasticity_relative_20samples_ablation_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
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
            metrics_to_analyze = ['Auc', 'Accuracy', 'Precision', 'Recall', 'F1']
            for metric in metrics_to_analyze:
                baseline_val = baseline_row[f'Avg_{metric}']
                current_val = row[f'Avg_{metric}']
                
                if baseline_val != "N/A" and current_val != "N/A":
                    try:
                        baseline_num = float(baseline_val)
                        current_num = float(current_val)
                        
                        # 所有指标都是越大越好
                        if baseline_num != 0:
                            improvement = (current_num - baseline_num) / baseline_num * 100
                            if improvement > 0:
                                print(f"  {metric}: {improvement:+.1f}% (更好)")
                            else:
                                print(f"  {metric}: {improvement:+.1f}% (更差)")
                        else:
                            if current_num > 0:
                                print(f"  {metric}: 从0提升到{current_num:.4f} (显著改进)")
                            else:
                                print(f"  {metric}: 仍为0 (无改进)")
                    except ValueError:
                        print(f"  {metric}: 无法比较 (数据格式错误)")
                else:
                    print(f"  {metric}: N/A (缺少数据)")
        
        # 额外统计信息
        print(f"\n" + "="*80)
        print("总体统计")
        print("="*80)
        
        # 统计有效预测数
        total_experiments = len(df)
        experiments_with_predictions = 0
        total_predictions = 0
        total_samples = 0
        
        for _, row in df.iterrows():
            valid_str = row['Total_Valid']
            if valid_str != "N/A":
                try:
                    valid, total = map(int, valid_str.split('/'))
                    if valid > 0:
                        experiments_with_predictions += 1
                    total_predictions += valid
                    total_samples += total
                except:
                    pass
        
        print(f"实验总数: {total_experiments}")
        print(f"有有效预测的实验: {experiments_with_predictions}")
        print(f"总预测数: {total_predictions}/{total_samples} ({total_predictions/total_samples*100:.1f}%)")
        
        # 找出最佳表现的实验
        best_experiments = {}
        for metric in metrics_to_analyze:
            col_name = f'Avg_{metric}'
            if col_name in df.columns:
                valid_scores = []
                for _, row in df.iterrows():
                    val = row[col_name]
                    if val != "N/A":
                        try:
                            valid_scores.append((float(val), row['Experiment']))
                        except:
                            pass
                
                if valid_scores:
                    best_score, best_exp = max(valid_scores)
                    best_experiments[metric] = (best_exp, best_score)
        
        if best_experiments:
            print(f"\n最佳表现:")
            for metric, (exp_name, score) in best_experiments.items():
                print(f"  {metric}: {exp_name} ({score:.4f})")
    
    print(f"\n✅ 处理完成!")
    print(f"📊 详细结果已保存到: {csv_filename}")
    print(f"🔍 共处理 {len(experiment_results)} 个实验")
    print(f"📈 分析了 {len(test_names)} 个测试集")


if __name__ == "__main__":
    main() 