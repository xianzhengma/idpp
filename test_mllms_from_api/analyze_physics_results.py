#!/usr/bin/env python3
"""
物理分析结果评估脚本
- 绝对值测试：计算Pearson相关系数
- 相对测试：计算ROC指标
- 导出CSV结果文件
"""

import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


def parse_results_file(file_path: str) -> Dict:
    """
    解析物理分析结果文件
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        解析后的数据字典
    """
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按分割线分割不同的测试组
    sections = content.split('------------------------------------------------------------')
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if len(lines) < 5:
            continue
            
        # 解析测试信息
        test_info = {}
        for line in lines[:10]:  # 前10行通常包含测试信息
            if ':' in line:
                key, value = line.split(':', 1)
                test_info[key.strip()] = value.strip()
        
        if 'Version' not in test_info or 'Property' not in test_info:
            continue
            
        version = test_info['Version']
        property_name = test_info['Property']
        test_type = test_info['Type']
        test_set = test_info['Test Set']
        
        # 创建唯一标识符
        test_id = f"{version}_{property_name}_{test_type}_{test_set}"
        
        # 解析结果数据
        if test_type == 'absolute':
            # 绝对值测试 - 解析 predicted 和 gt 值
            predicted_values = []
            gt_values = []
            
            for line in lines:
                if 'predicted=' in line and 'gt=' in line:
                    # 提取 predicted 和 gt 值
                    pred_match = re.search(r'predicted=([0-9.]+|None)', line)
                    gt_match = re.search(r'gt=([0-9.e-]+)', line)
                    
                    if pred_match and gt_match:
                        pred_str = pred_match.group(1)
                        if pred_str != 'None':
                            predicted_values.append(float(pred_str))
                            gt_values.append(float(gt_match.group(1)))
            
            results[test_id] = {
                'type': 'absolute',
                'property': property_name,
                'version': version,
                'test_set': test_set,
                'predicted': predicted_values,
                'ground_truth': gt_values
            }
            
        elif test_type == 'relative':
            # 相对测试 - 解析比较结果
            predicted_values = []
            gt_values = []
            
            for line in lines:
                if 'predicted=' in line and 'gt=' in line:
                    # 提取 predicted 和 gt 值
                    pred_match = re.search(r'predicted=([01])', line)
                    gt_match = re.search(r'gt=([01])', line)
                    
                    if pred_match and gt_match:
                        predicted_values.append(int(pred_match.group(1)))
                        gt_values.append(int(gt_match.group(1)))
            
            results[test_id] = {
                'type': 'relative',
                'property': property_name,
                'version': version,
                'test_set': test_set,
                'predicted': predicted_values,
                'ground_truth': gt_values
            }
    
    return results


def calculate_pearson_correlation(predicted: List[float], ground_truth: List[float]) -> Tuple[float, float]:
    """
    计算Pearson相关系数
    
    Args:
        predicted: 预测值列表
        ground_truth: 真实值列表
        
    Returns:
        相关系数和p值
    """
    if len(predicted) != len(ground_truth) or len(predicted) == 0:
        return 0.0, 1.0
        
    correlation, p_value = pearsonr(predicted, ground_truth)
    return correlation, p_value


def calculate_roc_metrics(predicted: List[int], ground_truth: List[int]) -> Dict:
    """
    计算ROC相关指标
    
    Args:
        predicted: 预测值列表 (0或1)
        ground_truth: 真实值列表 (0或1)
        
    Returns:
        ROC指标字典
    """
    if len(predicted) != len(ground_truth) or len(predicted) == 0:
        return {
            'auc': 0.0,
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # 计算AUC
    try:
        auc = roc_auc_score(ground_truth, predicted)
    except ValueError:
        # 如果只有一个类别，AUC无法计算
        auc = 0.5
    
    # 计算混淆矩阵
    cm = confusion_matrix(ground_truth, predicted)
    
    # 计算各种指标
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, sum(predicted))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def export_to_csv(absolute_results: Dict, relative_results: Dict, output_dir: str = ".") -> None:
    """
    将分析结果导出为CSV文件
    
    Args:
        absolute_results: 绝对值测试结果
        relative_results: 相对测试结果
        output_dir: 输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 导出绝对值测试结果
    if absolute_results:
        abs_data = []
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(
                data['predicted'], data['ground_truth']
            )
            
            # 计算额外的统计信息
            pred_array = np.array(data['predicted'])
            gt_array = np.array(data['ground_truth'])
            
            # 计算平均绝对误差和均方根误差
            mae = np.mean(np.abs(pred_array - gt_array))
            rmse = np.sqrt(np.mean((pred_array - gt_array)**2))
            
            abs_data.append({
                'Property': data['property'],
                'Version': data['version'],
                'Test_Set': data['test_set'],
                'Test_ID': test_id,
                'Sample_Count': len(data['predicted']),
                'Pearson_Correlation': correlation,
                'P_Value': p_value,
                'Significance': 'Significant' if p_value < 0.05 else 'Not Significant',
                'Correlation_Strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak',
                'Mean_Predicted': np.mean(pred_array),
                'Mean_Ground_Truth': np.mean(gt_array),
                'Std_Predicted': np.std(pred_array),
                'Std_Ground_Truth': np.std(gt_array),
                'MAE': mae,
                'RMSE': rmse,
                'Min_Predicted': np.min(pred_array),
                'Max_Predicted': np.max(pred_array),
                'Min_Ground_Truth': np.min(gt_array),
                'Max_Ground_Truth': np.max(gt_array)
            })
        
        df_abs = pd.DataFrame(abs_data)
        abs_filename = f"physics_analysis_absolute_{timestamp}.csv"
        abs_filepath = os.path.join(output_dir, abs_filename)
        df_abs.to_csv(abs_filepath, index=False, encoding='utf-8')
        print(f"✅ 绝对值测试结果已导出至: {abs_filepath}")
    
    # 导出相对测试结果
    if relative_results:
        rel_data = []
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            
            # 计算额外的统计信息
            pred_array = np.array(data['predicted'])
            gt_array = np.array(data['ground_truth'])
            
            rel_data.append({
                'Property': data['property'],
                'Version': data['version'],
                'Test_Set': data['test_set'],
                'Test_ID': test_id,
                'Sample_Count': len(data['predicted']),
                'AUC': metrics['auc'],
                'Accuracy': metrics['accuracy'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'Predicted_Positive_Rate': np.mean(pred_array),
                'Ground_Truth_Positive_Rate': np.mean(gt_array),
                'True_Positives': np.sum((pred_array == 1) & (gt_array == 1)),
                'True_Negatives': np.sum((pred_array == 0) & (gt_array == 0)),
                'False_Positives': np.sum((pred_array == 1) & (gt_array == 0)),
                'False_Negatives': np.sum((pred_array == 0) & (gt_array == 1))
            })
        
        df_rel = pd.DataFrame(rel_data)
        rel_filename = f"physics_analysis_relative_{timestamp}.csv"
        rel_filepath = os.path.join(output_dir, rel_filename)
        df_rel.to_csv(rel_filepath, index=False, encoding='utf-8')
        print(f"✅ 相对测试结果已导出至: {rel_filepath}")
    
    # 导出汇总统计信息
    summary_data = []
    
    # 绝对值测试汇总
    if absolute_results:
        all_correlations = []
        property_stats = {}
        
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(
                data['predicted'], data['ground_truth']
            )
            all_correlations.append(correlation)
            
            prop = data['property']
            if prop not in property_stats:
                property_stats[prop] = []
            property_stats[prop].append(correlation)
        
        # 总体统计
        summary_data.append({
            'Test_Type': 'Absolute',
            'Property': 'All',
            'Metric': 'Pearson_Correlation',
            'Mean': np.mean(all_correlations),
            'Std': np.std(all_correlations),
            'Min': np.min(all_correlations),
            'Max': np.max(all_correlations),
            'Count': len(all_correlations)
        })
        
        # 按属性统计
        for prop, corrs in property_stats.items():
            summary_data.append({
                'Test_Type': 'Absolute',
                'Property': prop,
                'Metric': 'Pearson_Correlation',
                'Mean': np.mean(corrs),
                'Std': np.std(corrs),
                'Min': np.min(corrs),
                'Max': np.max(corrs),
                'Count': len(corrs)
            })
    
    # 相对测试汇总
    if relative_results:
        all_aucs = []
        all_accuracies = []
        property_stats = {}
        
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            all_aucs.append(metrics['auc'])
            all_accuracies.append(metrics['accuracy'])
            
            prop = data['property']
            if prop not in property_stats:
                property_stats[prop] = {'aucs': [], 'accuracies': []}
            property_stats[prop]['aucs'].append(metrics['auc'])
            property_stats[prop]['accuracies'].append(metrics['accuracy'])
        
        # 总体统计 - AUC
        summary_data.append({
            'Test_Type': 'Relative',
            'Property': 'All',
            'Metric': 'AUC',
            'Mean': np.mean(all_aucs),
            'Std': np.std(all_aucs),
            'Min': np.min(all_aucs),
            'Max': np.max(all_aucs),
            'Count': len(all_aucs)
        })
        
        # 总体统计 - Accuracy
        summary_data.append({
            'Test_Type': 'Relative',
            'Property': 'All',
            'Metric': 'Accuracy',
            'Mean': np.mean(all_accuracies),
            'Std': np.std(all_accuracies),
            'Min': np.min(all_accuracies),
            'Max': np.max(all_accuracies),
            'Count': len(all_accuracies)
        })
        
        # 按属性统计
        for prop, stats in property_stats.items():
            summary_data.append({
                'Test_Type': 'Relative',
                'Property': prop,
                'Metric': 'AUC',
                'Mean': np.mean(stats['aucs']),
                'Std': np.std(stats['aucs']),
                'Min': np.min(stats['aucs']),
                'Max': np.max(stats['aucs']),
                'Count': len(stats['aucs'])
            })
            
            summary_data.append({
                'Test_Type': 'Relative',
                'Property': prop,
                'Metric': 'Accuracy',
                'Mean': np.mean(stats['accuracies']),
                'Std': np.std(stats['accuracies']),
                'Min': np.min(stats['accuracies']),
                'Max': np.max(stats['accuracies']),
                'Count': len(stats['accuracies'])
            })
    
    # 导出汇总统计
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_filename = f"physics_analysis_summary_{timestamp}.csv"
        summary_filepath = os.path.join(output_dir, summary_filename)
        df_summary.to_csv(summary_filepath, index=False, encoding='utf-8')
        print(f"✅ 汇总统计结果已导出至: {summary_filepath}")


def analyze_physics_results(file_path: str) -> None:
    """
    分析物理测试结果的主函数
    
    Args:
        file_path: 结果文件路径
    """
    print("=" * 80)
    print("物理分析结果评估")
    print("=" * 80)
    
    # 解析结果文件
    results = parse_results_file(file_path)
    
    if not results:
        print("❌ 未找到有效的测试结果")
        return
    
    # 分组结果
    absolute_results = {}
    relative_results = {}
    
    for test_id, data in results.items():
        if data['type'] == 'absolute':
            absolute_results[test_id] = data
        elif data['type'] == 'relative':
            relative_results[test_id] = data
    
    # 分析绝对值测试结果
    if absolute_results:
        print("\n📊 绝对值测试结果 (Pearson相关系数)")
        print("-" * 50)
        
        all_correlations = []
        property_correlations = {}
        
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(
                data['predicted'], data['ground_truth']
            )
            
            all_correlations.append(correlation)
            
            prop = data['property']
            if prop not in property_correlations:
                property_correlations[prop] = []
            property_correlations[prop].append(correlation)
            
            print(f"📈 {data['property']} - {data['test_set']} ({data['version']})")
            print(f"   样本数: {len(data['predicted'])}")
            print(f"   Pearson相关系数: {correlation:.4f}")
            print(f"   P值: {p_value:.4f}")
            print(f"   相关性: {'强' if abs(correlation) > 0.7 else '中等' if abs(correlation) > 0.3 else '弱'}")
            print()
        
        # 统计摘要
        if all_correlations:
            print("📋 绝对值测试总结:")
            print(f"   平均相关系数: {np.mean(all_correlations):.4f}")
            print(f"   相关系数范围: {np.min(all_correlations):.4f} ~ {np.max(all_correlations):.4f}")
            print(f"   标准差: {np.std(all_correlations):.4f}")
            
            print("\n📋 按属性分组的平均相关系数:")
            for prop, corrs in property_correlations.items():
                print(f"   {prop}: {np.mean(corrs):.4f} (±{np.std(corrs):.4f})")
    
    # 分析相对测试结果
    if relative_results:
        print("\n🎯 相对测试结果 (ROC指标)")
        print("-" * 50)
        
        all_aucs = []
        property_aucs = {}
        
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            
            all_aucs.append(metrics['auc'])
            
            prop = data['property']
            if prop not in property_aucs:
                property_aucs[prop] = []
            property_aucs[prop].append(metrics['auc'])
            
            print(f"🎯 {data['property']} - {data['test_set']} ({data['version']})")
            print(f"   样本数: {len(data['predicted'])}")
            print(f"   AUC: {metrics['auc']:.4f}")
            print(f"   准确率: {metrics['accuracy']:.4f}")
            print(f"   敏感性: {metrics['sensitivity']:.4f}")
            print(f"   特异性: {metrics['specificity']:.4f}")
            print(f"   精确率: {metrics['precision']:.4f}")
            print(f"   F1分数: {metrics['f1_score']:.4f}")
            print()
        
        # 统计摘要
        if all_aucs:
            print("📋 相对测试总结:")
            print(f"   平均AUC: {np.mean(all_aucs):.4f}")
            print(f"   AUC范围: {np.min(all_aucs):.4f} ~ {np.max(all_aucs):.4f}")
            print(f"   标准差: {np.std(all_aucs):.4f}")
            
            print("\n📋 按属性分组的平均AUC:")
            for prop, aucs in property_aucs.items():
                print(f"   {prop}: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")
    
    # 创建详细结果表格
    create_detailed_results_table(absolute_results, relative_results)
    
    # 导出CSV文件
    print("\n📁 导出CSV文件...")
    export_to_csv(absolute_results, relative_results)
    print("✅ 分析完成！")


def create_detailed_results_table(absolute_results: Dict, relative_results: Dict) -> None:
    """
    创建详细的结果表格
    
    Args:
        absolute_results: 绝对值测试结果
        relative_results: 相对测试结果
    """
    print("\n📋 详细结果表格")
    print("=" * 100)
    
    # 绝对值测试表格
    if absolute_results:
        print("\n绝对值测试结果:")
        abs_data = []
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(
                data['predicted'], data['ground_truth']
            )
            abs_data.append({
                'Property': data['property'],
                'Version': data['version'],
                'Test Set': data['test_set'],
                'Samples': len(data['predicted']),
                'Pearson Correlation': f"{correlation:.4f}",
                'P-value': f"{p_value:.4f}",
                'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
            })
        
        df_abs = pd.DataFrame(abs_data)
        print(df_abs.to_string(index=False))
    
    # 相对测试表格
    if relative_results:
        print("\n相对测试结果:")
        rel_data = []
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            rel_data.append({
                'Property': data['property'],
                'Version': data['version'],
                'Test Set': data['test_set'],
                'Samples': len(data['predicted']),
                'AUC': f"{metrics['auc']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df_rel = pd.DataFrame(rel_data)
        print(df_rel.to_string(index=False))


if __name__ == "__main__":
    # 分析结果文件
    # file_path = "test_mllms_from_api/gpt_physics_analysis_results/summary_results_20250708_152926.txt"
    file_path = "test_mllms_from_api/gemini_physics_analysis_results/summary_results_20250709_080535.txt"
    analyze_physics_results(file_path) 