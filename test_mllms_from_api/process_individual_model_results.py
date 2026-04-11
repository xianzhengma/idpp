#!/usr/bin/env python3
"""
处理个体模型结果文件，计算评价指标并生成论文格式的表格
支持相对(relative)和绝对(absolute)两种测试结果
"""

import os
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr

def extract_info_from_filename(filename):
    """
    从文件名中提取模型版本、测试集和测试类型
    """
    match = re.match(r'(\w+)_vjepa_(test\d+)_(relative|absolute)_results\.txt', filename)
    if match:
        return match.groups()
    return None, None, None

def extract_comparison_results(file_path, result_type):
    """
    从结果文件中提取预测结果和真实结果
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        property_match = re.search(r'属性: (\w+)', content)
        if not property_match:
            print(f"无法从文件 {file_path} 中提取属性")
            return None
        property_name = property_match.group(1)
        
        if result_type == 'relative':
            comparisons = re.findall(r'预测结果: ([\d.]+)\s+GT结果: (\d+)', content)
            if not comparisons:
                print(f"未找到比较结果在文件: {file_path}")
                return None
            predictions = [float(p) for p, _ in comparisons]
            ground_truths = [int(g) for _, g in comparisons]
        else: # absolute
            comparisons = re.findall(r'预测值: ([\d.-]+)\s+GT值: ([\d.-]+)', content)
            if not comparisons:
                print(f"未找到比较结果在文件: {file_path}")
                return None
            predictions = [float(p) for p, _ in comparisons]
            ground_truths = [float(g) for _, g in comparisons]

        return {
            'property_name': property_name,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, result_type, property_name=None):
    """
    根据结果类型计算评估指标
    
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        result_type: 结果类型 ('relative' 或 'absolute')
        property_name: 属性名称，用于elasticity过滤
    """
    metrics = {}
    try:
        if result_type == 'relative':
            if len(set(y_true)) < 2:
                metrics['AUC'] = 0.5
                print(f"警告: 只有一个真实类别 {set(y_true)}")
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred)
            metrics['Accuracy'] = sum(1 for p, g in zip(y_pred, y_true) if p == g) / len(y_true)
        else: # absolute
            # 对于elasticity属性，过滤掉ground truth小于0.1的数据
            if property_name and property_name.lower() == 'elasticity':
                filtered_indices = [i for i, gt in enumerate(y_true) if gt >= 0.1]
                if filtered_indices:
                    y_true_filtered = [y_true[i] for i in filtered_indices]
                    y_pred_filtered = [y_pred[i] for i in filtered_indices]
                    print(f"  -> elasticity过滤: 原始{len(y_true)}个样本，过滤后{len(y_true_filtered)}个样本 (GT>=0.1)")
                else:
                    print(f"  -> elasticity过滤: 所有样本GT都小于0.1，无法计算指标")
                    return None
            else:
                y_true_filtered = y_true
                y_pred_filtered = y_pred
            
            if len(y_true_filtered) < 2:
                metrics['Pearson_Correlation'] = 0.0
                metrics['MAE'] = 0.0
            else:
                correlation, _ = pearsonr(y_true_filtered, y_pred_filtered)
                metrics['Pearson_Correlation'] = correlation
                metrics['MAE'] = np.mean(np.abs(np.array(y_true_filtered) - np.array(y_pred_filtered)))
        return metrics
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None

def main():
    # base_dir = "test_mllms_from_api/qwen_max_physics_analysis_results"
    # base_dir = "test_mllms_from_api/gpt_physics_analysis_results"
    # base_dir = "test_mllms_from_api/gemini_physics_analysis_results"
    base_dir = "test_mllms_from_api/gpt4o_physics_analysis_results_absolute_100samples_icl"
    # base_dir = "test_mllms_from_api/qwen_physics_analysis_results_absolute_100samples_icl"
    # base_dir = "test_mllms_from_api/gemini_physics_analysis_results_relative_100samples_oracle"


    
    all_results = []
    
    print(f"开始处理目录: {base_dir}")
    
    for filename in sorted(os.listdir(base_dir)):
        if not filename.endswith("_results.txt"):
            continue

        file_path = os.path.join(base_dir, filename)
        model_version, test_set, result_type = extract_info_from_filename(filename)
        
        if not all([model_version, test_set, result_type]):
            print(f"跳过不匹配的文件: {filename}")
            continue
            
        print(f"\n处理文件: {filename}")
        
        result_data = extract_comparison_results(file_path, result_type)
        
        if result_data:
            metrics = calculate_metrics(result_data['ground_truths'], result_data['predictions'], result_type, result_data['property_name'])
            
            if metrics:
                row = {
                    'Model': f"{model_version}_vjepa",
                    'Property': result_data['property_name'],
                    'Test_Set': test_set,
                    'Type': result_type,
                    'Num_Samples': len(result_data['predictions']),
                }
                row.update(metrics)
                all_results.append(row)
                
                metrics_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
                print(f"  -> {row['Model']} - {row['Property']} - {row['Test_Set']} ({row['Type']}): {metrics_str}")

    if not all_results:
        print("没有处理任何结果，退出。")
        return

    df = pd.DataFrame(all_results)
    
    # 分离相对和绝对结果进行处理
    df_relative = df[df['Type'] == 'relative']
    df_absolute = df[df['Type'] == 'absolute']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not df_relative.empty:
        print("\n" + "="*50)
        print("生成相对结果的论文格式表格 (AUC)...")
        pivot_relative = df_relative.pivot_table(index='Property', columns='Test_Set', values='AUC')
        pivot_relative = pivot_relative.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_relative.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_relative = pivot_relative.round(4)
        
        # 添加Average列
        pivot_relative['Average'] = pivot_relative.mean(axis=1).round(4)
        
        output_filename_relative = f"paper_table_relative_results_{timestamp}.csv"
        pivot_relative.to_csv(output_filename_relative)
        print(f"\n相对结果表格已保存为: {output_filename_relative}")
        print("\n表格预览 (AUC):")
        print(pivot_relative.round(2).to_string())

    if not df_absolute.empty:
        print("\n" + "="*50)
        print("生成绝对结果的论文格式表格 (Pearson Correlation)...")
        pivot_absolute = df_absolute.pivot_table(index='Property', columns='Test_Set', values='Pearson_Correlation')
        pivot_absolute = pivot_absolute.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_absolute.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_absolute = pivot_absolute.round(4)
        
        # 添加Average列
        pivot_absolute['Average'] = pivot_absolute.mean(axis=1).round(4)

        output_filename_absolute = f"paper_table_absolute_results_{timestamp}.csv"
        pivot_absolute.to_csv(output_filename_absolute)
        print(f"\n绝对结果表格已保存为: {output_filename_absolute}")
        print("\n表格预览 (Pearson Correlation):")
        print(pivot_absolute.round(2).to_string())

    # 保存所有详细结果
    detailed_filename = f"detailed_all_results_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False)
    print(f"\n所有详细结果已保存为: {detailed_filename}")
    
    print("\n" + "="*50)
    print("总结:")
    print(f"- 处理了 {len(df)} 个文件")
    if not df_relative.empty:
        print(f"- 相对结果 (AUC): 平均 = {df_relative['AUC'].mean():.4f}")
    if not df_absolute.empty:
        print(f"- 绝对结果 (Pearson): 平均 = {df_absolute['Pearson_Correlation'].mean():.4f}")
        print(f"- 绝对结果 (MAE): 平均 = {df_absolute['MAE'].mean():.4f}")


if __name__ == "__main__":
    main() 