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
import matplotlib.pyplot as plt
from collections import defaultdict

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
    从结果文件中提取预测结果、真实结果以及对应的视频名称
    
    Args:
        file_path: 结果文件路径
        result_type: 结果类型 ('relative' 或 'absolute')
    
    Returns:
        dict: 包含属性名、预测值列表、真实值列表、视频名称列表和文件名
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
        
        # 根据结果类型提取视频名称
        if result_type == 'relative':
            # 相对比较格式：视频对: xxx vs yyy
            video_names = re.findall(r'视频对: ([^\n]+)', content)
        else:
            # 绝对比较格式：视频: xxx
            video_names = re.findall(r'视频: ([^\n]+)', content)
        
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

        # 确保视频名称数量与预测结果数量一致
        if len(video_names) != len(predictions):
            print(f"警告: 视频名称数量({len(video_names)})与预测结果数量({len(predictions)})不一致")
            # 如果数量不一致，截取较短的长度
            min_length = min(len(video_names), len(predictions))
            video_names = video_names[:min_length]
            predictions = predictions[:min_length]
            ground_truths = ground_truths[:min_length]

        return {
            'property_name': property_name,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'video_names': video_names,
            'filename': os.path.basename(file_path)
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
    # base_dir = "test_mllms_from_api/gpt4o_physics_analysis_results_absolute_100samples_icl"
    # base_dir = "test_mllms_from_api/qwen_physics_analysis_results_absolute_100samples_icl"
    base_dir = "test_mllms_from_api/gemini_physics_analysis_results_relative_100samples_oracle"


    
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
            video_names = result_data['video_names']
            print(f"视频名称列表 ({len(video_names)} 个):")
            for i, video_name in enumerate(video_names[:3]):  # 只打印前3个作为示例
                print(f"  {i+1}: {video_name}")
            if len(video_names) > 3:
                print(f"  ... (还有 {len(video_names)-3} 个)")
            
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
        
        output_filename_relative = f"paper_table_relative_results_{timestamp}.csv"
        pivot_relative.to_csv(output_filename_relative)
        print(f"\n相对结果表格已保存为: {output_filename_relative}")
        print("\n表格预览 (AUC):")
        print(pivot_relative.to_string())

    if not df_absolute.empty:
        print("\n" + "="*50)
        print("生成绝对结果的论文格式表格 (Pearson Correlation)...")
        pivot_absolute = df_absolute.pivot_table(index='Property', columns='Test_Set', values='Pearson_Correlation')
        pivot_absolute = pivot_absolute.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_absolute.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_absolute = pivot_absolute.round(4)

        output_filename_absolute = f"paper_table_absolute_results_{timestamp}.csv"
        pivot_absolute.to_csv(output_filename_absolute)
        print(f"\n绝对结果表格已保存为: {output_filename_absolute}")
        print("\n表格预览 (Pearson Correlation):")
        print(pivot_absolute.to_string())

    # 保存所有详细结果
    detailed_filename = f"detailed_all_results_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False)
    print(f"\n所有详细结果已保存为: {detailed_filename}")
    
    # 为absolute结果生成散点图
    if not df_absolute.empty:
        print("\n" + "="*50)
        print("生成absolute结果的散点图...")
        
        # 创建输出目录
        output_dir = os.environ.get("MLLM_SCATTER_OUT", "outputs/mllm_scatter")

        os.makedirs(output_dir, exist_ok=True)
        
        # 提取模型名称
        model_name = base_dir.split('/')[-1].split('_')[0]
        
        # 收集每个test split和property的原始数据
        test_splits = ['test1', 'test2', 'test3']
        scatter_data = {}  # {test_split: {property: {'gt': [], 'pred': [], 'video_names': []}}}
        
        # 重新读取原始数据来获取详细的predictions和ground_truths
        for filename in sorted(os.listdir(base_dir)):
            if not filename.endswith("_results.txt"):
                continue
            
            file_path = os.path.join(base_dir, filename)
            model_version, test_set, result_type = extract_info_from_filename(filename)
            
            if result_type == 'absolute' and test_set in test_splits:
                result_data = extract_comparison_results(file_path, result_type)
                if result_data:
                    property_name = result_data['property_name']
                    
                    # 对于elasticity属性，过滤掉ground truth小于0.1的数据
                    gt_values = result_data['ground_truths']
                    pred_values = result_data['predictions']
                    video_names = result_data['video_names']
                    
                    if property_name.lower() == 'elasticity':
                        filtered_indices = [i for i, gt in enumerate(gt_values) if gt >= 0.1]
                        if filtered_indices:
                            gt_values = [gt_values[i] for i in filtered_indices]
                            pred_values = [pred_values[i] for i in filtered_indices]
                            video_names = [video_names[i] for i in filtered_indices]
                            print(f"  -> 散点图elasticity过滤: 原始{len(result_data['ground_truths'])}个样本，过滤后{len(gt_values)}个样本 (GT>=0.1)")
                        else:
                            print(f"  -> 散点图elasticity过滤: 所有样本GT都小于0.1，跳过")
                            continue
                    
                    if test_set not in scatter_data:
                        scatter_data[test_set] = {}
                    if property_name not in scatter_data[test_set]:
                        scatter_data[test_set][property_name] = {'gt': [], 'pred': [], 'video_names': []}
                    
                    scatter_data[test_set][property_name]['gt'].extend(gt_values)
                    scatter_data[test_set][property_name]['pred'].extend(pred_values)
                    scatter_data[test_set][property_name]['video_names'].extend(video_names)
        
        # 为每个test split和property创建散点图
        for test_split in test_splits:
            if test_split in scatter_data:
                for property_name, data in scatter_data[test_split].items():
                    # 检查是否需要特殊处理
                    if property_name.lower() == 'viscosity' and test_split == 'test3':
                        # 按液体名称聚合
                        print(f"  -> 对 {property_name} {test_split} 按液体名称聚合...")
                        liquid_data = defaultdict(list)
                        
                        for i, video_name in enumerate(data['video_names']):
                            try:
                                # 从视频名称中解析液体名称
                                parts = video_name.split()
                                if len(parts) > 0:
                                    basename = parts[0].split(':')[1].strip() if ':' in parts[0] else parts[0].strip()
                                    basename_parts = basename.split('_')
                                    if len(basename_parts) <= 4:  # Short basename
                                        liquid_name = basename_parts[0]
                                    else:  # Long basename
                                        liquid_name = basename_parts[-4]
                                    
                                    liquid_data[liquid_name].append((data['pred'][i], data['gt'][i]))
                            except (IndexError, ValueError):
                                # Skip if can't parse properly
                                continue
                        
                        # 聚合数据：对每种液体的predictions取平均
                        aggregated_gt = []
                        aggregated_pred = []
                        for liquid_name, predictions_gts in liquid_data.items():
                            if len(predictions_gts) > 0:
                                pred_values = [pred for pred, gt in predictions_gts]
                                gt_values = [gt for pred, gt in predictions_gts]
                                
                                # predictions取平均，gt使用第一个值（应该都相同）
                                avg_pred = np.mean(pred_values)
                                gt_value = gt_values[0]  # gt应该都一样
                                
                                aggregated_pred.append(avg_pred)
                                aggregated_gt.append(gt_value)
                                print(f"    液体 {liquid_name}: {len(pred_values)} 个样本 -> 平均预测值 {avg_pred:.4f}, GT {gt_value:.4f}")
                        
                        final_gt = aggregated_gt
                        final_pred = aggregated_pred
                        print(f"    聚合后: {len(final_gt)} 个液体")
                        
                    elif property_name.lower() == 'friction' and test_split == 'test3':
                        # 按 object + surface 组合聚合
                        print(f"  -> 对 {property_name} {test_split} 按 object+surface 组合聚合...")
                        combination_data = defaultdict(list)
                        
                        for i, video_name in enumerate(data['video_names']):
                            try:
                                # 从视频名称中解析 object 和 surface 名称
                                parts = video_name.split()
                                if len(parts) > 0:
                                    basename = parts[0].split(':')[1].strip() if ':' in parts[0] else parts[0].strip()
                                    obj_name = basename.split('_')[0]
                                    surface_name = basename.split('_')[1]
                                    
                                    # # Skip eraser objects
                                    # if obj_name == 'eraser':
                                    #     continue
                                    
                                    # # Skip specific combinations
                                    # if (obj_name == 'wooden-box' and surface_name == 'table') or \
                                    #    (obj_name == 'wooden-box' and surface_name == 'towel'):
                                    #     continue
                                    
                                    combination_key = f"{obj_name}-{surface_name}"
                                    combination_data[combination_key].append((data['pred'][i], data['gt'][i]))
                            except (IndexError, ValueError):
                                # Skip if can't parse properly
                                continue
                        
                        # 聚合数据：对每种组合的predictions取平均
                        aggregated_gt = []
                        aggregated_pred = []
                        for combination_key, predictions_gts in combination_data.items():
                            if len(predictions_gts) > 0:
                                pred_values = [pred for pred, gt in predictions_gts]
                                gt_values = [gt for pred, gt in predictions_gts]
                                
                                # predictions取平均，gt使用第一个值（应该都相同）
                                avg_pred = np.mean(pred_values)
                                gt_value = gt_values[0]  # gt应该都一样
                                
                                aggregated_pred.append(avg_pred)
                                aggregated_gt.append(gt_value)
                                print(f"    组合 {combination_key}: {len(pred_values)} 个样本 -> 平均预测值 {avg_pred:.4f}, GT {gt_value:.4f}")
                        
                        final_gt = aggregated_gt
                        final_pred = aggregated_pred
                        print(f"    聚合后: {len(final_gt)} 个组合")
                        
                    else:
                        # 其他情况直接跳过
                        print(f"  -> 跳过 {property_name} {test_split} (不需要特殊处理)")
                        continue
                    
                    # 绘制散点图
                    if len(final_gt) > 0 and len(final_pred) > 0:
                        plt.figure(figsize=(8, 6))
                        plt.scatter(final_gt, final_pred, alpha=0.6)
                        plt.xlabel('Ground Truth')
                        plt.ylabel('Prediction')
                        
                        # 对于viscosity test3，使用对数刻度
                        if property_name.lower() == 'viscosity' and test_split == 'test3':
                            plt.xscale('log')  # 设置横轴为对数刻度
                            plt.yscale('log')  # 设置纵轴为对数刻度
                        
                        # 文件命名格式: model + property + test_i_basename.png
                        test_number = test_split.replace('test', '')
                        plot_filename = f"{model_name}_{property_name}_test_{test_number}_basename.png"
                        plot_path = os.path.join(output_dir, plot_filename)
                        
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"聚合散点图已保存: {plot_path}")
                    else:
                        print(f"  -> 警告: {property_name} {test_split} 聚合后没有有效数据")
    
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