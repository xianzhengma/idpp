#!/usr/bin/env python3
"""
处理多个实验目录的结果文件，计算评价指标并生成对比表格
支持相对(relative)和绝对(absolute)两种测试结果
自动发现和处理所有gemini_physics_analysis_results*目录
支持elasticity专用实验目录格式
"""

import os
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr
import glob

def extract_info_from_filename(filename):
    """
    从文件名中提取模型版本、测试集和测试类型
    """
    match = re.match(r'(\w+)_vjepa_(test\d+)_(relative|absolute)_results\.txt', filename)
    if match:
        return match.groups()
    return None, None, None

def get_experiment_name_from_dir(dir_path):
    """
    从目录名中提取实验名称，支持新的elasticity实验格式
    """
    dir_name = os.path.basename(dir_path)
    
    # 处理新的elasticity实验格式
    # 格式：gemini_physics_analysis_results_elasticity_relative_50samples[_features]
    if dir_name.startswith('gemini_physics_analysis_results_elasticity_relative_50samples'):
        if dir_name == 'gemini_physics_analysis_results_elasticity_relative_50samples':
            return 'Baseline'
        else:
            # 提取features部分
            features_part = dir_name.replace('gemini_physics_analysis_results_elasticity_relative_50samples_', '')
            
            # 映射features到友好的名称
            feature_mapping = {
                'concat': '+ Concat',
                'frameindex': '+ FrameIndex',
                'icl': '+ ICL',
                'oracle': '+ Oracle',
                'concat_frameindex': '+ Concat+FrameIndex',
                'concat_frameindex_icl': '+ Concat+FrameIndex+ICL',
                'concat_frameindex_icl_oracle': '+ Concat+FrameIndex+ICL+Oracle',
                'concat_icl': '+ Concat+ICL',
                'concat_oracle': '+ Concat+Oracle',
                'frameindex_icl': '+ FrameIndex+ICL',
                'frameindex_oracle': '+ FrameIndex+Oracle',
                'icl_oracle': '+ ICL+Oracle',
            }
            
            return feature_mapping.get(features_part, f'+ {features_part.replace("_", "+").title()}')
    
    # 原有的实验名称映射（保持向后兼容）
    experiment_mapping = {
        'gemini_physics_analysis_results': 'Baseline',
        'gemini_physics_analysis_results_icl-example': '+ ICL',
        'gemini_physics_analysis_results_icl': '+ ICL',
        'gemini_physics_analysis_results_follow-oracle-test': '+ Oracle',
        'gemini_physics_analysis_results_oracle': '+ Oracle',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test': '+ ICL+Oracle',
        'gemini_physics_analysis_results_icl_oracle': '+ ICL+Oracle',
        'gemini_physics_analysis_results_concat-mode': '+ Concat',
        'gemini_physics_analysis_results_concat': '+ Concat',
        'gemini_physics_analysis_results_frame-index': '+ FrameIndex',
        'gemini_physics_analysis_results_frameindex': '+ FrameIndex',
        'gemini_physics_analysis_results_icl-example_concat-mode': '+ ICL+Concat',
        'gemini_physics_analysis_results_concat_icl': '+ ICL+Concat',
        'gemini_physics_analysis_results_follow-oracle-test_concat-mode': '+ Oracle+Concat',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index': '+ ICL+Oracle+FrameIndex',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index': '+ AllFeatures'
    }
    
    return experiment_mapping.get(dir_name, dir_name)

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
            comparisons = re.findall(r'预测结果: (\d+)\s+GT结果: (\d+)', content)
            if not comparisons:
                print(f"未找到比较结果在文件: {file_path}")
                return None
            predictions = [int(p) for p, _ in comparisons]
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

def calculate_metrics(y_true, y_pred, result_type):
    """
    根据结果类型计算评估指标
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
            if len(y_true) < 2:
                metrics['Pearson_Correlation'] = 0.0
                metrics['MAE'] = 0.0
            else:
                correlation, _ = pearsonr(y_true, y_pred)
                metrics['Pearson_Correlation'] = correlation
                metrics['MAE'] = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return metrics
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None

def main():
    # 只处理elasticity_relative_50samples系列的实验目录
    pattern = "gemini_physics_analysis_results_elasticity_relative_50samples*"
    base_dirs = glob.glob(pattern)
    
    if not base_dirs:
        print(f"未找到匹配的目录: {pattern}")
        return
    
    base_dirs.sort()  # 按字母顺序排序
    
    all_results = []
    
    print(f"发现 {len(base_dirs)} 个 Elasticity 实验目录:")
    for dir_path in base_dirs:
        experiment_name = get_experiment_name_from_dir(dir_path)
        print(f"  - {dir_path} -> {experiment_name}")
    print()
    
    # 处理每个实验目录
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
            
        experiment_name = get_experiment_name_from_dir(base_dir)
        print(f"处理实验: {experiment_name} ({base_dir})")
        
        results_files = [f for f in os.listdir(base_dir) if f.endswith("_results.txt")]
        
        if not results_files:
            print(f"  警告: 目录 {base_dir} 中没有找到结果文件")
            continue
            
        for filename in sorted(results_files):
            file_path = os.path.join(base_dir, filename)
            model_version, test_set, result_type = extract_info_from_filename(filename)
            
            if not all([model_version, test_set, result_type]):
                print(f"  跳过不匹配的文件: {filename}")
                continue
                
            print(f"  处理文件: {filename}")
            
            result_data = extract_comparison_results(file_path, result_type)
            
            if result_data:
                metrics = calculate_metrics(result_data['ground_truths'], result_data['predictions'], result_type)
                
                if metrics:
                    row = {
                        'Experiment': experiment_name,
                        'Model': f"{model_version}_vjepa",
                        'Property': result_data['property_name'],
                        'Test_Set': test_set,
                        'Type': result_type,
                        'Num_Samples': len(result_data['predictions']),
                    }
                    row.update(metrics)
                    all_results.append(row)
                    
                    metrics_str = ", ".join([f"{k} = {v:.4f}" for k, v in metrics.items()])
                    print(f"    -> {row['Property']} - {row['Test_Set']} ({row['Type']}): {metrics_str}")
        print()

    if not all_results:
        print("没有处理任何结果，退出。")
        return

    df = pd.DataFrame(all_results)
    
    # 只处理相对比较结果（因为这些都是相对比较实验）
    df_relative = df[df['Type'] == 'relative']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成专门的elasticity对比表格
    print("\n" + "="*80)
    print("生成 Elasticity 实验对比表格...")

    if not df_relative.empty:
        print("\nElasticity 相对比较测试结果 (AUC) - 专用对比表格:")
        print("-" * 70)
            
            # 创建实验×测试集的透视表
        pivot_elasticity = df_relative.pivot_table(
                index='Experiment', 
                columns='Test_Set', 
                values='AUC',
                aggfunc='mean'
            )
        pivot_elasticity = pivot_elasticity.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_elasticity.columns = ['Test-1', 'Test-2', 'Test-3']
            
            # 计算平均值
        pivot_elasticity['Average'] = pivot_elasticity.mean(axis=1)
        pivot_elasticity = pivot_elasticity.round(4)
            
        # 按平均值排序（降序）
        pivot_elasticity = pivot_elasticity.sort_values('Average', ascending=False)
            
        # 美化显示
        print(pivot_elasticity.to_string())
            
        # 保存专门的elasticity CSV文件
        elasticity_filename = f"elasticity_relative_50samples_comparison_{timestamp}.csv"
        pivot_elasticity.to_csv(elasticity_filename)
        print(f"\n✓ Elasticity 实验对比表格已保存为: {elasticity_filename}")

        # 显示性能提升分析
        baseline_avg = pivot_elasticity.loc['Baseline', 'Average'] if 'Baseline' in pivot_elasticity.index else None
        print(f"\n📊 性能分析:")
        print(f"{'实验配置':<35} {'平均AUC':<10} {'相比Baseline':<12}")
        print("-" * 60)
        
        for exp_name in pivot_elasticity.index:
            avg_auc = pivot_elasticity.loc[exp_name, 'Average']
            if baseline_avg and exp_name != 'Baseline':
                improvement = ((avg_auc - baseline_avg) / baseline_avg) * 100
                improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            else:
                improvement_str = "Baseline"
            print(f"{exp_name:<35} {avg_auc:<10.4f} {improvement_str:<12}")
        
        # 显示最佳配置
        best_experiment = pivot_elasticity.index[0]  # 第一行是最高分数
        best_average = pivot_elasticity.loc[best_experiment, 'Average']
        print(f"\n🏆 最佳实验配置: {best_experiment}")
        print(f"   平均 AUC: {best_average:.4f}")
        
        if baseline_avg:
            improvement = ((best_average - baseline_avg) / baseline_avg) * 100
            print(f"   相比Baseline提升: {improvement:.1f}%")
        
        print("\n" + "="*70)

    # 保存详细结果
    detailed_filename = f"detailed_elasticity_experiments_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False)
    print(f"\n📁 所有详细结果已保存为: {detailed_filename}")
    
    print("\n" + "="*80)
    print("处理完成总结:")
    print(f"- 处理了 {len(base_dirs)} 个 Elasticity 实验目录")
    print(f"- 总共处理了 {len(df)} 个结果文件")
    print(f"- 所有实验都是 elasticity 相对比较测试")
    print(f"- 平均 AUC: {df_relative['AUC'].mean():.4f}")
    
    best_relative = df_relative.groupby('Experiment')['AUC'].mean().idxmax()
    print(f"- 最佳实验: {best_relative}")
    print("="*80)


if __name__ == "__main__":
    main() 