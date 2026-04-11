#!/usr/bin/env python3
"""
处理Gemini消融实验结果：100samples的absolute和relative任务
将100个样本分成5份（每份20个），计算每份的评估指标
生成类似论文表格样式的对比结果
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from datetime import datetime
import glob


def extract_info_from_filename(filename):
    """
    从文件名中提取模型版本、测试集和测试类型
    例如：v125_vjepa_test1_relative_results.txt
    """
    match = re.match(r'(v\d+)_vjepa_(test\d+)_(relative|absolute)_results\.txt', filename)
    if match:
        return match.groups()
    return None, None, None


def get_experiment_name_from_dir(dir_path):
    """
    从目录名中提取实验名称并映射到友好的策略名称
    """
    dir_name = os.path.basename(dir_path)
    
    # 映射目录名到策略名称（根据你的实际目录结构调整）
    experiment_mapping = {
        'gemini_physics_analysis_results_absolute_100samples': 'Baseline',
        'gemini_physics_analysis_results_relative_100samples': 'Baseline',
        'gemini_physics_analysis_results_absolute_100samples_frameindex': '+ Frame Index Provided',
        'gemini_physics_analysis_results_relative_100samples_frameindex': '+ Frame Index Provided',
        'gemini_physics_analysis_results_absolute_100samples_icl': '+ Few-Shot Examples',
        'gemini_physics_analysis_results_relative_100samples_icl': '+ Few-Shot Examples',
        'gemini_physics_analysis_results_absolute_100samples_oracle': '+ Oracle Estimation Teaching',
        'gemini_physics_analysis_results_relative_100samples_oracle': '+ Oracle Estimation Teaching',
        'gemini_physics_analysis_results_absolute_100samples_concat': '+ Black Frames in Between',
        'gemini_physics_analysis_results_relative_100samples_concat': '+ Black Frames in Between',
    }
    
    return experiment_mapping.get(dir_name, dir_name)


def extract_results_from_file(file_path, result_type, chunk_size=20):
    """
    从结果文件中提取预测结果和真实结果，并分成chunks
    返回5个chunk的结果
    对于elasticity的绝对值预测任务，过滤掉GT值小于0.1的样本
    相对比较任务不进行GT值过滤
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取属性名
        property_match = re.search(r'属性: (\w+)', content)
        if not property_match:
            print(f"无法从文件 {file_path} 中提取属性")
            return None
        property_name = property_match.group(1)
        
        # 根据结果类型提取数据
        if result_type == 'relative':
            # 相对比较：预测结果: 1.0, GT结果: 1
            # 同时需要提取GT值用于过滤
            pred_gt_pattern = r'预测结果: ([\d.-]+)\s+GT结果: (\d+)\s+GT值: ([\d.-]+(?:e-?\d+)?)\s+vs\s+([\d.-]+(?:e-?\d+)?)'
            matches = re.findall(pred_gt_pattern, content)
            
            if not matches:
                print(f"未找到完整的比较结果在文件: {file_path}")
                return None
            
            predictions = []
            ground_truths = []
            original_count = len(matches)
            
            for pred, gt_label, gt_val1, gt_val2 in matches:
                gt_val1_float = float(gt_val1)
                gt_val2_float = float(gt_val2)
                
                # 对于elasticity的相对比较，不需要过滤GT值
                # 相对比较关注的是两个值的相对大小关系，而不是绝对值
                
                predictions.append(float(pred))
                ground_truths.append(int(gt_label))
            
            # 相对比较任务不进行GT值过滤
                
        else: # absolute
            # 绝对值预测：预测值: 4000.0, GT值: 5.039669168531562e-05
            comparisons = re.findall(r'预测值: ([\d.-]+)\s+GT值: ([\d.-]+(?:e-?\d+)?)', content)
            if not comparisons:
                print(f"未找到比较结果在文件: {file_path}")
                return None
            
            predictions = []
            ground_truths = []
            original_count = len(comparisons)
            
            for pred, gt_val in comparisons:
                gt_val_float = float(gt_val)
                
                # 对于elasticity的绝对值预测，过滤掉GT值小于0.1的样本
                if property_name == 'elasticity':
                    if gt_val_float < 0.1:
                        continue
                
                predictions.append(float(pred))
                ground_truths.append(gt_val_float)
            
            if property_name == 'elasticity' and len(predictions) < original_count:
                filtered_count = original_count - len(predictions)
                print(f"    Elasticity绝对值过滤: 移除了{filtered_count}个GT值<0.1的样本")

        # 确保我们有100个样本
        if len(predictions) != 100:
            print(f"警告: 文件 {file_path} 中只有 {len(predictions)} 个样本，不是预期的100个")
        
        # 分成5个chunk，每个20个样本
        chunks = []
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(predictions))
            
            if start_idx < len(predictions):
                chunk_preds = predictions[start_idx:end_idx]
                chunk_gt = ground_truths[start_idx:end_idx]
                chunks.append({
                    'property_name': property_name,
                    'predictions': chunk_preds,
                    'ground_truths': chunk_gt,
                    'chunk_id': i + 1
                })
        
        return chunks
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None


def calculate_metrics(y_true, y_pred, result_type):
    """
    根据结果类型计算评估指标
    """
    try:
        if result_type == 'relative':
            # 对于相对比较，计算AUC
            if len(set(y_true)) < 2:
                return {'AUC': 0.5}
            else:
                # 将连续预测值转换为二分类概率
                y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
                accuracy = sum(1 for p, g in zip(y_pred_binary, y_true) if p == g) / len(y_true)
                
                # 如果预测是概率值，直接用于AUC计算
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except:
                    auc = roc_auc_score(y_true, y_pred_binary)
                
                return {'AUC': auc, 'Accuracy': accuracy}
        else: # absolute
            # 对于绝对值预测，计算相关系数
            if len(y_true) < 2:
                return {'Pearson_Correlation': 0.0}
            else:
                correlation, _ = pearsonr(y_true, y_pred)
                if np.isnan(correlation):
                    correlation = 0.0
                return {'Pearson_Correlation': correlation}
                
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None


def create_comparison_table(results_df, result_type, task_name):
    """
    创建类似论文中的对比表格
    """
    print(f"\n{'='*80}")
    print(f"Table: {task_name} prediction results for different MLLM prompting strategies")
    print(f"{'='*80}")
    
    # 选择主要指标
    if result_type == 'relative':
        main_metric = 'AUC'
    else:
        main_metric = 'Pearson_Correlation'
    
    # 创建透视表：Strategy × Test_Set
    pivot_table = results_df.pivot_table(
        index='Strategy', 
        columns='Test_Set', 
        values=main_metric,
        aggfunc='mean'
    )
    
    # 确保列的顺序
    expected_columns = ['test1', 'test2', 'test3']
    available_columns = [col for col in expected_columns if col in pivot_table.columns]
    pivot_table = pivot_table.reindex(columns=available_columns, fill_value=np.nan)
    
    # 重命名列
    column_mapping = {'test1': 'Test-1', 'test2': 'Test-2', 'test3': 'Test-3'}
    pivot_table.columns = [column_mapping.get(col, col) for col in pivot_table.columns]
    
    # 计算平均值
    pivot_table['Avg'] = pivot_table.mean(axis=1)
    
    # 按平均值排序（降序）
    pivot_table = pivot_table.sort_values('Avg', ascending=False)
    
    # 格式化数字（保留2位小数）
    pivot_table = pivot_table.round(2)
    
    # 显示表格
    print(f"\nStrategy{' '*(35-len('Strategy'))} | {'  '.join([f'{col:^8}' for col in pivot_table.columns])}")
    print("-" * 80)
    
    for strategy in pivot_table.index:
        row_data = [f"{pivot_table.loc[strategy, col]:.2f}" if not pd.isna(pivot_table.loc[strategy, col]) else "N/A" 
                   for col in pivot_table.columns]
        strategy_str = strategy[:35] + "..." if len(strategy) > 35 else strategy
        print(f"{strategy_str:<35} | {'  '.join([f'{val:>8}' for val in row_data])}")
    
    return pivot_table


def process_chunk_results(all_chunk_results, result_type):
    """
    处理所有chunk的结果，为每个chunk生成独立的表格
    """
    print(f"\n{'='*80}")
    print(f"分析结果：将100个样本分成5份（每份20个）进行独立评估")
    print(f"任务类型: {'相对比较' if result_type == 'relative' else '绝对值预测'}")
    print(f"{'='*80}")
    
    # 按chunk分组
    for chunk_id in range(1, 6):
        chunk_data = [result for result in all_chunk_results if result['chunk_id'] == chunk_id]
        
        if not chunk_data:
            continue
            
        print(f"\n第 {chunk_id} 份数据 (样本 {(chunk_id-1)*20+1}-{chunk_id*20}):")
        print("-" * 50)
        
        # 创建该chunk的DataFrame
        df_chunk = pd.DataFrame(chunk_data)
        
        # 创建对比表格
        task_name = f"Chunk {chunk_id} ({'Relative' if result_type == 'relative' else 'Absolute'})"
        pivot_table = create_comparison_table(df_chunk, result_type, task_name)
        
        # 分析最佳策略
        best_strategy = pivot_table.index[0]
        best_score = pivot_table.loc[best_strategy, 'Avg']
        baseline_score = pivot_table.loc['Baseline', 'Avg'] if 'Baseline' in pivot_table.index else None
        
        print(f"\n🏆 第{chunk_id}份最佳策略: {best_strategy}")
        main_metric = 'AUC' if result_type == 'relative' else 'Pearson Correlation'
        print(f"   平均{main_metric}: {best_score:.3f}")
        
        if baseline_score is not None and best_strategy != 'Baseline':
            if result_type == 'relative':
                improvement = ((best_score - baseline_score) / baseline_score) * 100
            else:
                improvement = ((best_score - baseline_score) / max(abs(baseline_score), 0.001)) * 100
            print(f"   相比Baseline提升: {improvement:+.1f}%")


def main():
    """
    主函数：处理所有gemini 100samples实验
    """
    print("开始处理Gemini消融实验 (100samples)...")
    
    # 找到所有相关的实验目录
    absolute_pattern = "gemini_physics_analysis_results_absolute_100samples*"
    relative_pattern = "gemini_physics_analysis_results_relative_100samples*"
    
    absolute_dirs = glob.glob(absolute_pattern)
    relative_dirs = glob.glob(relative_pattern)
    
    print(f"发现 {len(absolute_dirs)} 个绝对值实验目录")
    print(f"发现 {len(relative_dirs)} 个相对比较实验目录")
    
    if not absolute_dirs and not relative_dirs:
        print("未找到任何实验目录，请检查路径")
        return
    
    # 处理绝对值实验
    if absolute_dirs:
        print(f"\n{'='*80}")
        print("处理绝对值预测实验...")
        print(f"{'='*80}")
        
        all_absolute_chunk_results = []
        
        for base_dir in sorted(absolute_dirs):
            if not os.path.isdir(base_dir):
                continue
                
            experiment_name = get_experiment_name_from_dir(base_dir)
            print(f"\n处理实验: {experiment_name} ({base_dir})")
            
            results_files = [f for f in os.listdir(base_dir) if f.endswith("_absolute_results.txt")]
            
            for filename in sorted(results_files):
                file_path = os.path.join(base_dir, filename)
                model_version, test_set, result_type = extract_info_from_filename(filename)
                
                if not all([model_version, test_set, result_type]):
                    continue
                    
                print(f"  处理文件: {filename}")
                
                chunks = extract_results_from_file(file_path, result_type)
                
                if chunks:
                    for chunk in chunks:
                        metrics = calculate_metrics(chunk['ground_truths'], chunk['predictions'], result_type)
                        
                        if metrics:
                            result_row = {
                                'Strategy': experiment_name,
                                'Model': f"{model_version}_vjepa",
                                'Property': chunk['property_name'],
                                'Test_Set': test_set,
                                'Type': result_type,
                                'chunk_id': chunk['chunk_id'],
                                'Num_Samples': len(chunk['predictions']),
                            }
                            result_row.update(metrics)
                            all_absolute_chunk_results.append(result_row)
        
        # 处理绝对值实验的chunk结果
        if all_absolute_chunk_results:
            process_chunk_results(all_absolute_chunk_results, 'absolute')
    
    # 处理相对比较实验
    if relative_dirs:
        print(f"\n{'='*80}")
        print("处理相对比较实验...")
        print(f"{'='*80}")
        
        all_relative_chunk_results = []
        
        for base_dir in sorted(relative_dirs):
            if not os.path.isdir(base_dir):
                continue
                
            experiment_name = get_experiment_name_from_dir(base_dir)
            print(f"\n处理实验: {experiment_name} ({base_dir})")
            
            results_files = [f for f in os.listdir(base_dir) if f.endswith("_relative_results.txt")]
            
            for filename in sorted(results_files):
                file_path = os.path.join(base_dir, filename)
                model_version, test_set, result_type = extract_info_from_filename(filename)
                
                if not all([model_version, test_set, result_type]):
                    continue
                    
                print(f"  处理文件: {filename}")
                
                chunks = extract_results_from_file(file_path, result_type)
                
                if chunks:
                    for chunk in chunks:
                        metrics = calculate_metrics(chunk['ground_truths'], chunk['predictions'], result_type)
                        
                        if metrics:
                            result_row = {
                                'Strategy': experiment_name,
                                'Model': f"{model_version}_vjepa",
                                'Property': chunk['property_name'],
                                'Test_Set': test_set,
                                'Type': result_type,
                                'chunk_id': chunk['chunk_id'],
                                'Num_Samples': len(chunk['predictions']),
                            }
                            result_row.update(metrics)
                            all_relative_chunk_results.append(result_row)
        
        # 处理相对比较实验的chunk结果
        if all_relative_chunk_results:
            process_chunk_results(all_relative_chunk_results, 'relative')
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'all_absolute_chunk_results' in locals() and all_absolute_chunk_results:
        absolute_df = pd.DataFrame(all_absolute_chunk_results)
        absolute_filename = f"gemini_absolute_100samples_chunk_analysis_{timestamp}.csv"
        absolute_df.to_csv(absolute_filename, index=False)
        print(f"\n📁 绝对值实验详细结果已保存为: {absolute_filename}")
    
    if 'all_relative_chunk_results' in locals() and all_relative_chunk_results:
        relative_df = pd.DataFrame(all_relative_chunk_results)
        relative_filename = f"gemini_relative_100samples_chunk_analysis_{timestamp}.csv" 
        relative_df.to_csv(relative_filename, index=False)
        print(f"📁 相对比较实验详细结果已保存为: {relative_filename}")
    
    print(f"\n{'='*80}")
    print("✅ Gemini消融实验分析完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()