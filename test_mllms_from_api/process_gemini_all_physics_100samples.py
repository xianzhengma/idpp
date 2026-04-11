#!/usr/bin/env python3
"""
处理Gemini消融实验结果：100samples的所有物理属性（friction, viscosity, elasticity）
分别处理absolute和relative任务，将100个样本分成5份进行分析
为每种物理属性生成独立的对比表格
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
    
    # 映射目录名到策略名称
    experiment_mapping = {
        'gemini_physics_analysis_results_absolute_100samples': 'Baseline',
        'gemini_physics_analysis_results_relative_100samples': 'Baseline',
        'gemini_physics_analysis_results_absolute_100samples_frameindex': '+ Frame Index Provided',
        'gemini_physics_analysis_results_relative_100samples_frameindex': '+ Frame Index Provided',
        'gemini_physics_analysis_results_absolute_100samples_icl': '+ Few-Shot Examples',
        'gemini_physics_analysis_results_relative_100samples_icl': '+ Few-Shot Examples',
        'gemini_physics_analysis_results_absolute_100samples_oracle': '+ Oracle Estimation Teaching',
        'gemini_physics_analysis_results_relative_100samples_oracle': '+ Oracle Estimation Teaching',
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

        # 确保我们有足够的样本
        if len(predictions) == 0:
            print(f"警告: 文件 {file_path} 中没有找到有效样本")
            return None
        
        # 分成5个chunk，每个20个样本（如果样本不足100个，就按比例分配）
        total_samples = len(predictions)
        chunk_size = min(20, total_samples // 5) if total_samples >= 5 else total_samples
        
        chunks = []
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            
            if start_idx < total_samples:
                chunk_preds = predictions[start_idx:end_idx]
                chunk_gt = ground_truths[start_idx:end_idx]
                
                if len(chunk_preds) > 0:  # 确保chunk不为空
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
    print(f"We conduct the study on Gemini 2.5 Pro for the {task_name.lower()} task.")
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


def process_physics_property(all_results, property_name, result_type):
    """
    处理特定物理属性的结果
    """
    # 筛选特定物理属性的结果
    property_results = [r for r in all_results if r['Property'] == property_name]
    
    if not property_results:
        print(f"\n⚠️  未找到 {property_name} 的实验结果")
        return None
    
    print(f"\n{'='*80}")
    print(f"分析 {property_name.upper()} 物理属性 ({'相对比较' if result_type == 'relative' else '绝对值预测'})")
    print(f"找到 {len(property_results)} 个结果记录")
    print(f"{'='*80}")
    
    # 创建DataFrame
    df_property = pd.DataFrame(property_results)
    
    # 计算所有chunk的整体平均表现
    overall_results = []
    
    # 按Strategy和Test_Set分组，计算所有chunk的平均值
    for strategy in df_property['Strategy'].unique():
        for test_set in df_property['Test_Set'].unique():
            subset = df_property[(df_property['Strategy'] == strategy) & 
                               (df_property['Test_Set'] == test_set)]
            
            if not subset.empty:
                if result_type == 'relative':
                    avg_auc = subset['AUC'].mean()
                    overall_results.append({
                        'Strategy': strategy,
                        'Test_Set': test_set,
                        'Property': property_name,
                        'AUC': avg_auc
                    })
                else:
                    avg_corr = subset['Pearson_Correlation'].mean()
                    overall_results.append({
                        'Strategy': strategy,
                        'Test_Set': test_set,
                        'Property': property_name,
                        'Pearson_Correlation': avg_corr
                    })
    
    if overall_results:
        df_overall = pd.DataFrame(overall_results)
        
        # 创建对比表格
        task_name = f"{property_name.capitalize()} ({'Relative' if result_type == 'relative' else 'Absolute'})"
        pivot_table = create_comparison_table(df_overall, result_type, task_name)
        
        # 分析最佳策略
        if not pivot_table.empty:
            best_strategy = pivot_table.index[0]
            best_score = pivot_table.loc[best_strategy, 'Avg']
            baseline_score = pivot_table.loc['Baseline', 'Avg'] if 'Baseline' in pivot_table.index else None
            
            print(f"\n🏆 {property_name.capitalize()} 最佳策略: {best_strategy}")
            main_metric = 'AUC' if result_type == 'relative' else 'Pearson Correlation'
            print(f"   平均{main_metric}: {best_score:.3f}")
            
            if baseline_score is not None and best_strategy != 'Baseline':
                if result_type == 'relative':
                    improvement = ((best_score - baseline_score) / baseline_score) * 100
                else:
                    improvement = ((best_score - baseline_score) / max(abs(baseline_score), 0.001)) * 100
                print(f"   相比Baseline提升: {improvement:+.1f}%")
        
        return pivot_table
    
    return None


def main():
    """
    主函数：处理所有gemini 100samples实验，按物理属性分类
    """
    print("开始处理Gemini消融实验 (100samples) - 所有物理属性分析...")
    
    # 找到所有相关的实验目录
    absolute_patterns = ["gemini_physics_analysis_results_absolute_100samples*"]
    relative_patterns = ["gemini_physics_analysis_results_relative_100samples*"]
    
    all_dirs = []
    for pattern in absolute_patterns + relative_patterns:
        all_dirs.extend(glob.glob(pattern))
    
    print(f"发现 {len(all_dirs)} 个实验目录")
    
    if not all_dirs:
        print("未找到任何实验目录，请检查路径")
        return
    
    # 处理所有实验
    all_absolute_results = []
    all_relative_results = []
    
    for base_dir in sorted(all_dirs):
        if not os.path.isdir(base_dir):
            continue
            
        experiment_name = get_experiment_name_from_dir(base_dir)
        print(f"\n处理实验: {experiment_name} ({base_dir})")
        
        results_files = [f for f in os.listdir(base_dir) 
                        if f.endswith("_results.txt") and 
                        any(task_type in f for task_type in ['absolute', 'relative'])]
        
        for filename in sorted(results_files):
            file_path = os.path.join(base_dir, filename)
            model_version, test_set, result_type = extract_info_from_filename(filename)
            
            if not all([model_version, test_set, result_type]):
                continue
                
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
                        
                        if result_type == 'absolute':
                            all_absolute_results.append(result_row)
                        else:
                            all_relative_results.append(result_row)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理绝对值预测任务
    if all_absolute_results:
        print(f"\n{'='*100}")
        print("处理绝对值预测任务 (ABSOLUTE PREDICTION TASKS)")
        print(f"{'='*100}")
        
        # 按物理属性分别处理
        properties = list(set([r['Property'] for r in all_absolute_results]))
        absolute_tables = {}
        
        for prop in sorted(properties):
            table = process_physics_property(all_absolute_results, prop, 'absolute')
            if table is not None:
                absolute_tables[prop] = table
                
                # 保存每个属性的表格
                prop_filename = f"gemini_{prop}_absolute_100samples_table_{timestamp}.csv"
                table.to_csv(prop_filename)
                print(f"📁 {prop.capitalize()} 绝对值预测表格已保存为: {prop_filename}")
    
    # 处理相对比较任务
    if all_relative_results:
        print(f"\n{'='*100}")
        print("处理相对比较任务 (RELATIVE COMPARISON TASKS)")
        print(f"{'='*100}")
        
        # 按物理属性分别处理
        properties = list(set([r['Property'] for r in all_relative_results]))
        relative_tables = {}
        
        for prop in sorted(properties):
            table = process_physics_property(all_relative_results, prop, 'relative')
            if table is not None:
                relative_tables[prop] = table
                
                # 保存每个属性的表格
                prop_filename = f"gemini_{prop}_relative_100samples_table_{timestamp}.csv"
                table.to_csv(prop_filename)
                print(f"📁 {prop.capitalize()} 相对比较表格已保存为: {prop_filename}")
    
    # 保存详细结果
    if all_absolute_results:
        absolute_df = pd.DataFrame(all_absolute_results)
        absolute_filename = f"gemini_all_physics_absolute_100samples_detailed_{timestamp}.csv"
        absolute_df.to_csv(absolute_filename, index=False)
        print(f"\n📁 所有绝对值实验详细结果已保存为: {absolute_filename}")
    
    if all_relative_results:
        relative_df = pd.DataFrame(all_relative_results)
        relative_filename = f"gemini_all_physics_relative_100samples_detailed_{timestamp}.csv" 
        relative_df.to_csv(relative_filename, index=False)
        print(f"📁 所有相对比较实验详细结果已保存为: {relative_filename}")
    
    # 总结
    print(f"\n{'='*100}")
    print("✅ 所有物理属性消融实验分析完成！")
    print(f"{'='*100}")
    
    if all_absolute_results:
        abs_properties = list(set([r['Property'] for r in all_absolute_results]))
        print(f"📊 绝对值预测任务包含: {', '.join(sorted(abs_properties))}")
    
    if all_relative_results:
        rel_properties = list(set([r['Property'] for r in all_relative_results]))
        print(f"📊 相对比较任务包含: {', '.join(sorted(rel_properties))}")
    
    print(f"📁 共生成 {len(absolute_tables) if 'absolute_tables' in locals() else 0} 个绝对值预测表格")
    print(f"📁 共生成 {len(relative_tables) if 'relative_tables' in locals() else 0} 个相对比较表格")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()