import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from datetime import datetime
import glob

def extract_info_from_filename(filename):
    match = re.match('(v\\d+)_vjepa_(test\\d+)_(relative|absolute)_results\\.txt', filename)
    if match:
        return match.groups()
    return (None, None, None)

def get_experiment_name_from_dir(dir_path):
    dir_name = os.path.basename(dir_path)
    experiment_mapping = {'gemini_physics_analysis_results_absolute_100samples': 'Baseline', 'gemini_physics_analysis_results_relative_100samples': 'Baseline', 'gemini_physics_analysis_results_absolute_100samples_frameindex': '+ Frame Index Provided', 'gemini_physics_analysis_results_relative_100samples_frameindex': '+ Frame Index Provided', 'gemini_physics_analysis_results_absolute_100samples_icl': '+ Few-Shot Examples', 'gemini_physics_analysis_results_relative_100samples_icl': '+ Few-Shot Examples', 'gemini_physics_analysis_results_absolute_100samples_oracle': '+ Oracle Estimation Teaching', 'gemini_physics_analysis_results_relative_100samples_oracle': '+ Oracle Estimation Teaching', 'gemini_physics_analysis_results_relative_100samples_concat': '+ Black Frames in Between'}
    return experiment_mapping.get(dir_name, dir_name)

def extract_results_from_file(file_path, result_type, chunk_size=20):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        property_match = re.search('property:(\\w+)', content)
        if not property_match:
            print(f'Unable to retrieve file{file_path}Extract attributes from')
            return None
        property_name = property_match.group(1)
        if result_type == 'relative':
            pred_gt_pattern = 'Predicted results:([\\d.-]+)\\s+GT results:(\\d+)\\s+GT value:([\\d.-]+(?:e-?\\d+)?)\\s+vs\\s+([\\d.-]+(?:e-?\\d+)?)'
            matches = re.findall(pred_gt_pattern, content)
            if not matches:
                print(f'Complete comparison results not found in file:{file_path}')
                return None
            predictions = []
            ground_truths = []
            original_count = len(matches)
            for pred, gt_label, gt_val1, gt_val2 in matches:
                gt_val1_float = float(gt_val1)
                gt_val2_float = float(gt_val2)
                predictions.append(float(pred))
                ground_truths.append(int(gt_label))
        else:
            comparisons = re.findall('Predicted value:([\\d.-]+)\\s+GT value:([\\d.-]+(?:e-?\\d+)?)', content)
            if not comparisons:
                print(f'No comparison results found in file:{file_path}')
                return None
            predictions = []
            ground_truths = []
            original_count = len(comparisons)
            for pred, gt_val in comparisons:
                gt_val_float = float(gt_val)
                if property_name == 'elasticity':
                    if gt_val_float < 0.1:
                        continue
                predictions.append(float(pred))
                ground_truths.append(gt_val_float)
            if property_name == 'elasticity' and len(predictions) < original_count:
                filtered_count = original_count - len(predictions)
                print(f'Elasticity absolute value filtering: removed{filtered_count}samples with GT value <0.1')
        if len(predictions) == 0:
            print(f'Warning: File{file_path}No valid sample found in')
            return None
        total_samples = len(predictions)
        chunk_size = min(20, total_samples // 5) if total_samples >= 5 else total_samples
        chunks = []
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            if start_idx < total_samples:
                chunk_preds = predictions[start_idx:end_idx]
                chunk_gt = ground_truths[start_idx:end_idx]
                if len(chunk_preds) > 0:
                    chunks.append({'property_name': property_name, 'predictions': chunk_preds, 'ground_truths': chunk_gt, 'chunk_id': i + 1})
        return chunks
except Exception as e:
        print(f'Process files{file_path}An error occurred:{str(e)}')
        return None

def calculate_metrics(y_true, y_pred, result_type):
    try:
        if result_type == 'relative':
            if len(set(y_true)) < 2:
                return {'AUC': 0.5}
            else:
                y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
                accuracy = sum((1 for p, g in zip(y_pred_binary, y_true) if p == g)) / len(y_true)
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except:
                    auc = roc_auc_score(y_true, y_pred_binary)
                return {'AUC': auc, 'Accuracy': accuracy}
        elif len(y_true) < 2:
            return {'Pearson_Correlation': 0.0}
        else:
            correlation, _ = pearsonr(y_true, y_pred)
            if np.isnan(correlation):
                correlation=0.0
            return {'Pearson_Correlation': correlation}
    except Exception as e:
        print(f'An error occurred while calculating the indicator:{str(e)}')
        return None

def create_comparison_table(results_df, result_type, task_name):
    print(f"\n{'=' * 80}")
    print(f'Table: {task_name} prediction results for different MLLM prompting strategies')
    print(f'We conduct the study on Gemini 2.5 Pro for the {task_name.lower()} task.')
    print(f"{'=' * 80}")
    if result_type == 'relative':
        main_metric = 'AUC'
    else:
        main_metric = 'Pearson_Correlation'
    pivot_table = results_df.pivot_table(index='Strategy', columns='Test_Set', values=main_metric, aggfunc='mean')
    expected_columns = ['test1', 'test2', 'test3']
    available_columns = [col for col in expected_columns if col in pivot_table.columns]
    pivot_table = pivot_table.reindex(columns=available_columns, fill_value=np.nan)
    column_mapping = {'test1': 'Test-1', 'test2': 'Test-2', 'test3': 'Test-3'}
    pivot_table.columns = [column_mapping.get(col, col) for col in pivot_table.columns]
    pivot_table['Avg'] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values('Avg', ascending=False)
    pivot_table = pivot_table.round(2)
    print(f"\nStrategy{' ' * (35 - len('Strategy'))} | {' '.join([f'{col:^8}' for col in pivot_table.columns])}")
    print('-' * 80)
    for strategy in pivot_table.index:
        row_data = [f'{pivot_table.loc[strategy, col]:.2f}' if not pd.isna(pivot_table.loc[strategy, col]) else 'N/A' for col in pivot_table.columns]
        strategy_str = strategy[:35] + '...' if len(strategy) > 35 else strategy
        print(f"{strategy_str:<35} | {' '.join([f'{val:>8}' for val in row_data])}")
    return pivot_table

def process_physics_property(all_results, property_name, result_type):
    property_results = [r for r in all_results if r['Property'] == property_name]
    if not property_results:
        print(f'\n⚠️ did not turn up{property_name}experimental results')
        return None
    print(f"\n{'=' * 80}")
    print(f"analyze{property_name.upper()}physical properties ({('relative comparison' if result_type == 'relative' else 'Absolute value prediction')})")
    print(f'turn up{len(property_results)}result records')
    print(f"{'=' * 80}")
    df_property = pd.DataFrame(property_results)
    overall_results = []
    for strategy in df_property['Strategy'].unique():
        for test_set in df_property['Test_Set'].unique():
            subset = df_property[(df_property['Strategy'] == strategy) & (df_property['Test_Set'] == test_set)]
            if not subset.empty:
                if result_type == 'relative':
                    avg_auc = subset['AUC'].mean()
                    overall_results.append({'Strategy': strategy, 'Test_Set': test_set, 'Property': property_name, 'AUC': avg_auc})
                else:
                    avg_corr = subset['Pearson_Correlation'].mean()
                    overall_results.append({'Strategy': strategy, 'Test_Set': test_set, 'Property': property_name, 'Pearson_Correlation': avg_corr})
    if overall_results:
        df_overall = pd.DataFrame(overall_results)        task_name = f"{property_name.capitalize()} ({('Relative' if result_type == 'relative' else 'Absolute')})"
        pivot_table = create_comparison_table(df_overall, result_type, task_name)
        if not pivot_table.empty:
            best_strategy = pivot_table.index[0]
            best_score = pivot_table.loc[best_strategy, 'Avg']
            baseline_score = pivot_table.loc['Baseline', 'Avg'] if 'Baseline' in pivot_table.index else None
            print(f'\n🏆 {property_name.capitalize()}Best strategy:{best_strategy}')
            main_metric = 'AUC' if result_type == 'relative' else 'Pearson Correlation'
            print(f'average{main_metric}: {best_score:.3f}')
            if baseline_score is not None and best_strategy != 'Baseline':
                if result_type == 'relative':
                    improvement = (best_score - baseline_score) / baseline_score * 100
                else:
                    improvement = (best_score - baseline_score) / max(abs(baseline_score), 0.001) * 100
                print(f'Improved compared to Baseline:{improvement:+.1f}%')
        return pivot_table
    return None

def main():
    print('Start processing Gemini ablation experiment (100samples) - analysis of all physical properties...')
    absolute_patterns = ['gemini_physics_analysis_results_absolute_100samples*']
    relative_patterns = ['gemini_physics_analysis_results_relative_100samples*']
    all_dirs = []
    for pattern in absolute_patterns + relative_patterns:
        all_dirs.extend(glob.glob(pattern))
    print(f'Discover{len(all_dirs)}experiment directory')
    if not all_dirs:
        print('No experiment directories found, please check the path')
        return
    all_absolute_results = []
    all_relative_results = []
    for base_dir in sorted(all_dirs):
        if not os.path.isdir(base_dir):
            continue
        experiment_name = get_experiment_name_from_dir(base_dir)
        print(f'\nProcessing experiments:{experiment_name} ({base_dir})')
        results_files = [f for f in os.listdir(base_dir) if f.endswith('_results.txt') and any((task_type in f for task_type in ['absolute', 'relative']))]
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
                        result_row = {'Strategy': experiment_name, 'Model': f'{model_version}_vjepa', 'Property': chunk['property_name'], 'Test_Set': test_set, 'Type': result_type, 'chunk_id': chunk['chunk_id'], 'Num_Samples': len(chunk['predictions'])}
                        result_row.update(metrics)
                        if result_type == 'absolute':
                            all_absolute_results.append(result_row)
                        else:
                            all_relative_results.append(result_row)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if all_absolute_results:
        print(f"\n{'=' * 100}")
        print('Handling absolute value prediction tasks (ABSOLUTE PREDICTION TASKS)')
        print(f"{'=' * 100}")
        properties = list(set([r['Property'] for r in all_absolute_results]))
        absolute_tables = {}
        for prop in sorted(properties):
            table = process_physics_property(all_absolute_results, prop, 'absolute')
            if table is not None:
                absolute_tables[prop] = table
                prop_filename = f'gemini_{prop}_absolute_100samples_table_{timestamp}.csv'
                table.to_csv(prop_filename)
                print(f'📁 {prop.capitalize()}The absolute value prediction table has been saved as:{prop_filename}')
    if all_relative_results:
        print(f"\n{'=' * 100}")
        print('Handle relative comparison tasks (RELATIVE COMPARISON TASKS)')
print(f"{'=' * 100}")
        properties = list(set([r['Property'] for r in all_relative_results]))
        relative_tables = {}
        for prop in sorted(properties):
            table = process_physics_property(all_relative_results, prop, 'relative')
            if table is not None:
                relative_tables[prop] = table
                prop_filename = f'gemini_{prop}_relative_100samples_table_{timestamp}.csv'
                table.to_csv(prop_filename)
                print(f'📁 {prop.capitalize()}The relative comparison table has been saved as:{prop_filename}')
    if all_absolute_results:
        absolute_df = pd.DataFrame(all_absolute_results)
        absolute_filename = f'gemini_all_physics_absolute_100samples_detailed_{timestamp}.csv'
        absolute_df.to_csv(absolute_filename, index=False)
        print(f'\n📁 All the detailed results of absolute value experiments have been saved as: {absolute_filename}')
    if all_relative_results:
        relative_df = pd.DataFrame(all_relative_results)
        relative_filename = f'gemini_all_physics_relative_100samples_detailed_{timestamp}.csv'
        relative_df.to_csv(relative_filename, index=False)
        print(f'📁 All relative comparison experiment detailed results have been saved as:{relative_filename}')
    print(f"\n{'=' * 100}")
    print('✅ All physical properties ablation experimental analysis completed!')
    print(f"{'=' * 100}")
    if all_absolute_results:
        abs_properties = list(set([r['Property'] for r in all_absolute_results]))
        print(f"📊 The absolute value prediction task includes:{', '.join(sorted(abs_properties))}")
    if all_relative_results:
        rel_properties = list(set([r['Property'] for r in all_relative_results]))
        print(f"📊 Relative comparison tasks include:{', '.join(sorted(rel_properties))}")
    print(f"📁 Co-generation{(len(absolute_tables) if 'absolute_tables' in locals() else 0)}absolute value prediction table")
    print(f"📁 Co-generation{(len(relative_tables) if 'relative_tables' in locals() else 0)}relative comparison table")
    print(f"{'=' * 100}")
if __name__ == '__main__':
    main()