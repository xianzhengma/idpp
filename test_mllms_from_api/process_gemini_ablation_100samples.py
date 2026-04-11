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
    experiment_mapping = {'gemini_physics_analysis_results_absolute_100samples': 'Baseline', 'gemini_physics_analysis_results_relative_100samples': 'Baseline', 'gemini_physics_analysis_results_absolute_100samples_frameindex': '+ Frame Index Provided', 'gemini_physics_analysis_results_relative_100samples_frameindex': '+ Frame Index Provided', 'gemini_physics_analysis_results_absolute_100samples_icl': '+ Few-Shot Examples', 'gemini_physics_analysis_results_relative_100samples_icl': '+ Few-Shot Examples', 'gemini_physics_analysis_results_absolute_100samples_oracle': '+ Oracle Estimation Teaching', 'gemini_physics_analysis_results_relative_100samples_oracle': '+ Oracle Estimation Teaching', 'gemini_physics_analysis_results_absolute_100samples_concat': '+ Black Frames in Between', 'gemini_physics_analysis_results_relative_100samples_concat': '+ Black Frames in Between'}
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
        if len(predictions) != 100:
            print(f'Warning: File{file_path}Only in{len(predictions)}samples, not the expected 100')
        chunks = []
        for i in range(5):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(predictions))
            if start_idx < len(predictions):
                chunk_preds = predictions[start_idx:end_idx]
                chunk_gt = ground_truths[start_idx:end_idx]
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

def process_chunk_results(all_chunk_results, result_type):
    print(f"\n{'=' * 80}")
    print(f'Analysis results: Divide 100 samples into 5 parts (20 pieces each) for independent evaluation')
    print(f"Task type:{('relative comparison' if result_type == 'relative' else 'Absolute value prediction')}")
    print(f"{'=' * 80}")
    for chunk_id in range(1, 6):
        chunk_data = [result for result in all_chunk_results if result['chunk_id'] == chunk_id]
        if not chunk_data:
            continue
        print(f'\n{chunk_id}data (sample{(chunk_id - 1) * 20 + 1}-{chunk_id * 20}):')
        print('-' * 50)
        df_chunk = pd.DataFrame(chunk_data)
        task_name = f"Chunk {chunk_id} ({('Relative' if result_type == 'relative' else 'Absolute')})"
        pivot_table = create_comparison_table(df_chunk, result_type, task_name)
        best_strategy = pivot_table.index[0]
        best_score = pivot_table.loc[best_strategy, 'Avg']
        baseline_score = pivot_table.loc['Baseline', 'Avg'] if 'Baseline' in pivot_table.index else None
        print(f'\n🏆{chunk_id}The best strategy:{best_strategy}')
        main_metric = 'AUC' if result_type == 'relative' else 'Pearson Correlation'
        print(f'average{main_metric}: {best_score:.3f}')
        if baseline_score is not None and best_strategy != 'Baseline':
            if result_type == 'relative':
                improvement = (best_score - baseline_score) / baseline_score * 100
            else:
                improvement = (best_score - baseline_score) / max(abs(baseline_score), 0.001) * 100            print(f'Improved compared to Baseline:{improvement:+.1f}%')

def main():
    print('Start processing Gemini ablation experiment (100samples)...')
    absolute_pattern = 'gemini_physics_analysis_results_absolute_100samples*'
    relative_pattern = 'gemini_physics_analysis_results_relative_100samples*'
    absolute_dirs = glob.glob(absolute_pattern)
    relative_dirs = glob.glob(relative_pattern)
    print(f'Discover{len(absolute_dirs)}Absolute value experiment directory')
    print(f'Discover{len(relative_dirs)}catalog of relative comparison experiments')
    if not absolute_dirs and (not relative_dirs):
        print('No experiment directories found, please check the path')
        return
    if absolute_dirs:
        print(f"\n{'=' * 80}")
        print('Handling absolute value prediction experiments...')
        print(f"{'=' * 80}")
        all_absolute_chunk_results = []
        for base_dir in sorted(absolute_dirs):
            if not os.path.isdir(base_dir):
                continue
            experiment_name = get_experiment_name_from_dir(base_dir)
            print(f'\nProcessing experiments:{experiment_name} ({base_dir})')
            results_files = [f for f in os.listdir(base_dir) if f.endswith('_absolute_results.txt')]
            for filename in sorted(results_files):
                file_path = os.path.join(base_dir, filename)
                model_version, test_set, result_type = extract_info_from_filename(filename)
                if not all([model_version, test_set, result_type]):
                    continue
                print(f'Process files:{filename}')
                chunks = extract_results_from_file(file_path, result_type)
                if chunks:
                    for chunk in chunks:
                        metrics = calculate_metrics(chunk['ground_truths'], chunk['predictions'], result_type)
                        if metrics:
                            result_row = {'Strategy': experiment_name, 'Model': f'{model_version}_vjepa', 'Property': chunk['property_name'], 'Test_Set': test_set, 'Type': result_type, 'chunk_id': chunk['chunk_id'], 'Num_Samples': len(chunk['predictions'])}
                            result_row.update(metrics)
                            all_absolute_chunk_results.append(result_row)
        if all_absolute_chunk_results:
            process_chunk_results(all_absolute_chunk_results, 'absolute')
    if relative_dirs:
        print(f"\n{'=' * 80}")
        print('Handling Relative Comparison Experiments...')
        print(f"{'=' * 80}")
        all_relative_chunk_results = []
        for base_dir in sorted(relative_dirs):
            if not os.path.isdir(base_dir):
                continue
            experiment_name = get_experiment_name_from_dir(base_dir)
            print(f'\nProcessing experiments:{experiment_name} ({base_dir})')
            results_files = [f for f in os.listdir(base_dir) if f.endswith('_relative_results.txt')]
            for filename in sorted(results_files):
                file_path = os.path.join(base_dir, filename)
                model_version, test_set, result_type = extract_info_from_filename(filename)
                if not all([model_version, test_set, result_type]):
                    continue
                print(f'Process files:{filename}')
                chunks = extract_results_from_file(file_path, result_type)
                if chunks:
                    for chunk in chunks:
                        metrics = calculate_metrics(chunk['ground_truths'], chunk['predictions'], result_type)
                        if metrics:
                            result_row = {'Strategy': experiment_name, 'Model': f'{model_version}_vjepa', 'Property': chunk['property_name'], 'Test_Set': test_set, 'Type': result_type, 'chunk_id': chunk['chunk_id'], 'Num_Samples': len(chunk['predictions'])}
                            result_row.update(metrics)
                            all_relative_chunk_results.append(result_row)
        if all_relative_chunk_results:
            process_chunk_results(all_relative_chunk_results, 'relative')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
if 'all_absolute_chunk_results' in locals() and all_absolute_chunk_results:
        absolute_df = pd.DataFrame(all_absolute_chunk_results)
        absolute_filename = f'gemini_absolute_100samples_chunk_analysis_{timestamp}.csv'
        absolute_df.to_csv(absolute_filename, index=False)
        print(f'\n📁 The detailed results of the absolute value experiment have been saved as: {absolute_filename}')
    if 'all_relative_chunk_results' in locals() and all_relative_chunk_results:
        relative_df = pd.DataFrame(all_relative_chunk_results)
        relative_filename = f'gemini_relative_100samples_chunk_analysis_{timestamp}.csv'
        relative_df.to_csv(relative_filename, index=False)
        print(f'📁 The detailed results of the relative comparison experiment have been saved as:{relative_filename}')
    print(f"\n{'=' * 80}")
    print('✅ Gemini ablation experiment analysis completed!')
    print(f"{'=' * 80}")
if __name__ == '__main__':
    main()