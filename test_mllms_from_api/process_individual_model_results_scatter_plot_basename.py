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
    match = re.match('(\\w+)_vjepa_(test\\d+)_(relative|absolute)_results\\.txt', filename)
    if match:
        return match.groups()
    return (None, None, None)

def extract_comparison_results(file_path, result_type):
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        property_match = re.search('property:(\\w+)', content)
        if not property_match:
            print(f'Unable to retrieve file{file_path}Extract attributes from')
            return None
        property_name = property_match.group(1)
        if result_type == 'relative':
            video_names = re.findall('Video pair:([^\\n]+)', content)
        else:
            video_names = re.findall('video:([^\\n]+)', content)
        if result_type == 'relative':
            comparisons = re.findall('Predicted results:([\\d.]+)\\s+GT results:(\\d+)', content)
            if not comparisons:
                print(f'No comparison results found in file:{file_path}')
                return None
            predictions = [float(p) for p, _ in comparisons]
            ground_truths = [int(g) for _, g in comparisons]
        else:
            comparisons = re.findall('Predicted value:([\\d.-]+)\\s+GT value:([\\d.-]+)', content)
            if not comparisons:
                print(f'No comparison results found in file:{file_path}')
                return None
            predictions = [float(p) for p, _ in comparisons]
            ground_truths = [float(g) for _, g in comparisons]
        if len(video_names) != len(predictions):
            print(f'Warning: Number of video names ({len(video_names)}) and the number of predicted results ({len(predictions)}) is inconsistent')
            min_length = min(len(video_names), len(predictions))
            video_names = video_names[:min_length]
            predictions = predictions[:min_length]
            ground_truths = ground_truths[:min_length]
        return {'property_name': property_name, 'predictions': predictions, 'ground_truths': ground_truths, 'video_names': video_names, 'filename': os.path.basename(file_path)}
    except Exception as e:
        print(f'Process files{file_path}An error occurred:{str(e)}')
        return None

def calculate_metrics(y_true, y_pred, result_type, property_name=None):
    metrics = {}
    try:
        if result_type == 'relative':
            if len(set(y_true)) < 2:
                metrics['AUC'] = 0.5
                print(f'Warning: There is only one true category{set(y_true)}')
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred)
            metrics['Accuracy'] = sum((1 for p, g in zip(y_pred, y_true) if p == g)) / len(y_true)
        else:
            if property_name and property_name.lower() == 'elasticity':
                filtered_indices = [i for i, gt in enumerate(y_true) if gt >= 0.1]
                if filtered_indices:
                    y_true_filtered = [y_true[i] for i in filtered_indices]
                    y_pred_filtered = [y_pred[i] for i in filtered_indices]
                    print(f'-> elasticity filter: original{len(y_true)}samples, after filtering{len(y_true_filtered)}samples (GT>=0.1)')
                else:
                    print(f'-> Elasticity filtering: All sample GTs are less than 0.1, and the indicator cannot be calculated')
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
        print(f'An error occurred while calculating the indicator:{str(e)}')
        return None

def main():
    base_dir = 'test_mllms_from_api/gemini_physics_analysis_results_relative_100samples_oracle'
    all_results = []
    print(f'Start processing the directory:{base_dir}')
    for filename in sorted(os.listdir(base_dir)):
        if not filename.endswith('_results.txt'):
            continue
        file_path = os.path.join(base_dir, filename)
        model_version, test_set, result_type = extract_info_from_filename(filename)
        if not all([model_version, test_set, result_type]):
            print(f'Skip unmatched files:{filename}')
            continue
        print(f'\nProcess files:{filename}')
        result_data = extract_comparison_results(file_path, result_type)
        if result_data:
            video_names = result_data['video_names']
            print(f'Video name list ({len(video_names)}indivual):')
            for i, video_name in enumerate(video_names[:3]):
                print(f' {i + 1}: {video_name}')
            if len(video_names) > 3:
                print(f'... (besides{len(video_names) - 3}indivual)')
            metrics = calculate_metrics(result_data['ground_truths'], result_data['predictions'], result_type, result_data['property_name'])
            if metrics:
                row = {'Model': f'{model_version}_vjepa', 'Property': result_data['property_name'], 'Test_Set': test_set, 'Type': result_type, 'Num_Samples': len(result_data['predictions'])}
                row.update(metrics)
                all_results.append(row)
                metrics_str = ', '.join([f'{k} = {v:.4f}' for k, v in metrics.items()])
                print(f" -> {row['Model']} - {row['Property']} - {row['Test_Set']} ({row['Type']}): {metrics_str}")
    if not all_results:
        print('No results are processed, exit.')
        return
    df = pd.DataFrame(all_results)
    df_relative = df[df['Type'] == 'relative']
    df_absolute = df[df['Type'] == 'absolute']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not df_relative.empty:
        print('\n' + '=' * 50)
        print('Generate paper-formatted tables of relative results (AUC)...')
        pivot_relative = df_relative.pivot_table(index='Property', columns='Test_Set', values='AUC')
        pivot_relative = pivot_relative.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_relative.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_relative = pivot_relative.round(4)
        output_filename_relative = f'paper_table_relative_results_{timestamp}.csv'
        pivot_relative.to_csv(output_filename_relative)
        print(f'\nThe relative result table has been saved as: {output_filename_relative}')
        print('\nTable preview (AUC):')
        print(pivot_relative.to_string())
    if not df_absolute.empty:
        print('\n' + '=' * 50)
        print('A paper-formatted table that generates absolute results (Pearson Correlation)...')
        pivot_absolute = df_absolute.pivot_table(index='Property', columns='Test_Set', values='Pearson_Correlation')
        pivot_absolute = pivot_absolute.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_absolute.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_absolute = pivot_absolute.round(4)
        output_filename_absolute = f'paper_table_absolute_results_{timestamp}.csv'
        pivot_absolute.to_csv(output_filename_absolute)
        print(f'\nThe absolute result table has been saved as: {output_filename_absolute}')
        print('\nTable preview (Pearson Correlation):')
        print(pivot_absolute.to_string())
    detailed_filename = f'detailed_all_results_{timestamp}.csv'
    df.to_csv(detailed_filename, index=False)
    print(f'\nAll detailed results have been saved as: {detailed_filename}')
    if not df_absolute.empty:
        print('\n' + '=' * 50)
        print('Generate a scatterplot of absolute results...')
        output_dir = os.environ.get('MLLM_SCATTER_OUT', 'outputs/mllm_scatter')
        os.makedirs(output_dir, exist_ok=True)
        model_name = base_dir.split('/')[-1].split('_')[0]
        test_splits = ['test1', 'test2', 'test3']
        scatter_data = {}        for filename in sorted(os.listdir(base_dir)):
            if not filename.endswith('_results.txt'):
                continue
            file_path = os.path.join(base_dir, filename)
            model_version, test_set, result_type = extract_info_from_filename(filename)
            if result_type == 'absolute' and test_set in test_splits:
                result_data = extract_comparison_results(file_path, result_type)
                if result_data:
                    property_name = result_data['property_name']
                    gt_values = result_data['ground_truths']
                    pred_values = result_data['predictions']
                    video_names = result_data['video_names']
                    if property_name.lower() == 'elasticity':
                        filtered_indices = [i for i, gt in enumerate(gt_values) if gt >= 0.1]
                        if filtered_indices:
                            gt_values = [gt_values[i] for i in filtered_indices]
                            pred_values = [pred_values[i] for i in filtered_indices]
                            video_names = [video_names[i] for i in filtered_indices]
                            print(f"-> Scatter plot elasticity filter: original{len(result_data['ground_truths'])}samples, after filtering{len(gt_values)}samples (GT>=0.1)")
                        else:
                            print(f'-> Scatter plot elasticity filtering: all sample GT is less than 0.1, skip')
                            continue
                    if test_set not in scatter_data:
                        scatter_data[test_set] = {}
                    if property_name not in scatter_data[test_set]:
                        scatter_data[test_set][property_name] = {'gt': [], 'pred': [], 'video_names': []}
                    scatter_data[test_set][property_name]['gt'].extend(gt_values)
                    scatter_data[test_set][property_name]['pred'].extend(pred_values)
                    scatter_data[test_set][property_name]['video_names'].extend(video_names)
        for test_split in test_splits:
            if test_split in scatter_data:
                for property_name, data in scatter_data[test_split].items():
                    if property_name.lower() == 'viscosity' and test_split == 'test3':
                        print(f'-> Right{property_name} {test_split}Aggregate by liquid name...')
                        liquid_data = defaultdict(list)
                        for i, video_name in enumerate(data['video_names']):
                            try:
                                parts = video_name.split()
                                if len(parts) > 0:
                                    basename = parts[0].split(':')[1].strip() if ':' in parts[0] else parts[0].strip()
                                    basename_parts = basename.split('_')
                                    if len(basename_parts) <= 4:
                                        liquid_name = basename_parts[0]
                                    else:
                                        liquid_name = basename_parts[-4]
                                    liquid_data[liquid_name].append((data['pred'][i], data['gt'][i]))
                            except (IndexError, ValueError):
                                continue
                        aggregated_gt = []
                        aggregated_pred = []
                        for liquid_name, predictions_gts in liquid_data.items():
                            if len(predictions_gts) > 0:
                                pred_values = [pred for pred, gt in predictions_gts]
                                gt_values = [gt for pred, gt in predictions_gts]
                                avg_pred = np.mean(pred_values)
                                gt_value = gt_values[0]
                                aggregated_pred.append(avg_pred)
                                aggregated_gt.append(gt_value)
                                print(f'liquid{liquid_name}: {len(pred_values)}samples -> average predicted value{avg_pred:.4f}, GT {gt_value:.4f}')
                        final_gt = aggregated_gt
                        final_pred = aggregated_pred
                        print(f'After aggregation:{len(final_gt)}liquid')
                    elif property_name.lower() == 'friction' and test_split == 'test3':
                        print(f'-> Right{property_name} {test_split}Aggregate by object+surface combination...')
                        combination_data = defaultdict(list)
                        for i, video_name in enumerate(data['video_names']):
                            try:
                                parts = video_name.split()
                                if len(parts) > 0:
                                    basename = parts[0].split(':')[1].strip() if ':' in parts[0] else parts[0].strip()
                                    obj_name = basename.split('_')[0]
                                    surface_name = basename.split('_')[1]
                                    combination_key = f'{obj_name}-{surface_name}'
                                    combination_data[combination_key].append((data['pred'][i], data['gt'][i]))
                            except (IndexError, ValueError):
                                continue
                        aggregated_gt = []
                        aggregated_pred = []
                        for combination_key, predictions_gts in combination_data.items():
                            if len(predictions_gts) > 0:
                                pred_values = [pred for pred, gt in predictions_gts]
                                gt_values = [gt for pred, gt in predictions_gts]
                                avg_pred = np.mean(pred_values)
                                gt_value = gt_values[0]
                                aggregated_pred.append(avg_pred)
                                aggregated_gt.append(gt_value)
                                print(f'combination{combination_key}: {len(pred_values)}samples -> average predicted value{avg_pred:.4f}, GT {gt_value:.4f}')
                        final_gt = aggregated_gt
                        final_pred = aggregated_pred
                        print(f'After aggregation:{len(final_gt)}combinations')
                    else:
                        print(f'-> skip{property_name} {test_split}(No special treatment required)')
                        continue
                    if len(final_gt) > 0 and len(final_pred) > 0:
                        plt.figure(figsize=(8, 6))
                        plt.scatter(final_gt, final_pred, alpha=0.6)
                        plt.xlabel('Ground Truth')
                        plt.ylabel('Prediction')
                        if property_name.lower() == 'viscosity' and test_split == 'test3':
                            plt.xscale('log')
                            plt.yscale('log')
                        test_number = test_split.replace('test', '')
                        plot_filename = f'{model_name}_{property_name}_test_{test_number}_basename.png'
                        plot_path = os.path.join(output_dir, plot_filename)
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f'Aggregated scatter plot saved:{plot_path}')
                    else:
                        print(f'-> Warning:{property_name} {test_split}There is no valid data after aggregation')
    print('\n' + '=' * 50)
    print('Summarize:')
    print(f'- processed{len(df)}files')
    if not df_relative.empty:
        print(f"- Relative results (AUC): average ={df_relative['AUC'].mean():.4f}")
    if not df_absolute.empty:
        print(f"- Absolute results (Pearson): Average ={df_absolute['Pearson_Correlation'].mean():.4f}")
        print(f"- Absolute result (MAE): average ={df_absolute['MAE'].mean():.4f}")
if __name__ == '__main__':
    main()
