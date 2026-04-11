import os
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr

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
        return {'property_name': property_name, 'predictions': predictions, 'ground_truths': ground_truths}
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
    base_dir = 'test_mllms_from_api/gpt4o_physics_analysis_results_absolute_100samples_icl'
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
        pivot_relative['Average'] = pivot_relative.mean(axis=1).round(4)
        output_filename_relative = f'paper_table_relative_results_{timestamp}.csv'
        pivot_relative.to_csv(output_filename_relative)
        print(f'\nThe relative result table has been saved as: {output_filename_relative}')
        print('\nTable preview (AUC):')
        print(pivot_relative.round(2).to_string())
    if not df_absolute.empty:
        print('\n' + '=' * 50)
        print('A paper-formatted table that generates absolute results (Pearson Correlation)...')
        pivot_absolute = df_absolute.pivot_table(index='Property', columns='Test_Set', values='Pearson_Correlation')
        pivot_absolute = pivot_absolute.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_absolute.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_absolute = pivot_absolute.round(4)
        pivot_absolute['Average'] = pivot_absolute.mean(axis=1).round(4)
        output_filename_absolute = f'paper_table_absolute_results_{timestamp}.csv'
        pivot_absolute.to_csv(output_filename_absolute)
        print(f'\nThe absolute result table has been saved as: {output_filename_absolute}')
        print('\nTable preview (Pearson Correlation):')
        print(pivot_absolute.round(2).to_string())
    detailed_filename = f'detailed_all_results_{timestamp}.csv'
    df.to_csv(detailed_filename, index=False)
    print(f'\nAll detailed results have been saved as: {detailed_filename}')
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