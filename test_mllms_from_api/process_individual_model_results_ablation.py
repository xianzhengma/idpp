import os
import re
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr
import glob

def extract_info_from_filename(filename):
    match = re.match('(\\w+)_vjepa_(test\\d+)_(relative|absolute)_results\\.txt', filename)
    if match:
        return match.groups()
    return (None, None, None)

def get_experiment_name_from_dir(dir_path):
    dir_name = os.path.basename(dir_path)
    if dir_name.startswith('gemini_physics_analysis_results_elasticity_relative_50samples'):
        if dir_name == 'gemini_physics_analysis_results_elasticity_relative_50samples':
            return 'Baseline'
        else:
            features_part = dir_name.replace('gemini_physics_analysis_results_elasticity_relative_50samples_', '')
            feature_mapping = {'concat': '+ Concat', 'frameindex': '+ FrameIndex', 'icl': '+ ICL', 'oracle': '+ Oracle', 'concat_frameindex': '+ Concat+FrameIndex', 'concat_frameindex_icl': '+ Concat+FrameIndex+ICL', 'concat_frameindex_icl_oracle': '+ Concat+FrameIndex+ICL+Oracle', 'concat_icl': '+ Concat+ICL', 'concat_oracle': '+ Concat+Oracle', 'frameindex_icl': '+ FrameIndex+ICL', 'frameindex_oracle': '+ FrameIndex+Oracle', 'icl_oracle': '+ ICL+Oracle'}
            return feature_mapping.get(features_part, f"+ {features_part.replace('_', '+').title()}")
    experiment_mapping = {'gemini_physics_analysis_results': 'Baseline', 'gemini_physics_analysis_results_icl-example': '+ ICL', 'gemini_physics_analysis_results_icl': '+ ICL', 'gemini_physics_analysis_results_follow-oracle-test': '+ Oracle', 'gemini_physics_analysis_results_oracle': '+ Oracle', 'gemini_physics_analysis_results_icl-example_follow-oracle-test': '+ ICL+Oracle', 'gemini_physics_analysis_results_icl_oracle': '+ ICL+Oracle', 'gemini_physics_analysis_results_concat-mode': '+ Concat', 'gemini_physics_analysis_results_concat': '+ Concat', 'gemini_physics_analysis_results_frame-index': '+ FrameIndex', 'gemini_physics_analysis_results_frameindex': '+ FrameIndex', 'gemini_physics_analysis_results_icl-example_concat-mode': '+ ICL+Concat', 'gemini_physics_analysis_results_concat_icl': '+ ICL+Concat', 'gemini_physics_analysis_results_follow-oracle-test_concat-mode': '+ Oracle+Concat', 'gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index': '+ ICL+Oracle+FrameIndex', 'gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index': '+ AllFeatures'}
    return experiment_mapping.get(dir_name, dir_name)

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
            comparisons = re.findall('Predicted results:(\\d+)\\s+GT results:(\\d+)', content)
            if not comparisons:
                print(f'No comparison results found in file:{file_path}')
                return None
            predictions = [int(p) for p, _ in comparisons]
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

def calculate_metrics(y_true, y_pred, result_type):
    metrics = {}
    try:
        if result_type == 'relative':
            if len(set(y_true)) < 2:
                metrics['AUC'] = 0.5
print(f'Warning: There is only one true category{set(y_true)}')
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred)
            metrics['Accuracy'] = sum((1 for p, g in zip(y_pred, y_true) if p == g)) / len(y_true)
        elif len(y_true) < 2:
            metrics['Pearson_Correlation'] = 0.0
            metrics['MAE'] = 0.0
        else:
            correlation, _ = pearsonr(y_true, y_pred)
            metrics['Pearson_Correlation'] = correlation
            metrics['MAE'] = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return metrics
    except Exception as e:
        print(f'An error occurred while calculating the indicator:{str(e)}')
        return None

def main():
    pattern = 'gemini_physics_analysis_results_elasticity_relative_50samples*'
    base_dirs = glob.glob(pattern)
    if not base_dirs:
        print(f'No matching directory found:{pattern}')
        return
    base_dirs.sort()
    all_results = []
    print(f'Discover{len(base_dirs)}Elasticity experiment directories:')
    for dir_path in base_dirs:
        experiment_name = get_experiment_name_from_dir(dir_path)
        print(f' - {dir_path} -> {experiment_name}')
    print()
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        experiment_name = get_experiment_name_from_dir(base_dir)
        print(f'Processing experiments:{experiment_name} ({base_dir})')
        results_files = [f for f in os.listdir(base_dir) if f.endswith('_results.txt')]
        if not results_files:
            print(f'Warning: table of contents{base_dir}No result file found in')
            continue
        for filename in sorted(results_files):
            file_path = os.path.join(base_dir, filename)
            model_version, test_set, result_type = extract_info_from_filename(filename)
            if not all([model_version, test_set, result_type]):
                print(f'Skip unmatched files:{filename}')
                continue
            print(f'Process files:{filename}')
            result_data = extract_comparison_results(file_path, result_type)
            if result_data:
                metrics = calculate_metrics(result_data['ground_truths'], result_data['predictions'], result_type)
                if metrics:
                    row = {'Experiment': experiment_name, 'Model': f'{model_version}_vjepa', 'Property': result_data['property_name'], 'Test_Set': test_set, 'Type': result_type, 'Num_Samples': len(result_data['predictions'])}
                    row.update(metrics)
                    all_results.append(row)
                    metrics_str = ', '.join([f'{k} = {v:.4f}' for k, v in metrics.items()])
                    print(f" -> {row['Property']} - {row['Test_Set']} ({row['Type']}): {metrics_str}")
        print()
    if not all_results:
        print('No results are processed, exit.')
        return
    df = pd.DataFrame(all_results)
    df_relative = df[df['Type'] == 'relative']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('\n' + '=' * 80)
    print('Generate Elasticity experiment comparison table...')
    if not df_relative.empty:
        print('\nElasticity relative comparison test results (AUC) - special comparison table:')
        print('-' * 70)
        pivot_elasticity = df_relative.pivot_table(index='Experiment', columns='Test_Set', values='AUC', aggfunc='mean')
        pivot_elasticity = pivot_elasticity.reindex(columns=['test1', 'test2', 'test3'], fill_value=np.nan)
        pivot_elasticity.columns = ['Test-1', 'Test-2', 'Test-3']
        pivot_elasticity['Average'] = pivot_elasticity.mean(axis=1)
        pivot_elasticity = pivot_elasticity.round(4)
        pivot_elasticity = pivot_elasticity.sort_values('Average', ascending=False)
        print(pivot_elasticity.to_string())
        elasticity_filename = f'elasticity_relative_50samples_comparison_{timestamp}.csv'
        pivot_elasticity.to_csv(elasticity_filename)
        print(f'\n✓ The Elasticity experiment comparison table has been saved as: {elasticity_filename}')
        baseline_avg = pivot_elasticity.loc['Baseline', 'Average'] if 'Baseline' in pivot_elasticity.index else None
        print(f'\n📊 Performance analysis:')print(f"{'Experimental configuration':<35} {'Average AUC':<10} {'Compared to Baseline':<12}")
        print('-' * 60)
        for exp_name in pivot_elasticity.index:
            avg_auc = pivot_elasticity.loc[exp_name, 'Average']
            if baseline_avg and exp_name != 'Baseline':
                improvement = (avg_auc - baseline_avg) / baseline_avg * 100
                improvement_str = f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%'
            else:
                improvement_str = 'Baseline'
            print(f'{exp_name:<35} {avg_auc:<10.4f} {improvement_str:<12}')
        best_experiment = pivot_elasticity.index[0]
        best_average = pivot_elasticity.loc[best_experiment, 'Average']
        print(f'\n🏆 Best Experimental configuration: {best_experiment}')
        print(f'Average AUC:{best_average:.4f}')
        if baseline_avg:
            improvement = (best_average - baseline_avg) / baseline_avg * 100
            print(f'Improved compared to Baseline:{improvement:.1f}%')
        print('\n' + '=' * 70)
    detailed_filename = f'detailed_elasticity_experiments_{timestamp}.csv'
    df.to_csv(detailed_filename, index=False)
    print(f'\n📁 All detailed results have been saved as: {detailed_filename}')
    print('\n' + '=' * 80)
    print('Processing completion summary:')
    print(f'- processed{len(base_dirs)}Elasticity experiment directories')
    print(f'- Processed in total{len(df)}result files')
    print(f'- All experiments are elasticity relative comparison tests')
    print(f"- Average AUC:{df_relative['AUC'].mean():.4f}")
    best_relative = df_relative.groupby('Experiment')['AUC'].mean().idxmax()
    print(f'- Best experiments:{best_relative}')
    print('=' * 80)
if __name__ == '__main__':
    main()