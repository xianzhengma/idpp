import os
import glob
import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score
import numpy as np

def get_experiment_name_from_dir(dir_path: str) -> str:
    dir_name = os.path.basename(dir_path)
    if dir_name == 'gemini_physics_analysis_results_elasticity_relative_20samples':
        return 'Baseline'
    elif dir_name.startswith('gemini_physics_analysis_results_elasticity_relative_20samples_'):
        suffix = dir_name.replace('gemini_physics_analysis_results_elasticity_relative_20samples_', '')
        mapping = {'concat': '+ Concat', 'icl': '+ ICL', 'oracle': '+ Oracle', 'frameindex': '+ FrameIndex', 'concat_icl': '+ Concat+ICL', 'concat_oracle': '+ Concat+Oracle', 'concat_frameindex': '+ Concat+FrameIndex', 'icl_oracle': '+ ICL+Oracle', 'icl_frameindex': '+ ICL+FrameIndex', 'oracle_frameindex': '+ Oracle+FrameIndex', 'concat_icl_oracle': '+ Concat+ICL+Oracle', 'concat_icl_frameindex': '+ Concat+ICL+FrameIndex', 'concat_oracle_frameindex': '+ Concat+Oracle+FrameIndex', 'icl_oracle_frameindex': '+ ICL+Oracle+FrameIndex', 'concat_icl_oracle_frameindex': '+ Concat+ICL+Oracle+FrameIndex'}
        return mapping.get(suffix, f"+ {suffix.replace('_', '+').title()}")
    return dir_name

def extract_comparisons_and_gts(result_file: str) -> List[Dict]:
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = '--- comparison (\\d+) ---\\n.*?Video pair:(.*?)\\n.*?Predicted results:(.*?)\\n.*?GT results:(.*?)\\n'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        comparison_id, video_pair, predicted_str, gt_str = match
        predicted=None
        if predicted_str != 'None' and predicted_str.strip():
            try:
                predicted = float(predicted_str)
            except ValueError:
                predicted=None
        gt=None
        if gt_str != 'None' and gt_str.strip():
            try:
                gt = int(gt_str)
            except ValueError:
                gt=None
        results.append({'comparison_id': int(comparison_id), 'video_pair': video_pair.strip(), 'predicted': predicted, 'gt': gt})
    return results

def calculate_auc_metrics(predictions: List[float], ground_truths: List[int]) -> Dict[str, float]:
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {'auc': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None, 'valid_predictions': 0, 'total_samples': len(ground_truths), 'prediction_stats': None}
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if p is not None and g is not None]
    if len(valid_pairs) == 0:
        return {'auc': None, 'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None, 'valid_predictions': 0, 'total_samples': len(ground_truths), 'prediction_stats': None}
    valid_predictions, valid_gts = zip(*valid_pairs)
    valid_predictions = list(valid_predictions)
    valid_gts = list(valid_gts)
    prediction_stats = {'min': np.min(valid_predictions), 'max': np.max(valid_predictions), 'mean': np.mean(valid_predictions), 'std': np.std(valid_predictions), 'unique_values': sorted(list(set(valid_predictions)))}
    try:
        unique_gts = set(valid_gts)
        if len(unique_gts) == 1:
            auc=None
        else:
            auc = roc_auc_score(valid_gts, valid_predictions)
    except Exception as e:
        print(f'Warning: AUC calculation failed:{e}')
        auc=None
    binary_predictions = [1 if p >= 0.5 else 0 for p in valid_predictions]
    accuracy = sum((bp == g for bp, g in zip(binary_predictions, valid_gts))) / len(valid_gts)
    from sklearn.metrics import precision_score, recall_score, f1_score
    try:
        precision = precision_score(valid_gts, binary_predictions, zero_division=0)
        recall = recall_score(valid_gts, binary_predictions, zero_division=0)
        f1 = f1_score(valid_gts, binary_predictions, zero_division=0)
    except Exception as e:
        print(f'Warning: Classification indicator calculation failed:{e}')
        precision=None
        recall=None        f1 = None
    return {'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'valid_predictions': len(valid_predictions), 'total_samples': len(ground_truths), 'prediction_stats': prediction_stats}

def process_single_experiment(exp_dir: str) -> Dict:
    print(f'Processing experiments:{exp_dir}')
    result_files = glob.glob(os.path.join(exp_dir, '*relative*results.txt'))
    if not result_files:
        print(f'Warning: elasticity relative result file not found')
        return None
    all_results = []
    test_metrics = {}
    for result_file in result_files:
        filename = os.path.basename(result_file)
        if 'test1' in filename:
            test_name = 'Test-1'
        elif 'test2' in filename:
            test_name = 'Test-2'
        elif 'test3' in filename:
            test_name = 'Test-3'
        else:
            test_name = 'Unknown'
        results = extract_comparisons_and_gts(result_file)
        all_results.extend(results)
        if results:
            predictions = [r['predicted'] for r in results]
            ground_truths = [r['gt'] for r in results]
            metrics = calculate_auc_metrics(predictions, ground_truths)
            test_metrics[test_name] = metrics
            valid_count = metrics['valid_predictions']
            total_count = metrics['total_samples']
            print(f'  {test_name}: {valid_count}/{total_count}Effective prediction')
            if metrics['prediction_stats'] is not None:
                stats = metrics['prediction_stats']
                print(f"Predicted value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"Predicted value mean ± standard deviation:{stats['mean']:.3f}±{stats['std']:.3f}")
                print(f"Predicted value distribution:{stats['unique_values']}")
            if metrics['auc'] is not None:
                print(f"    AUC: {metrics['auc']:.4f}")
            if metrics['accuracy'] is not None:
                print(f"Accuracy:{metrics['accuracy']:.4f}")
    return {'experiment_name': get_experiment_name_from_dir(exp_dir), 'directory': exp_dir, 'test_metrics': test_metrics, 'total_results': len(all_results)}

def main():
    print('Start processing the Elasticity Relative Ablation experiment results...')
    experiment_dirs = []
    base_patterns = ['gemini_physics_analysis_results_elasticity_relative_20samples', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat', 'gemini_physics_analysis_results_elasticity_relative_20samples_icl', 'gemini_physics_analysis_results_elasticity_relative_20samples_oracle', 'gemini_physics_analysis_results_elasticity_relative_20samples_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle', 'gemini_physics_analysis_results_elasticity_relative_20samples_icl_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_oracle_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle_frameindex']
    for pattern in base_patterns:
        matching_dirs = glob.glob(pattern)
        experiment_dirs.extend(matching_dirs)
    experiment_dirs = sorted(list(set(experiment_dirs)))
    if not experiment_dirs:
        print('❌ No experimental results directory found')
        print('Please ensure that the elasticity relative ablation experiment has been run and the results directory exists')
        return
    print(f'turn up{len(experiment_dirs)}Experiment directory:')
    for exp_dir in experiment_dirs:
        print(f'  - {exp_dir}')
    print()
    experiment_results = []
    for exp_dir in experiment_dirs:
        result = process_single_experiment(exp_dir)
        if result:
            experiment_results.append(result)
    if not experiment_results:
        print('❌ No experimental results were successfully processed')
        return
    print('Generate comparison table...')
    table_data = []
    test_names = ['Test-1', 'Test-2', 'Test-3']
    for exp_result in experiment_results:
        row = {'Experiment': exp_result['experiment_name'], 'Directory': os.path.basename(exp_result['directory'])}
        avg_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'valid_count': [], 'total_count': []}
        for test_name in test_names:
            if test_name in exp_result['test_metrics']:
                metrics = exp_result['test_metrics'][test_name]
                auc_val = metrics['auc']
                row[f'{test_name}_AUC'] = f'{auc_val:.4f}' if auc_val is not None else 'N/A'
                if auc_val is not None:
                    avg_metrics['auc'].append(auc_val)
                acc_val = metrics['accuracy']
                row[f'{test_name}_Acc'] = f'{acc_val:.4f}' if acc_val is not None else 'N/A'
                if acc_val is not None:
                    avg_metrics['accuracy'].append(acc_val)
                prec_val = metrics['precision']
                row[f'{test_name}_Prec'] = f'{prec_val:.4f}' if prec_val is not None else 'N/A'
                if prec_val is not None:
                    avg_metrics['precision'].append(prec_val)
                rec_val = metrics['recall']
                row[f'{test_name}_Rec'] = f'{rec_val:.4f}' if rec_val is not None else 'N/A'
                if rec_val is not None:
                    avg_metrics['recall'].append(rec_val)
                f1_val = metrics['f1_score']
                row[f'{test_name}_F1'] = f'{f1_val:.4f}' if f1_val is not None else 'N/A'
                if f1_val is not None:
                    avg_metrics['f1_score'].append(f1_val)
                row[f'{test_name}_Valid'] = f"{metrics['valid_predictions']}/{metrics['total_samples']}"
                avg_metrics['valid_count'].append(metrics['valid_predictions'])
                avg_metrics['total_count'].append(metrics['total_samples'])
            else:
                row[f'{test_name}_AUC'] = 'N/A'
                row[f'{test_name}_Acc'] = 'N/A'
                row[f'{test_name}_Prec'] = 'N/A'
                row[f'{test_name}_Rec'] = 'N/A'
                row[f'{test_name}_F1'] = 'N/A'
                row[f'{test_name}_Valid'] = 'N/A'
        for metric_name in ['auc', 'accuracy', 'precision', 'recall', 'f1_score']:
            if avg_metrics[metric_name]:
                avg_val = np.mean(avg_metrics[metric_name])
                row[f"Avg_{metric_name.replace('_score', '').title()}"] = f'{avg_val:.4f}'
            else:
                row[f"Avg_{metric_name.replace('_score', '').title()}"] = 'N/A'
        if avg_metrics['valid_count'] and avg_metrics['total_count']:
            total_valid = sum(avg_metrics['valid_count'])
            total_all = sum(avg_metrics['total_count'])
            row['Total_Valid'] = f'{total_valid}/{total_all}'
        else:
            row['Total_Valid'] = 'N/A'
        table_data.append(row)
    df = pd.DataFrame(table_data)
    experiment_order = ['Baseline', '+ Concat', '+ ICL', '+ Oracle', '+ FrameIndex', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+ICL+FrameIndex', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex', '+ Concat+ICL+Oracle+FrameIndex']
    df['exp_order'] = df['Experiment'].map({exp: i for i, exp in enumerate(experiment_order)})
    df = df.sort_values('exp_order').drop('exp_order', axis=1)
    print('\n' + '=' * 120)
    print('Elasticity Relative Ablation Study Results - Overview')
    print('=' * 120)
    overview_columns = ['Experiment', 'Avg_Auc', 'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_F1', 'Total_Valid']
    overview_df = df[overview_columns].copy()
    print(overview_df.to_string(index=False))
    print('=' * 120)
print(f'\nDetailed results (decomposed by test set):')
    print('-' * 160)
    detail_columns = ['Experiment']
    for test_name in test_names:
        detail_columns.extend([f'{test_name}_AUC', f'{test_name}_Acc', f'{test_name}_Valid'])
    detail_df = df[detail_columns].copy()
    print(detail_df.to_string(index=False))
    print('-' * 160)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'elasticity_relative_20samples_ablation_results_{timestamp}.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    if 'Baseline' in df['Experiment'].values:
        print(f'\n' + '=' * 80)
        print('Performance analysis (improvements compared to Baseline)')
        print('=' * 80)
        baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
        for _, row in df.iterrows():
            if row['Experiment'] == 'Baseline':
                continue
            print(f"\n{row['Experiment']}:")
            metrics_to_analyze = ['Auc', 'Accuracy', 'Precision', 'Recall', 'F1']
            for metric in metrics_to_analyze:
                baseline_val = baseline_row[f'Avg_{metric}']
                current_val = row[f'Avg_{metric}']
                if baseline_val != 'N/A' and current_val != 'N/A':
                    try:
                        baseline_num = float(baseline_val)
                        current_num = float(current_val)
                        if baseline_num != 0:
                            improvement = (current_num - baseline_num) / baseline_num * 100
                            if improvement > 0:
                                print(f' {metric}: {improvement:+.1f}% (better)')
                            else:
                                print(f' {metric}: {improvement:+.1f}% (worse)')
                        elif current_num > 0:
                            print(f' {metric}: Increase from 0 to{current_num:.4f}(significant improvement)')
                        else:
                            print(f' {metric}: Still 0 (no improvement)')
                    except ValueError:
                        print(f' {metric}: Unable to compare (wrong data format)')
                else:
                    print(f' {metric}: N/A (missing data)')
        print(f'\n' + '=' * 80)
        print('Overall statistics')
        print('=' * 80)
        total_experiments = len(df)
        experiments_with_predictions = 0
        total_predictions = 0
        total_samples = 0
        for _, row in df.iterrows():
            valid_str = row['Total_Valid']
            if valid_str != 'N/A':
                try:
                    valid, total = map(int, valid_str.split('/'))
                    if valid > 0:
                        experiments_with_predictions += 1
                    total_predictions += valid
                    total_samples += total
                except:
                    pass
        print(f'Total number of experiments:{total_experiments}')
        print(f'Experiments with valid predictions:{experiments_with_predictions}')
        print(f'Total forecasts:{total_predictions}/{total_samples} ({total_predictions / total_samples * 100:.1f}%)')
        best_experiments = {}
        for metric in metrics_to_analyze:
            col_name = f'Avg_{metric}'
            if col_name in df.columns:
                valid_scores = []
                for _, row in df.iterrows():
                    val = row[col_name]
                    if val != 'N/A':
                        try:
                            valid_scores.append((float(val), row['Experiment']))
                        except:
                            pass
                if valid_scores:
                    best_score, best_exp = max(valid_scores)
                    best_experiments[metric] = (best_exp, best_score)
        if best_experiments:
            print(f'\nBest performance:')
            for metric, (exp_name, score) in best_experiments.items():
                print(f' {metric}: {exp_name} ({score:.4f})')
    print(f'\n✅ Processing completed!')
    print(f'📊 Detailed results have been saved to:{csv_filename}')
    print(f'🔍Co-processing{len(experiment_results)}experiments')
    print(f'📈 analyzed{len(test_names)}test set')if __name__ == '__main__':
    main()
