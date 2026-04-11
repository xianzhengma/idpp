import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

def parse_results_file(file_path: str) -> Dict:
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split('------------------------------------------------------------')
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        if len(lines) < 5:
            continue
        test_info = {}
        for line in lines[:10]:
            if ':' in line:
                key, value = line.split(':', 1)
                test_info[key.strip()] = value.strip()
        if 'Version' not in test_info or 'Property' not in test_info:
            continue
        version = test_info['Version']
        property_name = test_info['Property']
        test_type = test_info['Type']
        test_set = test_info['Test Set']
        test_id = f'{version}_{property_name}_{test_type}_{test_set}'
        if test_type == 'absolute':
            predicted_values = []
            gt_values = []
            for line in lines:
                if 'predicted=' in line and 'gt=' in line:
                    pred_match = re.search('predicted=([0-9.]+|None)', line)
                    gt_match = re.search('gt=([0-9.e-]+)', line)
                    if pred_match and gt_match:
                        pred_str = pred_match.group(1)
                        if pred_str != 'None':
                            predicted_values.append(float(pred_str))
                            gt_values.append(float(gt_match.group(1)))
            results[test_id] = {'type': 'absolute', 'property': property_name, 'version': version, 'test_set': test_set, 'predicted': predicted_values, 'ground_truth': gt_values}
        elif test_type == 'relative':
            predicted_values = []
            gt_values = []
            for line in lines:
                if 'predicted=' in line and 'gt=' in line:
                    pred_match = re.search('predicted=([01])', line)
                    gt_match = re.search('gt=([01])', line)
                    if pred_match and gt_match:
                        predicted_values.append(int(pred_match.group(1)))
                        gt_values.append(int(gt_match.group(1)))
            results[test_id] = {'type': 'relative', 'property': property_name, 'version': version, 'test_set': test_set, 'predicted': predicted_values, 'ground_truth': gt_values}
    return results

def calculate_pearson_correlation(predicted: List[float], ground_truth: List[float]) -> Tuple[float, float]:
    if len(predicted) != len(ground_truth) or len(predicted) == 0:
        return (0.0, 1.0)
    correlation, p_value = pearsonr(predicted, ground_truth)
    return (correlation, p_value)

def calculate_roc_metrics(predicted: List[int], ground_truth: List[int]) -> Dict:
    if len(predicted) != len(ground_truth) or len(predicted) == 0:
        return {'auc': 0.0, 'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    try:
        auc = roc_auc_score(ground_truth, predicted)
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(ground_truth, predicted)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, sum(predicted))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return {'auc': auc, 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

def export_to_csv(absolute_results: Dict, relative_results: Dict, output_dir: str='.') -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if absolute_results:
        abs_data = []
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(data['predicted'], data['ground_truth'])
            pred_array = np.array(data['predicted'])
            gt_array = np.array(data['ground_truth'])
            mae = np.mean(np.abs(pred_array - gt_array))
            rmse = np.sqrt(np.mean((pred_array - gt_array) ** 2))
            abs_data.append({'Property': data['property'], 'Version': data['version'], 'Test_Set': data['test_set'], 'Test_ID': test_id, 'Sample_Count': len(data['predicted']), 'Pearson_Correlation': correlation, 'P_Value': p_value, 'Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Correlation_Strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak', 'Mean_Predicted': np.mean(pred_array), 'Mean_Ground_Truth': np.mean(gt_array), 'Std_Predicted': np.std(pred_array), 'Std_Ground_Truth': np.std(gt_array), 'MAE': mae, 'RMSE': rmse, 'Min_Predicted': np.min(pred_array), 'Max_Predicted': np.max(pred_array), 'Min_Ground_Truth': np.min(gt_array), 'Max_Ground_Truth': np.max(gt_array)})
        df_abs = pd.DataFrame(abs_data)
        abs_filename = f'physics_analysis_absolute_{timestamp}.csv'
        abs_filepath = os.path.join(output_dir, abs_filename)
        df_abs.to_csv(abs_filepath, index=False, encoding='utf-8')
        print(f'✅ Absolute value test results have been exported to:{abs_filepath}')
    if relative_results:
        rel_data = []
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            pred_array = np.array(data['predicted'])
            gt_array = np.array(data['ground_truth'])
            rel_data.append({'Property': data['property'], 'Version': data['version'], 'Test_Set': data['test_set'], 'Test_ID': test_id, 'Sample_Count': len(data['predicted']), 'AUC': metrics['auc'], 'Accuracy': metrics['accuracy'], 'Sensitivity': metrics['sensitivity'], 'Specificity': metrics['specificity'], 'Precision': metrics['precision'], 'Recall': metrics['recall'], 'F1_Score': metrics['f1_score'], 'Predicted_Positive_Rate': np.mean(pred_array), 'Ground_Truth_Positive_Rate': np.mean(gt_array), 'True_Positives': np.sum((pred_array == 1) & (gt_array == 1)), 'True_Negatives': np.sum((pred_array == 0) & (gt_array == 0)), 'False_Positives': np.sum((pred_array == 1) & (gt_array == 0)), 'False_Negatives': np.sum((pred_array == 0) & (gt_array == 1))})
        df_rel = pd.DataFrame(rel_data)
        rel_filename = f'physics_analysis_relative_{timestamp}.csv'
        rel_filepath = os.path.join(output_dir, rel_filename)
        df_rel.to_csv(rel_filepath, index=False, encoding='utf-8')
        print(f'✅ Relative test results have been exported to:{rel_filepath}')
    summary_data = []
    if absolute_results:
        all_correlations = []
        property_stats = {}
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(data['predicted'], data['ground_truth'])
            all_correlations.append(correlation)
            prop = data['property']
            if prop not in property_stats:
                property_stats[prop] = []
            property_stats[prop].append(correlation)
        summary_data.append({'Test_Type': 'Absolute', 'Property': 'All', 'Metric': 'Pearson_Correlation', 'Mean': np.mean(all_correlations), 'Std': np.std(all_correlations), 'Min': np.min(all_correlations), 'Max': np.max(all_correlations), 'Count': len(all_correlations)})
        for prop, corrs in property_stats.items():
            summary_data.append({'Test_Type': 'Absolute', 'Property': prop, 'Metric': 'Pearson_Correlation', 'Mean': np.mean(corrs), 'Std': np.std(corrs), 'Min': np.min(corrs), 'Max': np.max(corrs), 'Count': len(corrs)})
    if relative_results:
        all_aucs = []
        all_accuracies = []
        property_stats = {}
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
all_aucs.append(metrics['auc'])
            all_accuracies.append(metrics['accuracy'])
            prop = data['property']
            if prop not in property_stats:
                property_stats[prop] = {'aucs': [], 'accuracies': []}
            property_stats[prop]['aucs'].append(metrics['auc'])
            property_stats[prop]['accuracies'].append(metrics['accuracy'])
        summary_data.append({'Test_Type': 'Relative', 'Property': 'All', 'Metric': 'AUC', 'Mean': np.mean(all_aucs), 'Std': np.std(all_aucs), 'Min': np.min(all_aucs), 'Max': np.max(all_aucs), 'Count': len(all_aucs)})
        summary_data.append({'Test_Type': 'Relative', 'Property': 'All', 'Metric': 'Accuracy', 'Mean': np.mean(all_accuracies), 'Std': np.std(all_accuracies), 'Min': np.min(all_accuracies), 'Max': np.max(all_accuracies), 'Count': len(all_accuracies)})
        for prop, stats in property_stats.items():
            summary_data.append({'Test_Type': 'Relative', 'Property': prop, 'Metric': 'AUC', 'Mean': np.mean(stats['aucs']), 'Std': np.std(stats['aucs']), 'Min': np.min(stats['aucs']), 'Max': np.max(stats['aucs']), 'Count': len(stats['aucs'])})
            summary_data.append({'Test_Type': 'Relative', 'Property': prop, 'Metric': 'Accuracy', 'Mean': np.mean(stats['accuracies']), 'Std': np.std(stats['accuracies']), 'Min': np.min(stats['accuracies']), 'Max': np.max(stats['accuracies']), 'Count': len(stats['accuracies'])})
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_filename = f'physics_analysis_summary_{timestamp}.csv'
        summary_filepath = os.path.join(output_dir, summary_filename)
        df_summary.to_csv(summary_filepath, index=False, encoding='utf-8')
        print(f'✅ Summary statistical results have been exported to:{summary_filepath}')

def analyze_physics_results(file_path: str) -> None:
    print('=' * 80)
    print('Evaluation of physical analysis results')
    print('=' * 80)
    results = parse_results_file(file_path)
    if not results:
        print('❌ No valid test results found')
        return
    absolute_results = {}
    relative_results = {}
    for test_id, data in results.items():
        if data['type'] == 'absolute':
            absolute_results[test_id] = data
        elif data['type'] == 'relative':
            relative_results[test_id] = data
    if absolute_results:
        print('\n📊 Absolute value test results (Pearson correlation coefficient)')
        print('-' * 50)
        all_correlations = []
        property_correlations = {}
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(data['predicted'], data['ground_truth'])
            all_correlations.append(correlation)
            prop = data['property']
            if prop not in property_correlations:
                property_correlations[prop] = []
            property_correlations[prop].append(correlation)
            print(f"📈 {data['property']} - {data['test_set']} ({data['version']})")
            print(f"Number of samples:{len(data['predicted'])}")
            print(f'Pearson correlation coefficient:{correlation:.4f}')
            print(f'P value:{p_value:.4f}')
            print(f"Relevance:{('powerful' if abs(correlation) > 0.7 else 'medium' if abs(correlation) > 0.3 else 'weak')}")
            print()
        if all_correlations:
            print('📋 Absolute value test summary:')
            print(f'Average correlation coefficient:{np.mean(all_correlations):.4f}')
            print(f'Correlation coefficient range:{np.min(all_correlations):.4f} ~ {np.max(all_correlations):.4f}')
            print(f'Standard deviation:{np.std(all_correlations):.4f}')
            print('\n📋 Average correlation coefficient grouped by property:')
            for prop, corrs in property_correlations.items():
                print(f' {prop}: {np.mean(corrs):.4f} (±{np.std(corrs):.4f})')
    if relative_results:
        print('\n🎯 Relative test results (ROC indicator)')
        print('-' * 50)
        all_aucs = []
        property_aucs = {}
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            all_aucs.append(metrics['auc'])prop = data['property']
            if prop not in property_aucs:
                property_aucs[prop] = []
            property_aucs[prop].append(metrics['auc'])
            print(f"🎯 {data['property']} - {data['test_set']} ({data['version']})")
            print(f"Number of samples:{len(data['predicted'])}")
            print(f" AUC: {metrics['auc']:.4f}")
            print(f"Accuracy:{metrics['accuracy']:.4f}")
            print(f"Sensitivity:{metrics['sensitivity']:.4f}")
            print(f"Specificity:{metrics['specificity']:.4f}")
            print(f"Accuracy:{metrics['precision']:.4f}")
            print(f"F1 score:{metrics['f1_score']:.4f}")
            print()
        if all_aucs:
            print('📋 Relative test summary:')
            print(f'Average AUC:{np.mean(all_aucs):.4f}')
            print(f'AUC range:{np.min(all_aucs):.4f} ~ {np.max(all_aucs):.4f}')
            print(f'Standard deviation:{np.std(all_aucs):.4f}')
            print('\n📋 Average AUC grouped by property:')
            for prop, aucs in property_aucs.items():
                print(f' {prop}: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})')
    create_detailed_results_table(absolute_results, relative_results)
    print('\n📁 Export CSV file...')
    export_to_csv(absolute_results, relative_results)
    print('✅Analysis completed!')

def create_detailed_results_table(absolute_results: Dict, relative_results: Dict) -> None:
    print('\n📋 Detailed result table')
    print('=' * 100)
    if absolute_results:
        print('\nAbsolute value test result:')
        abs_data = []
        for test_id, data in absolute_results.items():
            correlation, p_value = calculate_pearson_correlation(data['predicted'], data['ground_truth'])
            abs_data.append({'Property': data['property'], 'Version': data['version'], 'Test Set': data['test_set'], 'Samples': len(data['predicted']), 'Pearson Correlation': f'{correlation:.4f}', 'P-value': f'{p_value:.4f}', 'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'})
        df_abs = pd.DataFrame(abs_data)
        print(df_abs.to_string(index=False))
    if relative_results:
        print('\nRelative test results:')
        rel_data = []
        for test_id, data in relative_results.items():
            metrics = calculate_roc_metrics(data['predicted'], data['ground_truth'])
            rel_data.append({'Property': data['property'], 'Version': data['version'], 'Test Set': data['test_set'], 'Samples': len(data['predicted']), 'AUC': f"{metrics['auc']:.4f}", 'Accuracy': f"{metrics['accuracy']:.4f}", 'Sensitivity': f"{metrics['sensitivity']:.4f}", 'Specificity': f"{metrics['specificity']:.4f}", 'F1-Score': f"{metrics['f1_score']:.4f}"})
        df_rel = pd.DataFrame(rel_data)
        print(df_rel.to_string(index=False))
if __name__ == '__main__':
    file_path = 'test_mllms_from_api/gemini_physics_analysis_results/summary_results_20250709_080535.txt'
    analyze_physics_results(file_path)