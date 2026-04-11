import os
import json
import re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VjepaSubsetMetricsCalculator:

    def __init__(self):
        self.gpt_results_dir = 'test_mllms_from_api/gpt_physics_analysis_results'
        self.vjepa_results_dir = 'scatter_plot_results_5.25'
        self.vjepa_paths = {'v110': '5.25_result_v110_vjepa_test1_20000itr', 'v119': '5.25_result_v119vjepa_test1_25000itr', 'v120': '5.25_result_v120vjepa_test1_25000itr', 'v121': '5.25_result_v121_vjepa_test1_2000itr-v2', 'v124': '5.25_result_v124_vjepa_test1_3250itr', 'v125': '5.25_result_v125_vjepa_test1_2250itr'}
        self.special_paths = {'v119_test3': '5.25_result_v119vjepa_test3_25000itr_16frames'}
        self.processing_functions = {'viscosity': self._process_viscosity, 'friction': self._process_friction, 'elasticity': self._process_elasticity}
        self.results = {}

    def _sigmoid(self, x: float) -> float:
        try:
            return 1 / (1 + np.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _process_viscosity(self, pred: float, is_absolute: bool=True) -> float:
        if is_absolute:
            return self._sigmoid(pred) * 0.01
        else:
            return 1.0 if pred > 0 else 0.0

    def _process_friction(self, pred: float, is_absolute: bool=True) -> float:
        if is_absolute:
            return self._sigmoid(pred) * 0.2
        else:
            return 1.0 if pred > 0 else 0.0

    def _process_elasticity(self, pred: float, is_absolute: bool=True) -> float:
        return pred

    def _parse_gpt_absolute_results(self, file_path: str) -> Dict:
        results = {'version': '', 'property': '', 'test_set': '', 'samples': []}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            version_match = re.search('Version:(\\w+)', content)
            property_match = re.search('property:(\\w+)', content)
            test_set_match = re.search('Test set:(\\w+)', content)
            if version_match:
                results['version'] = version_match.group(1)
            if property_match:
                results['property'] = property_match.group(1)
            if test_set_match:
                results['test_set'] = test_set_match.group(1)
            sample_pattern = '--- sample \\d+ ---\\s*video:([^\\s]+)\\s*Predicted value:([^\\s]+)\\s*GT value:([^\\s]+)'
            samples = re.findall(sample_pattern, content)
            for video_name, pred_str, gt_str in samples:
                try:
                    pred = float(pred_str)
                    gt = float(gt_str)
                    results['samples'].append({'video': video_name, 'pred': pred, 'gt': gt})
                except ValueError:
                    print(f'Warning: Unable to parse sample data{video_name}')
                    continue
            print(f"Parse the absolute value test results:{results['version']} {results['property']} {results['test_set']}, number of samples:{len(results['samples'])}")
        except Exception as e:
            print(f'Parsing GPT absolute value result file failed{file_path}: {e}')
        return results

    def _parse_gpt_relative_results(self, file_path: str) -> Dict:
        results = {'version': '', 'property': '', 'test_set': '', 'comparisons': []}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            version_match = re.search('Version:(\\w+)', content)
            property_match = re.search('property:(\\w+)', content)
            test_set_match = re.search('Test set:(\\w+)', content)
            if version_match:
                results['version'] = version_match.group(1)
            if property_match:
                results['property'] = property_match.group(1)
            if test_set_match:
                results['test_set'] = test_set_match.group(1)comparison_pattern = '--- comparison \\d+ ---\\s*Video pair:([^\\s]+) vs ([^\\s]+)\\s*Predicted results:([^\\s]+)\\s*GT results:([^\\s]+)'
            comparisons = re.findall(comparison_pattern, content)
            for video1, video2, pred_str, gt_str in comparisons:
                try:
                    pred = int(pred_str)
                    gt = int(gt_str)
                    results['comparisons'].append({'video1': video1, 'video2': video2, 'pred': pred, 'gt': gt})
                except ValueError:
                    print(f'Warning: Unable to parse comparison data{video1} vs {video2}')
                    continue
            print(f"Parse relative comparison test results:{results['version']} {results['property']} {results['test_set']}, comparison number:{len(results['comparisons'])}")
        except Exception as e:
            print(f'Failed to parse GPT relative comparison result file{file_path}: {e}')
        return results

    def _get_vjepa_path(self, version: str, test_set: str) -> str:
        special_key = f'{version}_{test_set}'
        if special_key in self.special_paths:
            return self.special_paths[special_key]
        if version not in self.vjepa_paths:
            return ''
        base_path = self.vjepa_paths[version]
        result_path = base_path.replace('test1', test_set)
        return result_path

    def _extract_vjepa_absolute_data(self, version: str, test_set: str, video_names: List[str]) -> Dict:
        results = {'version': version, 'test_set': test_set, 'samples': []}
        result_path = self._get_vjepa_path(version, test_set)
        if not result_path:
            print(f'Warning: version not found{version}path mapping')
            return results
        full_path = os.path.join(self.vjepa_results_dir, result_path)
        results_file = os.path.join(full_path, 'results.txt')
        if not os.path.exists(results_file):
            print(f'Warning: vjepa results file not found{results_file}')
            return results
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
            file_pattern = 'File: ([^,]+), Prediction: ([^,]+), GroundTruth: ([^\\s]+)'
            matches = re.findall(file_pattern, content)
            vjepa_data = {}
            for filename, pred_str, gt_str in matches:
                try:
                    pred = float(pred_str)
                    gt = float(gt_str)
                    vjepa_data[filename] = {'pred': pred, 'gt': gt}
                except ValueError:
                    continue
            for video_name in video_names:
                if video_name in vjepa_data:
                    results['samples'].append({'video': video_name, 'pred': vjepa_data[video_name]['pred'], 'gt': vjepa_data[video_name]['gt']})
                else:
                    print(f'Warning: Video not found in vjepa results{video_name}')
            print(f"Extract vjepa absolute value data:{version} {test_set}, number of samples:{len(results['samples'])}")
        except Exception as e:
            print(f'Failed to extract vjepa absolute value data{full_path}: {e}')
        return results

    def _extract_video_id(self, video_name: str) -> str:
        pattern = '(video_\\d+_id_\\d+)'
        match = re.search(pattern, video_name)
        if match:
            return match.group(1)
        pattern_old = '(video_\\d+_id_\\d+)'
        match_old = re.search(pattern_old, video_name)
        if match_old:
            return match_old.group(1)
        print(f"Warning: Unable to retrieve from '{video_name}' Extract video_X_id_Y format")
        return video_name

    def _extract_vjepa_relative_data(self, version: str, test_set: str, video_pairs: List[Tuple[str, str]]) -> Dict:
        results = {'version': version, 'test_set': test_set, 'comparisons': []}
        result_path = self._get_vjepa_path(version, test_set)
        if not result_path:
            print(f'Warning: version not found{version}path mapping')
            return results
        full_path = os.path.join(self.vjepa_results_dir, result_path)
        if not os.path.exists(full_path):            print(f'Warning: vjepa results directory not found{full_path}')
            return results
        try:
            target_pairs = []
            for video1, video2 in video_pairs:
                id1 = self._extract_video_id(video1)
                id2 = self._extract_video_id(video2)
                target_pairs.append((id1, id2))
            for filename in os.listdir(full_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(full_path, filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for key, value in data.items():
                            if '--' in key:
                                parts = key.split('--')
                                if len(parts) == 2:
                                    video1, video2 = parts
                                    for target_id1, target_id2 in target_pairs:
                                        if video1 == target_id1 and video2 == target_id2 or (video1 == target_id2 and video2 == target_id1):
                                            results['comparisons'].append({'video1': video1, 'video2': video2, 'pred': value['pred'], 'gt': value['gt']})
                                            break
                    except Exception as e:
                        print(f'Failed to read json file{json_path}: {e}')
                        continue
            print(f"Extract vjepa relative comparison data:{version} {test_set}, comparison number:{len(results['comparisons'])}")
        except Exception as e:
            print(f'Failed to extract vjepa relative comparison data{full_path}: {e}')
        return results

    def _calculate_pearson_correlation(self, predictions: List[float], ground_truths: List[float]) -> Tuple[float, float]:
        if len(predictions) != len(ground_truths) or len(predictions) < 2:
            return (0.0, 1.0)
        try:
            corr, p_val = stats.pearsonr(predictions, ground_truths)
            return (corr, p_val)
        except Exception as e:
            print(f'Failed to calculate Pearson correlation coefficient:{e}')
            return (0.0, 1.0)

    def _calculate_roc_auc(self, predictions: List[float], ground_truths: List[int]) -> float:
        if len(predictions) != len(ground_truths) or len(predictions) < 2:
            return 0.0
        try:
            if len(set(ground_truths)) < 2:
                print('Warning: No change in label, unable to calculate ROC-AUC')
                return 0.0
            auc = roc_auc_score(ground_truths, predictions)
            return auc
        except Exception as e:
            print(f'Calculating ROC-AUC failed:{e}')
            return 0.0

    def process_model_results(self, version: str):
        print(f'\n=== Processing model{version} ===')
        model_results = {'version': version, 'absolute': {}, 'relative': {}}
        for test_set in ['test1', 'test2', 'test3']:
            for property_name in ['viscosity', 'friction', 'elasticity']:
                gpt_file = f'{version}_vjepa_{test_set}_absolute_results.txt'
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                if os.path.exists(gpt_path):
                    gpt_results = self._parse_gpt_absolute_results(gpt_path)
                    if gpt_results['samples'] and gpt_results['property'] == property_name:
                        video_names = [sample['video'] for sample in gpt_results['samples']]
                        vjepa_data = self._extract_vjepa_absolute_data(version, test_set, video_names)
                        if vjepa_data['samples']:
                            processed_preds = []
                            gts = []
                            for sample in vjepa_data['samples']:
                                processed_pred = self.processing_functions[property_name](sample['pred'], is_absolute=True)
                                processed_preds.append(processed_pred)
                                gts.append(sample['gt'])
corr, p_val = self._calculate_pearson_correlation(processed_preds, gts)
                            key = f'{property_name}_{test_set}'
                            model_results['absolute'][key] = {'property': property_name, 'test_set': test_set, 'sample_count': len(vjepa_data['samples']), 'pearson_corr': corr, 'p_value': p_val}
                            print(f"Absolute value test{property_name} {test_set}: number of samples ={len(vjepa_data['samples'])}, Pearson={corr:.4f}")
        for test_set in ['test1', 'test2', 'test3']:
            for property_name in ['viscosity', 'friction', 'elasticity']:
                gpt_file = f'{version}_vjepa_{test_set}_relative_results.txt'
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                if os.path.exists(gpt_path):
                    gpt_results = self._parse_gpt_relative_results(gpt_path)
                    if gpt_results['comparisons'] and gpt_results['property'] == property_name:
                        video_pairs = [(comp['video1'], comp['video2']) for comp in gpt_results['comparisons']]
                        vjepa_data = self._extract_vjepa_relative_data(version, test_set, video_pairs)
                        if vjepa_data['comparisons']:
                            processed_preds = []
                            gts = []
                            for comp in vjepa_data['comparisons']:
                                processed_pred = self.processing_functions[property_name](comp['pred'], is_absolute=False)
                                processed_preds.append(processed_pred)
                                gts.append(comp['gt'])
                            auc = self._calculate_roc_auc(processed_preds, gts)
                            key = f'{property_name}_{test_set}'
                            model_results['relative'][key] = {'property': property_name, 'test_set': test_set, 'comparison_count': len(vjepa_data['comparisons']), 'roc_auc': auc}
                            print(f"relative comparison test{property_name} {test_set}: comparison number={len(vjepa_data['comparisons'])}, ROC-AUC={auc:.4f}")
        self.results[version] = model_results

    def generate_report(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_lines = []
        report_lines.append('=' * 80)
        report_lines.append(f'Indicator calculation results of the VJEPA model on the GPT test subset')
        report_lines.append(f"Generation time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append('=' * 80)
        report_lines.append('\n=== Absolute value test results (Pearson correlation coefficient) ===')
        report_lines.append(f"{'Version':<10} {'property':<12} {'test set':<8} {'Number of samples':<8} {'Pearson':<10} {'P value':<10}")
        report_lines.append('-' * 70)
        for version in sorted(self.results.keys()):
            model_results = self.results[version]
            for key, result in sorted(model_results['absolute'].items()):
                report_lines.append(f"{version:<10} {result['property']:<12} {result['test_set']:<8} {result['sample_count']:<8} {result['pearson_corr']:<10.4f} {result['p_value']:<10.4f}")
        report_lines.append('\n=== relative comparison test results (ROC-AUC) ===')
        report_lines.append(f"{'Version':<10} {'property':<12} {'test set':<8} {'Comparative number':<8} {'ROC-AUC':<10}")
        report_lines.append('-' * 60)
        for version in sorted(self.results.keys()):
            model_results = self.results[version]
            for key, result in sorted(model_results['relative'].items()):
                report_lines.append(f"{version:<10} {result['property']:<12} {result['test_set']:<8} {result['comparison_count']:<8} {result['roc_auc']:<10.4f}")
        report_lines.append('\n=== Average results grouped by property ===')
        property_stats = {}
        for version in self.results.keys():
            model_results = self.results[version]
            for key, result in model_results['absolute'].items():
                prop = result['property']
                if prop not in property_stats:
                    property_stats[prop] = {'absolute_pearson': [], 'relative_auc': []}                property_stats[prop]['absolute_pearson'].append(result['pearson_corr'])
            for key, result in model_results['relative'].items():
                prop = result['property']
                if prop not in property_stats:
                    property_stats[prop] = {'absolute_pearson': [], 'relative_auc': []}
                property_stats[prop]['relative_auc'].append(result['roc_auc'])
        for prop, stats in sorted(property_stats.items()):
            avg_pearson = np.mean(stats['absolute_pearson']) if stats['absolute_pearson'] else 0.0
            avg_auc = np.mean(stats['relative_auc']) if stats['relative_auc'] else 0.0
            report_lines.append(f'{prop:<12}Average Pearson:{avg_pearson:.4f}, average ROC-AUC:{avg_auc:.4f}')
        report_lines.append('\n' + '=' * 80)
        report_lines.append('Report generation completed')
        report_lines.append('=' * 80)
        return '\n'.join(report_lines)

    def save_detailed_results(self, filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f'Detailed results have been saved to{filename}')
        except Exception as e:
            print(f'Failed to save detailed results:{e}')

    def run(self):
        print('Start calculating the indicators of the vjepa model on the GPT test subset...')
        for version in self.vjepa_paths.keys():
            try:
                self.process_model_results(version)
            except Exception as e:
                print(f'Processing model{version}An error occurred:{e}')
                continue
        report = self.generate_report()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'vjepa_subset_metrics_report_{timestamp}.txt'
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f'\nReport saved to{report_filename}')
        except Exception as e:
            print(f'Failed to save report:{e}')
        detailed_filename = f'vjepa_subset_metrics_{timestamp}.json'
        self.save_detailed_results(detailed_filename)
        print('\n' + '=' * 50)
        print('Calculation completed! Summary of results:')
        print('=' * 50)
        print(report)
if __name__ == '__main__':
    calculator = VjepaSubsetMetricsCalculator()
    calculator.run()
