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

class SVDSubsetMetricsCalculator:

    def __init__(self):
        self.gpt_results_dir = 'test_mllms_from_api/gpt_physics_analysis_results'
        self.svd_results_dir = 'scatter_plot_results_5.25'
        self.svd_paths = {'v120': {'test1': '5.25_result_v120test1_375itr', 'test2': '5.25_result_v120test2_375itr', 'test3': '5.25_result_v120test3_375itr'}, 'v121': {'test1': '5.25_result_v121test1_2500itr', 'test2': '5.25_result_v121test2_2500itr', 'test3': '5.25_result_v121test3_2500itr'}, 'v122': {'test1': '5.25_result_v122test1_5125itr', 'test2': '5.25_result_v122test2_5125itr', 'test3': '5.25_result_v122test3_5125itr_16frames'}, 'v123': {'test1': '5.25_result_v123test1_1000itr', 'test2': '5.25_result_v123test2_1000itr', 'test3': '5.25_result_v123test3_1000itr'}, 'v124': {'test1': '5.25_result_v124_test1_3000itr', 'test2': '5.25_result_v124_test2_3000itr', 'test3': '5.25_result_v124_test3_3000itr'}, 'v125': {'test1': '5.25_result_v125_test1_1375itr', 'test2': '5.25_result_v125_test2_1375itr', 'test3': '5.25_result_v125_test3_1375itr'}}
        self.svd_properties = {'v120': 'friction', 'v121': 'friction', 'v122': 'viscosity', 'v123': 'viscosity', 'v124': 'elasticity', 'v125': 'elasticity'}
        self.svd_test_types = {'v120': 'absolute', 'v121': 'relative', 'v122': 'absolute', 'v123': 'relative', 'v124': 'absolute', 'v125': 'relative'}
        self.svd_to_gpt_mapping = {'v120': 'v120', 'v121': 'v121', 'v122': 'v119', 'v123': 'v110', 'v124': 'v124', 'v125': 'v125'}
        self.processing_functions = {'viscosity': self._process_viscosity, 'friction': self._process_friction, 'elasticity': self._process_elasticity}
        self.results = {}

    def _process_viscosity(self, pred: float, is_absolute: bool) -> float:
        if is_absolute:
            return self._sigmoid(pred) * 0.01
        else:
            return 1 if pred > 0 else 0

    def _process_friction(self, pred: float, is_absolute: bool) -> float:
        if is_absolute:
            return self._sigmoid(pred) * 0.2
        else:
            return 1 if pred > 0 else 0

    def _process_elasticity(self, pred: float, is_absolute: bool) -> float:
        return pred

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def _extract_video_id(self, video_name: str) -> str:
        pattern = '(video_\\d+_id_\\d+)'
        match = re.search(pattern, video_name)
        if match:
            return match.group(1)
        print(f"Warning: Unable to retrieve from '{video_name}' Extract video_X_id_Y format")
        return video_name

    def _parse_gpt_absolute_results(self, filepath: str) -> Dict:
        results = {'samples': []}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            pattern = 'video:([^\\n]+)\\n.*?GT value:([^\\n]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            for video_name, gt_str in matches:
                video_name = video_name.strip()
                try:
                    gt = float(gt_str.strip())
                    results['samples'].append({'video': video_name, 'gt': gt})
                except ValueError:
                    print(f"Warning: Unable to parse GT value '{gt_str}' for video{video_name}")
                    continue
        except Exception as e:
            print(f'Parsing GPT absolute value results failed{filepath}: {e}')
        return results

    def _parse_gpt_relative_results(self, filepath: str) -> Dict:
        results = {'comparisons': []}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            pattern = 'Video pair:([^\\n]+) vs ([^\\n]+)\\n.*?GT results:([^\\n]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            for video1, video2, gt_str in matches:
                video1 = video1.strip()
                video2 = video2.strip()
                try:
                    gt = int(gt_str.strip())
                    results['comparisons'].append({'video1': video1, 'video2': video2, 'gt': gt})
                except ValueError:
                    print(f"Warning: Unable to parse GT value '{gt_str}' for comparison{video1} vs {video2}")
                    continue
        except Exception as e:
            print(f'Failed to parse GPT relative comparison results{filepath}: {e}')
        return results

    def _extract_svd_absolute_data(self, version: str, video_names: List[str], test_name: str) -> Dict:
        results = {'version': version, 'samples': []}
        if version not in self.svd_paths:
            print(f'Warning: version not found{version}path mapping')
            return results
        test_number = test_name.replace('test', '')
        test_key = f'test{test_number}'
        if test_key not in self.svd_paths[version]:
            print(f'Warning: version not found{version}of{test_key}path mapping')
            return results
        result_path = self.svd_paths[version][test_key]
        full_path = os.path.join(self.svd_results_dir, result_path)
        if not os.path.exists(full_path):
            print(f'Warning: SVD results directory not found{full_path}')
            return results
        try:
            svd_data = {}
            for filename in os.listdir(full_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(full_path, filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        for key, value in data.items():
                            if 'pred' in value and 'gt' in value:
                                svd_data[key] = {'pred': value['pred'], 'gt': value['gt']}
                    except Exception as e:
                        print(f'Failed to read SVD JSON file{json_path}: {e}')
                        continue
            found_count = 0
            not_found = []
            for video_name in video_names:
                matched = False
                if video_name in svd_data:
                    results['samples'].append({'video': video_name, 'pred': svd_data[video_name]['pred'], 'gt': svd_data[video_name]['gt']})
                    matched = True
                    found_count += 1
                if not matched:
                    video_base = video_name.replace('.mp4', '')
                    for svd_key in svd_data.keys():
                        svd_base = svd_key.replace('.mp4', '')
                        if video_base == svd_base:
                            results['samples'].append({'video': video_name, 'pred': svd_data[svd_key]['pred'], 'gt': svd_data[svd_key]['gt']})
                            matched = True
                            found_count += 1
                            break
                if not matched:
                    video_base = video_name.replace('.mp4', '')
                    video_normalized = video_base.replace('-0.00', '_0.00').replace('_0.00', '_0.0')
                    for svd_key in svd_data.keys():
                        svd_base = svd_key.replace('.mp4', '')
                        svd_normalized = svd_base.replace('-0.00', '_0.00').replace('_0.00', '_0.0')
                        if video_normalized == svd_normalized or video_base in svd_base or svd_base in video_base:
                            results['samples'].append({'video': video_name, 'pred': svd_data[svd_key]['pred'], 'gt': svd_data[svd_key]['gt']})
                            matched = True
                            found_count += 1
                            break
                if not matched:
                    id_match = re.search('id_(\\d+)', video_name)
                    if id_match:
                        video_id = id_match.group(1)
                        for svd_key in svd_data.keys():
                            if f'id_{video_id}' in svd_key:
                                results['samples'].append({'video': video_name, 'pred': svd_data[svd_key]['pred'], 'gt': svd_data[svd_key]['gt']})
                                matched = True
                                found_count += 1
                                break
                if not matched:
                    not_found.append(video_name)
            print(f'Extract SVD absolute value data:{version}, total number of videos:{len(video_names)}, matching successfully:{found_count}, not matched:{len(not_found)}')
            if len(not_found) > 0 and len(not_found) <= 5:
                print(f'Examples of unmatched videos:{not_found[:5]}')
            print(f'There are total in SVD database{len(svd_data)}videos')
        except Exception as e:
            print(f'Failed to extract SVD absolute value data{full_path}: {e}')
        return results

    def _extract_svd_relative_data(self, version: str, video_pairs: List[Tuple[str, str]], test_name: str) -> Dict:
        results = {'version': version, 'comparisons': []}
        if version not in self.svd_paths:
            print(f'Warning: version not found{version}path mapping')
            return results
        test_number = test_name.replace('test', '')
        test_key = f'test{test_number}'
        if test_key not in self.svd_paths[version]:
            print(f'Warning: version not found{version}of{test_key}path mapping')
            return results
        result_path = self.svd_paths[version][test_key]
        full_path = os.path.join(self.svd_results_dir, result_path)
        if not os.path.exists(full_path):
            print(f'Warning: SVD results directory not found{full_path}')
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
                        print(f'Failed to read SVD JSON file{json_path}: {e}')
                        continue
            print(f"Extract SVD relative comparison data:{version}, comparison number:{len(results['comparisons'])}")
        except Exception as e:
            print(f'Failed to extract SVD relative comparison data{full_path}: {e}')
        return results

    def _calculate_pearson(self, preds: List[float], gts: List[float]) -> Tuple[float, float]:
        try:
            correlation, p_value = stats.pearsonr(preds, gts)
            return (correlation, p_value)
        except Exception as e:
            print(f'Failed to calculate Pearson correlation coefficient:{e}')
            return (0.0, 1.0)

    def _calculate_roc_auc(self, preds: List[float], gts: List[int]) -> float:
        try:
            return roc_auc_score(gts, preds)
        except Exception as e:
            print(f'Calculating ROC-AUC failed:{e}')
            return 0.0

    def calculate_metrics(self):
        print('Start calculating the indicators of the SVD model on the GPT test subset...')
        for version, property_name in self.svd_properties.items():
            test_type = self.svd_test_types[version]
            gpt_version = self.svd_to_gpt_mapping[version]
print(f'\n=== Process SVD model {version} ({property_name}, {test_type}) -> GPT {gpt_version} ===')
            gpt_files = []
            for filename in os.listdir(self.gpt_results_dir):
                if filename.startswith(f'{gpt_version}_') and filename.endswith('.txt'):
                    if test_type == 'absolute' and 'absolute' in filename:
                        gpt_files.append(filename)
                    elif test_type == 'relative' and 'relative' in filename:
                        gpt_files.append(filename)
            if not gpt_files:
                print(f'Warning: version not found{gpt_version}of{test_type}GPT test results')
                continue
            for gpt_file in gpt_files:
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                if test_type == 'absolute':
                    gpt_data = self._parse_gpt_absolute_results(gpt_path)
                    print(f"Parse the absolute value test results:{gpt_file.replace('.txt', '')}, number of samples:{len(gpt_data['samples'])}")
                    video_names = [sample['video'] for sample in gpt_data['samples']]
                    test_name_full = gpt_file.replace('.txt', '').replace(f'{gpt_version}_vjepa_', '')
                    test_name = test_name_full.split('_')[0]
                    svd_data = self._extract_svd_absolute_data(version, video_names, test_name)
                    if len(svd_data['samples']) > 0:
                        processed_preds = []
                        processed_gts = []
                        for sample in svd_data['samples']:
                            processed_pred = self.processing_functions[property_name](sample['pred'], True)
                            processed_preds.append(processed_pred)
                            processed_gts.append(sample['gt'])
                        correlation, p_value = self._calculate_pearson(processed_preds, processed_gts)
                        print(f'Absolute value test{property_name}: number of samples ={len(processed_preds)}, Pearson={correlation:.4f}')
                        test_name = gpt_file.replace('.txt', '').replace(f'{gpt_version}_', '')
                        result_key = f'{version}_{property_name}_{test_name}'
                        self.results[result_key] = {'version': version, 'property': property_name, 'test_type': 'absolute', 'test_name': test_name, 'samples': len(processed_preds), 'pearson': correlation, 'p_value': p_value}
                elif test_type == 'relative':
                    gpt_data = self._parse_gpt_relative_results(gpt_path)
                    print(f"Parse relative comparison test results:{gpt_file.replace('.txt', '')}, comparison number:{len(gpt_data['comparisons'])}")
                    video_pairs = [(comp['video1'], comp['video2']) for comp in gpt_data['comparisons']]
                    test_name_full = gpt_file.replace('.txt', '').replace(f'{gpt_version}_vjepa_', '')
                    test_name = test_name_full.split('_')[0]
                    svd_data = self._extract_svd_relative_data(version, video_pairs, test_name)
                    if len(svd_data['comparisons']) > 0:
                        processed_preds = []
                        processed_gts = []
                        for comp in svd_data['comparisons']:
                            processed_pred = self.processing_functions[property_name](comp['pred'], False)
                            processed_preds.append(processed_pred)
                            processed_gts.append(comp['gt'])
                        roc_auc = self._calculate_roc_auc(processed_preds, processed_gts)
                        print(f'relative comparison test{property_name}: comparison number={len(processed_preds)}, ROC-AUC={roc_auc:.4f}')
                        test_name = gpt_file.replace('.txt', '').replace(f'{gpt_version}_', '')
                        result_key = f'{version}_{property_name}_{test_name}'
                        self.results[result_key] = {'version': version, 'property': property_name, 'test_type': 'relative', 'test_name': test_name, 'comparisons': len(processed_preds), 'roc_auc': roc_auc}

    def save_results(self):timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = f'svd_subset_metrics_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        report_filename = f'svd_subset_metrics_report_{timestamp}.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('Indicator calculation results of the SVD model on the GPT test subset\n')
            f.write(f"Generation time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('=' * 80 + '\n\n')
            f.write('=== Absolute value test results (Pearson correlation coefficient) ===\n')
            f.write(f"{'Version':<10} {'property':<12} {'test set':<20} {'Number of samples':<8} {'Pearson':<10} {'P value':<10}\n")
            f.write('-' * 80 + '\n')
            for key, result in self.results.items():
                if result['test_type'] == 'absolute':
                    f.write(f"{result['version']:<10} {result['property']:<12} {result['test_name']:<20} {result['samples']:<8} {result['pearson']:<10.4f} {result['p_value']:<10.4f}\n")
            f.write('\n')
            f.write('=== relative comparison test results (ROC-AUC) ===\n')
            f.write(f"{'Version':<10} {'property':<12} {'test set':<20} {'Comparative number':<8} {'ROC-AUC':<10}\n")
            f.write('-' * 60 + '\n')
            for key, result in self.results.items():
                if result['test_type'] == 'relative':
                    f.write(f"{result['version']:<10} {result['property']:<12} {result['test_name']:<20} {result['comparisons']:<8} {result['roc_auc']:<10.4f}\n")
            f.write('\n')
            f.write('=== Average results grouped by property ===\n')
            properties = set((result['property'] for result in self.results.values()))
            for prop in sorted(properties):
                prop_results = [r for r in self.results.values() if r['property'] == prop]
                absolute_results = [r for r in prop_results if r['test_type'] == 'absolute']
                relative_results = [r for r in prop_results if r['test_type'] == 'relative']
                avg_pearson = np.mean([r['pearson'] for r in absolute_results]) if absolute_results else 0
                avg_roc_auc = np.mean([r['roc_auc'] for r in relative_results]) if relative_results else 0
                f.write(f'{prop:<12}Average Pearson:{avg_pearson:.4f}, average ROC-AUC:{avg_roc_auc:.4f}\n')
            f.write('\n' + '=' * 80 + '\n')
            f.write('Report generation completed\n')
            f.write('=' * 80 + '\n')
        print(f'Report saved to{report_filename}')
        print(f'Detailed results have been saved to{json_filename}')
        return (report_filename, json_filename)

    def generate_summary(self):
        print('\n' + '=' * 50)
        print('Calculation completed! Summary of results:')
        print('=' * 50)
        total_absolute = sum((1 for r in self.results.values() if r['test_type'] == 'absolute'))
        total_relative = sum((1 for r in self.results.values() if r['test_type'] == 'relative'))
        print(f'Total completed:{total_absolute}an absolute value test,{total_relative}relative comparison test')
        properties = set((result['property'] for result in self.results.values()))
        for prop in sorted(properties):
            print(f'\n{prop.upper()}Test results:')
            prop_results = [r for r in self.results.values() if r['property'] == prop]
            for result in prop_results:
                if result['test_type'] == 'absolute':
                    print(f" {result['version']} {result['test_name']}: Pearson={result['pearson']:.4f}(Number of samples ={result['samples']})")
                else:
                    print(f" {result['version']} {result['test_name']}: ROC-AUC={result['roc_auc']:.4f}(Comparison number={result['comparisons']})")

def main():
    calculator = SVDSubsetMetricsCalculator()
    calculator.calculate_metrics()
    report_file, json_file = calculator.save_results()
    calculator.generate_summary()
    return (report_file, json_file)
if __name__ == '__main__':
    main()