import os
import glob
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from gpt_physics_analyzer import GPTPhysicsAnalyzer

class RetryFailedPredictionsRelative:

    def __init__(self):
        self.analyzer = GPTPhysicsAnalyzer()
        self.property_type = 'elasticity'
        self.analysis_type = 'relative'
        self.experiment_configs = {'gemini_physics_analysis_results_elasticity_relative_20samples': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': False, 'concat_mode': False}, 'gemini_physics_analysis_results_elasticity_relative_20samples_frameindex': {'frame_index': True, 'icl_examples': None, 'follow_oracle_test': False, 'concat_mode': False}, 'gemini_physics_analysis_results_elasticity_relative_20samples_icl': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': False, 'concat_mode': False}, 'gemini_physics_analysis_results_elasticity_relative_20samples_oracle': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': True, 'concat_mode': False}, 'gemini_physics_analysis_results_elasticity_relative_20samples_concat': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': False, 'concat_mode': True}, 'gemini_physics_analysis_results_relative_100samples_oracle': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': True, 'concat_mode': False}}

    def parse_result_file(self, result_file: str) -> List[Dict]:
        samples = []
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = '--- comparison (\\d+) ---\\n.*?Video pair:(.*?)\\n.*?Predicted results:(.*?)\\n.*?GT results:(.*?)\\n.*?GT value:(.*?)\\n.*?GPT full reply:\\n-{40}\\n(.*?)\\n-{40}'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            comparison_id, video_pair, predicted_str, gt_str, gt_values, gpt_response = match
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
            video_pair_parts = video_pair.split(' vs ')
            video1 = video_pair_parts[0].strip() if len(video_pair_parts) > 0 else ''
            video2 = video_pair_parts[1].strip() if len(video_pair_parts) > 1 else ''
            samples.append({'comparison_id': int(comparison_id), 'video_pair': video_pair.strip(), 'video1': video1, 'video2': video2, 'predicted': predicted, 'gt': gt, 'gt_values': gt_values.strip(), 'gpt_response': gpt_response.strip(), 'is_failed': predicted is None})
        return samples

    def get_data_directory_for_test(self, test_name: str) -> str:
        test_mapping = {'Test-1': '/path/to/shared/genesis_data/5.24_elasticity_synthetic_test1', 'Test-2': '/path/to/shared/genesis_data/5.24_elasticity_synthetic_test2', 'Test-3': '/path/to/shared/genesis_data/2.16_elasticity_test3'}
        return test_mapping.get(test_name)

    def get_test_name_from_filename(self, filename: str) -> str:
        if 'test1' in filename:
            return 'Test-1'
        elif 'test2' in filename:
            return 'Test-2'
        elif 'test3' in filename:
            return 'Test-3'
        else:
            return 'Unknown'

    def generate_icl_examples(self, data_dir: str, num_examples: int=3) -> List[Dict]:
        try:
            icl_examples = self.analyzer.sample_icl_examples(train_dir=data_dir, property_type=self.property_type, analysis_type=self.analysis_type, dataset_class='WebVidv3ElasticityTest', num_examples=num_examples, frame_size=(224, 224))
            return icl_examples
        except Exception as e:
            print(f'Failed to generate ICL example:{e}')
            return []

    def retry_failed_comparison(self, sample: Dict, data_dir: str, config: Dict) -> Optional[str]:
        video1_path = os.path.join(data_dir, sample['video1'])video2_path = os.path.join(data_dir, sample['video2'])
        if not os.path.exists(video1_path):
            print(f'Video file 1 does not exist:{video1_path}')
            return None
        if not os.path.exists(video2_path):
            print(f'Video file 2 does not exist:{video2_path}')
            return None
        try:
            icl_examples = None
            if config['icl_examples'] is not None:
                if len(config['icl_examples']) == 0:
                    icl_examples = self.generate_icl_examples(data_dir, num_examples=3)
                else:
                    icl_examples = config['icl_examples']
            gpt_result = self.analyzer.compare_two_videos_with_gpt(video_path1=video1_path, video_path2=video2_path, property_type=self.property_type, max_frames_per_video=16, frame_size=(224, 224), model_version='gpt-4o', concat_mode=config['concat_mode'], frame_index=config['frame_index'], icl_examples=icl_examples, follow_oracle_test=config['follow_oracle_test'])
            predicted_comparison = self.analyzer.extract_comparison_result(gpt_result)
            if predicted_comparison is not None:
                print(f'✓ Retest successful:{predicted_comparison}')
                return gpt_result
            else:
                print(f'✗ The retest still failed and the comparison results could not be extracted.')
                return None
        except Exception as e:
            print(f'✗ Error in retesting:{e}')
            return None

    def update_result_file(self, result_file: str, samples: List[Dict], updated_samples: Dict[int, str]) -> None:
        if not updated_samples:
            print('There are no samples to update')
            return
        backup_file = result_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(result_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        print(f'Backup created:{backup_file}')
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        for comparison_id, new_gpt_response in updated_samples.items():
            new_predicted = self.analyzer.extract_comparison_result(new_gpt_response)
            if new_predicted is not None:
                pattern = f'(--- Compare{comparison_id} ---\\n.*?Video pair:.*?\\n.*?)Predicted results:None(\\n.*?GT results:.*?\\n.*?GT value:.*?\\n.*?GPT full reply:\\n-{{40}}\\n).*?(\\n-{{40}})'
                replacement = f'\\g<1>Prediction results:{new_predicted}\\g<2>{new_gpt_response}\\g<3>'
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'✓ Updated{len(updated_samples)}a comparison result')

    def process_experiment_directory(self, exp_dir: str) -> None:
        print(f'\nProcessing experiment directory: {exp_dir}')
        exp_name = os.path.basename(exp_dir)
        if exp_name not in self.experiment_configs:
            print(f'Warning: Unknown experimental configuration:{exp_name}')
            return
        config = self.experiment_configs[exp_name]
        print(f"Configuration: frame_index={config['frame_index']}, icl={('yes' if config['icl_examples'] is not None else 'no')}, oracle={('yes' if config['follow_oracle_test'] else 'no')}, concat={('yes' if config['concat_mode'] else 'no')}")
        result_files = glob.glob(os.path.join(exp_dir, '*relative_results.txt'))
        if not result_files:
            print(f'Warning: Results file not found')
            return
        total_retried = 0
        total_successful = 0
        for result_file in result_files:
            filename = os.path.basename(result_file)
            test_name = self.get_test_name_from_filename(filename)
            data_dir = self.get_data_directory_for_test(test_name)
            if not data_dir or not os.path.exists(data_dir):
                print(f'Warning: data directory does not exist:{data_dir}')
                continue
            print(f'Process files:{filename} ({test_name})')
            samples = self.parse_result_file(result_file)            failed_samples = [s for s in samples if s['is_failed']]
            if not failed_samples:
                print(f'no failed comparison')
                continue
            print(f'turn up{len(failed_samples)}failed comparison')
            updated_samples = {}
            for sample in failed_samples:
                print(f"test-retest comparison{sample['comparison_id']}: {sample['video1'][:50]}... vs {sample['video2'][:50]}...")
                new_gpt_response = self.retry_failed_comparison(sample, data_dir, config)
                if new_gpt_response:
                    updated_samples[sample['comparison_id']] = new_gpt_response
                    total_successful += 1
                total_retried += 1
                time.sleep(15)
            if updated_samples:
                self.update_result_file(result_file, samples, updated_samples)
            print(f'Finish:{len(updated_samples)}/{len(failed_samples)}A comparison retest was successful')
        print(f'Experiment directory processing completed: Total retests{total_retried}comparison, success{total_successful}indivual')

    def dry_run_analysis(self, target_experiments: Optional[List[str]]=None) -> None:
        print('🔍 Trial run mode: count the number of failed comparisons')
        all_experiments = list(self.experiment_configs.keys())
        if target_experiments:
            valid_experiments = []
            for exp in target_experiments:
                if exp in all_experiments and os.path.exists(exp):
                    valid_experiments.append(exp)
                else:
                    print(f'Warning: Experiment directory does not exist or configuration is unknown:{exp}')
            experiments_to_process = valid_experiments
        else:
            experiments_to_process = [exp for exp in all_experiments if os.path.exists(exp)]
        if not experiments_to_process:
            print('❌ No processable experiment directory found')
            return
        print(f'analyze{len(experiments_to_process)}Experiment directory:')
        for exp in experiments_to_process:
            print(f'  - {exp}')
        print()
        total_experiments = 0
        total_files = 0
        total_comparisons = 0
        total_failed = 0
        experiment_stats = []
        for exp_dir in experiments_to_process:
            exp_name = os.path.basename(exp_dir)
            if exp_name not in self.experiment_configs:
                continue
            print(f'Analytical experiments:{exp_name}')
            result_files = glob.glob(os.path.join(exp_dir, '*relative_results.txt'))
            if not result_files:
                print(f'Warning: Results file not found')
                continue
            exp_total_comparisons = 0
            exp_total_failed = 0
            file_stats = []
            for result_file in result_files:
                filename = os.path.basename(result_file)
                test_name = self.get_test_name_from_filename(filename)
                samples = self.parse_result_file(result_file)
                failed_samples = [s for s in samples if s['is_failed']]
                file_stats.append({'filename': filename, 'test_name': test_name, 'total_comparisons': len(samples), 'failed_comparisons': len(failed_samples), 'success_rate': (len(samples) - len(failed_samples)) / len(samples) * 100 if samples else 0})
                exp_total_comparisons += len(samples)
                exp_total_failed += len(failed_samples)
                print(f'  {test_name}: {len(failed_samples)}/{len(samples)}fail ({(len(samples) - len(failed_samples)) / len(samples) * 100:.1f}% success rate)')
            experiment_stats.append({'experiment': exp_name, 'total_comparisons': exp_total_comparisons, 'failed_comparisons': exp_total_failed, 'success_rate': (exp_total_comparisons - exp_total_failed) / exp_total_comparisons * 100 if exp_total_comparisons else 0, 'file_stats': file_stats})
            total_experiments += 1
            total_files += len(result_files)
            total_comparisons += exp_total_comparisons
            total_failed += exp_total_failed
print(f'Experiment summary:{exp_total_failed}/{exp_total_comparisons}fail ({(exp_total_comparisons - exp_total_failed) / exp_total_comparisons * 100:.1f}% success rate)')
            print()
        print('=' * 80)
        print('Trial run statistics summary')
        print('=' * 80)
        print(f'Total number of experiments:{total_experiments}')
        print(f'Total number of files:{total_files}')
        print(f'Total number of comparisons:{total_comparisons}')
        print(f'Number of failed comparisons:{total_failed}')
        print(f'Overall success rate:{(total_comparisons - total_failed) / total_comparisons * 100:.1f}%')
        print()
        experiment_stats.sort(key=lambda x: x['success_rate'])
        print('Experiment success rate ranking (from low to high):')
        print('-' * 80)
        for i, exp_stat in enumerate(experiment_stats, 1):
            print(f"{i:2d}. {exp_stat['experiment']:<60} {exp_stat['failed_comparisons']:3d}/{exp_stat['total_comparisons']:3d}fail ({exp_stat['success_rate']:5.1f}% success)")
        print()
        print('Number of comparisons that need to be retested:')
        print('-' * 40)
        for exp_stat in experiment_stats:
            if exp_stat['failed_comparisons'] > 0:
                print(f"{exp_stat['experiment']:<60} {exp_stat['failed_comparisons']:3d}comparison")
        print(f'\ntotal needs to be retested: {total_failed}comparison')
        print(f'Estimated retest time:{total_failed * 15 / 60:.1f}minutes (calculated as 15 seconds per comparison)')

    def run(self, target_experiments: Optional[List[str]]=None) -> None:
        print('Start comparing predictions for failed retests...')
        all_experiments = list(self.experiment_configs.keys())
        if target_experiments:
            valid_experiments = []
            for exp in target_experiments:
                if exp in all_experiments and os.path.exists(exp):
                    valid_experiments.append(exp)
                else:
                    print(f'Warning: Experiment directory does not exist or configuration is unknown:{exp}')
            experiments_to_process = valid_experiments
        else:
            experiments_to_process = [exp for exp in all_experiments if os.path.exists(exp)]
        if not experiments_to_process:
            print('❌ No processable experiment directory found')
            return
        print(f'will handle{len(experiments_to_process)}Experiment directory:')
        for exp in experiments_to_process:
            print(f' - {exp}')
        total_start_time = time.time()
        for exp_dir in experiments_to_process:
            try:
                self.process_experiment_directory(exp_dir)
            except Exception as e:
                print(f'❌ Processing Experiment Directory{exp_dir}An error occurred:{e}')
                continue
        total_time = time.time() - total_start_time
        print(f'\n✅ The retest process is completed, the total time taken: {total_time:.1f}Second')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Retest failed predictions in elasticity relative ablation experiments')
    parser.add_argument('--experiments', nargs='*', help='Specify the name of the experiment directory to be processed. If not specified, all experiments will be processed.')
    parser.add_argument('--dry-run', action='store_true', help='Trial run mode only counts the number of failed comparisons and does not actually retest.')
    args = parser.parse_args()
    retrier = RetryFailedPredictionsRelative()
    if args.dry_run:
        retrier.dry_run_analysis(target_experiments=args.experiments)
        return
    retrier.run(target_experiments=args.experiments)
if __name__ == '__main__':
    main()