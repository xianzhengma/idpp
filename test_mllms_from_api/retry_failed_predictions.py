import os
import glob
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from gpt_physics_analyzer import GPTPhysicsAnalyzer

class RetryFailedPredictions:

    def __init__(self):
        self.analyzer = GPTPhysicsAnalyzer()
        self.property_type = 'elasticity'
        self.analysis_type = 'absolute'
        self.experiment_configs = {'gemini_physics_analysis_results_elasticity_absolute_20samples': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': False}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex': {'frame_index': True, 'icl_examples': None, 'follow_oracle_test': False}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_icl': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': False}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_oracle': {'frame_index': False, 'icl_examples': None, 'follow_oracle_test': True}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl': {'frame_index': True, 'icl_examples': [], 'follow_oracle_test': False}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_oracle': {'frame_index': True, 'icl_examples': None, 'follow_oracle_test': True}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_icl_oracle': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': True}, 'gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl_oracle': {'frame_index': True, 'icl_examples': [], 'follow_oracle_test': True}, 'gemini_physics_analysis_results_absolute_100samples_icl': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': False}, 'gpt4o_physics_analysis_results_absolute_100samples_icl': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': False}, 'qwen_physics_analysis_results_absolute_100samples_icl': {'frame_index': False, 'icl_examples': [], 'follow_oracle_test': False}}

    def parse_result_file(self, result_file: str) -> List[Dict]:
        samples = []
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = '--- sample (\\d+) ---\\n.*?video:(.*?)\\n.*?Predicted value:(.*?)\\n.*?GT value:(.*?)\\n.*?GPT full reply:\\n-{40}\\n(.*?)\\n-{40}'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            sample_id, video, predicted_str, gt_str, gpt_response = match
            predicted=None
            if predicted_str != 'None' and predicted_str.strip():
                try:
                    predicted = float(predicted_str)
                except ValueError:
                    predicted=None
            gt=None
            if gt_str != 'None' and gt_str.strip():
                try:
                    gt = float(gt_str)
                except ValueError:
                    gt=None
            samples.append({'sample_id': int(sample_id), 'video': video.strip(), 'predicted': predicted, 'gt': gt, 'gpt_response': gpt_response.strip(), 'is_failed': predicted is None})
        return samples

    def get_data_directory_for_test(self, test_name: str, filename: str='') -> str:
        if 'v119' in filename:
            test_mapping = {'Test-1': '/path/to/shared/genesis_data/5.2_viscosity_synthetic_test1', 'Test-2': '/path/to/shared/genesis_data/5.2_viscosity_synthetic_test2', 'Test-3': '/path/to/shared/genesis_data/7.15_new_viscosity_video_clips_16frames_merged'}
        elif 'v120' in filename:
            test_mapping = {'Test-1': '/path/to/shared/genesis_data/7.18_friction_synthetic_test1', 'Test-2': '/path/to/shared/genesis_data/7.18_friction_synthetic_test2', 'Test-3': '/path/to/shared/genesis_data/7.18_friction_test3_rgb_seg_videos_152frames_abs_test'}
        elif 'v124' in filename:
            test_mapping = {'Test-1': '/path/to/shared/genesis_data/4.11_mpm_bounce_ball_videos_test1_sorted', 'Test-2': '/path/to/shared/genesis_data/4.11_mpm_bounce_ball_videos_test2_sorted', 'Test-3': '/path/to/shared/genesis_data/7.15_new_elasticity_video_clips_16frames'}
        else:            test_mapping = {'Test-1': '/path/to/shared/genesis_data/4.11_mpm_bounce_ball_videos_test1_sorted', 'Test-2': '/path/to/shared/genesis_data/4.11_mpm_bounce_ball_videos_test2_sorted', 'Test-3': '/path/to/shared/genesis_data/4.11_mpm_bounce_ball_videos_test3_sorted'}
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

    def generate_icl_examples(self, data_dir: str, filename: str='', num_examples: int=3) -> List[Dict]:
        try:
            if 'v119' in filename:
                property_type = 'viscosity'
                dataset_class = 'WebVidv2Viscosity'
            elif 'v120' in filename:
                property_type = 'friction'
                dataset_class = 'WebVidv3FrictionTest'
            elif 'v124' in filename:
                property_type = 'elasticity'
                dataset_class = 'WebVidv2Viscosity'
            else:
                property_type = 'elasticity'
                dataset_class = 'WebVidv2Viscosity'
            icl_examples = self.analyzer.sample_icl_examples(train_dir=data_dir, property_type=property_type, analysis_type=self.analysis_type, dataset_class=dataset_class, num_examples=num_examples, frame_size=(224, 224))
            return icl_examples
        except Exception as e:
            print(f'Failed to generate ICL example:{e}')
            return []

    def retry_failed_sample(self, sample: Dict, data_dir: str, config: Dict, filename: str='') -> Optional[str]:
        video_path = os.path.join(data_dir, sample['video'])
        if not os.path.exists(video_path):
            print(f'Video file does not exist:{video_path}')
            return None
        try:
            if 'v119' in filename:
                property_type = 'viscosity'
            elif 'v120' in filename:
                property_type = 'friction'
            elif 'v124' in filename:
                property_type = 'elasticity'
            else:
                property_type = self.property_type
            icl_examples = None
            if config['icl_examples'] is not None:
                if len(config['icl_examples']) == 0:
                    icl_examples = self.generate_icl_examples(data_dir, filename, num_examples=3)
                else:
                    icl_examples = config['icl_examples']
            gpt_result = self.analyzer.analyze_single_video_with_gpt(video_path=video_path, property_type=property_type, max_frames=16, frame_size=(224, 224), model_version='gpt-4o', frame_index=config['frame_index'], icl_examples=icl_examples, follow_oracle_test=config['follow_oracle_test'])
            predicted_value = self.analyzer.extract_predicted_value(gpt_result, property_type)
            if predicted_value is not None:
                print(f'✓ Retest successful:{predicted_value}')
                return gpt_result
            else:
                print(f'✗ The retest still failed and the predicted value could not be extracted')
                return None
        except Exception as e:
            print(f'✗ Error in retesting:{e}')
            return None

    def update_result_file(self, result_file: str, samples: List[Dict], updated_samples: Dict[int, str], filename: str='') -> None:
        if not updated_samples:
            print('There are no samples to update')
            return
        if 'v119' in filename:
            property_type = 'viscosity'
        elif 'v120' in filename:
            property_type = 'friction'
        elif 'v124' in filename:
            property_type = 'elasticity'
        else:
            property_type = self.property_type
        backup_file = result_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(result_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        print(f'Backup created:{backup_file}')
        with open(result_file, 'r', encoding='utf-8') as f:
content = f.read()
        for sample_id, new_gpt_response in updated_samples.items():
            new_predicted = self.analyzer.extract_predicted_value(new_gpt_response, property_type)
            if new_predicted is not None:
                pattern = f'(--- Sample{sample_id} ---\\n.*?video:.*?\\n.*?)Predicted value:None(\\n.*?GT value:.*?\\n.*?GPT full reply:\\n-{{40}}\\n).*?(\\n-{{40}})'
                replacement = f'\\g<1>Predicted value:{new_predicted}\\g<2>{new_gpt_response}\\g<3>'
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'✓ Updated{len(updated_samples)}sample results')

    def process_experiment_directory(self, exp_dir: str) -> None:
        print(f'\nProcessing experiment directory: {exp_dir}')
        exp_name = os.path.basename(exp_dir)
        if exp_name not in self.experiment_configs:
            print(f'Warning: Unknown experimental configuration:{exp_name}')
            return
        config = self.experiment_configs[exp_name]
        print(f"Configuration: frame_index={config['frame_index']}, icl={('yes' if config['icl_examples'] is not None else 'no')}, oracle={('yes' if config['follow_oracle_test'] else 'no')}")
        result_files = glob.glob(os.path.join(exp_dir, '*absolute_results.txt'))
        if not result_files:
            print(f'Warning: Results file not found')
            return
        total_retried = 0
        total_successful = 0
        for result_file in result_files:
            filename = os.path.basename(result_file)
            test_name = self.get_test_name_from_filename(filename)
            data_dir = self.get_data_directory_for_test(test_name, filename)
            if not data_dir or not os.path.exists(data_dir):
                print(f'Warning: data directory does not exist:{data_dir}')
                continue
            print(f'Process files:{filename} ({test_name})')
            samples = self.parse_result_file(result_file)
            failed_samples = [s for s in samples if s['is_failed']]
            if not failed_samples:
                print(f'No failed samples')
                continue
            print(f'turn up{len(failed_samples)}failed samples')
            updated_samples = {}
            for sample in failed_samples:
                print(f"Retest sample{sample['sample_id']}: {sample['video']}")
                new_gpt_response = self.retry_failed_sample(sample, data_dir, config, filename)
                if new_gpt_response:
                    updated_samples[sample['sample_id']] = new_gpt_response
                    total_successful += 1
                total_retried += 1
                time.sleep(15)
            if updated_samples:
                self.update_result_file(result_file, samples, updated_samples, filename)
            print(f'Finish:{len(updated_samples)}/{len(failed_samples)}Samples were retested successfully')
        print(f'Experiment directory processing completed: Total retests{total_retried}sample, successful{total_successful}indivual')

    def dry_run_analysis(self, target_experiments: Optional[List[str]]=None) -> None:
        print('🔍 Trial run mode: count the number of failed samples')
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
            print(f' - {exp}')
        print()
        total_experiments = 0
        total_files = 0total_samples = 0
        total_failed = 0
        experiment_stats = []
        for exp_dir in experiments_to_process:
            exp_name = os.path.basename(exp_dir)
            if exp_name not in self.experiment_configs:
                continue
            print(f'Analytical experiments:{exp_name}')
            result_files = glob.glob(os.path.join(exp_dir, '*absolute_results.txt'))
            if not result_files:
                print(f'Warning: Results file not found')
                continue
            exp_total_samples = 0
            exp_total_failed = 0
            file_stats = []
            for result_file in result_files:
                filename = os.path.basename(result_file)
                test_name = self.get_test_name_from_filename(filename)
                samples = self.parse_result_file(result_file)
                failed_samples = [s for s in samples if s['is_failed']]
                file_stats.append({'filename': filename, 'test_name': test_name, 'total_samples': len(samples), 'failed_samples': len(failed_samples), 'success_rate': (len(samples) - len(failed_samples)) / len(samples) * 100 if samples else 0})
                exp_total_samples += len(samples)
                exp_total_failed += len(failed_samples)
                print(f' {test_name}: {len(failed_samples)}/{len(samples)}fail ({(len(samples) - len(failed_samples)) / len(samples) * 100:.1f}% success rate)')
            experiment_stats.append({'experiment': exp_name, 'total_samples': exp_total_samples, 'failed_samples': exp_total_failed, 'success_rate': (exp_total_samples - exp_total_failed) / exp_total_samples * 100 if exp_total_samples else 0, 'file_stats': file_stats})
            total_experiments += 1
            total_files += len(result_files)
            total_samples += exp_total_samples
            total_failed += exp_total_failed
            print(f'Experiment summary:{exp_total_failed}/{exp_total_samples}fail ({(exp_total_samples - exp_total_failed) / exp_total_samples * 100:.1f}% success rate)')
            print()
        print('=' * 80)
        print('Trial run statistics summary')
        print('=' * 80)
        print(f'Total number of experiments:{total_experiments}')
        print(f'Total number of files:{total_files}')
        print(f'Total number of samples:{total_samples}')
        print(f'Number of failed samples:{total_failed}')
        print(f'Overall success rate:{(total_samples - total_failed) / total_samples * 100:.1f}%')
        print()
        experiment_stats.sort(key=lambda x: x['success_rate'])
        print('Experiment success rate ranking (from low to high):')
        print('-' * 80)
        for i, exp_stat in enumerate(experiment_stats, 1):
            print(f"{i:2d}. {exp_stat['experiment']:<50} {exp_stat['failed_samples']:3d}/{exp_stat['total_samples']:3d}fail ({exp_stat['success_rate']:5.1f}% success)")
        print()
        print('Number of samples that need to be retested:')
        print('-' * 40)
        for exp_stat in experiment_stats:
            if exp_stat['failed_samples'] > 0:
                print(f"{exp_stat['experiment']:<50} {exp_stat['failed_samples']:3d}samples")
        print(f'\ntotal needs to be retested: {total_failed}samples')
        print(f'Estimated retest time:{total_failed * 15 / 60:.1f}minutes (calculated as 15 seconds per sample)')

    def run(self, target_experiments: Optional[List[str]]=None) -> None:
        print('Start retesting failed predictions...')
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
            returnprint(f'will handle{len(experiments_to_process)}Experiment directory:')
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
    parser = argparse.ArgumentParser(description='Retest failed predictions in elasticity absolute ablation experiments')
    parser.add_argument('--experiments', nargs='*', help='Specify the name of the experiment directory to be processed. If not specified, all experiments will be processed.')
    parser.add_argument('--dry-run', action='store_true', help='Trial run mode only counts the number of failed samples and does not actually retest')
    args = parser.parse_args()
    retrier = RetryFailedPredictions()
    if args.dry_run:
        retrier.dry_run_analysis(target_experiments=args.experiments)
        return
    retrier.run(target_experiments=args.experiments)
if __name__ == '__main__':
    main()