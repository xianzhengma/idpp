import os
import glob
import re
from typing import Dict, List

def check_experiment_progress() -> None:
    print('Check the progress of the Elasticity Relative Ablation experiment...')
    experiment_dirs = ['gemini_physics_analysis_results_elasticity_relative_20samples', 'gemini_physics_analysis_results_elasticity_relative_20samples_frameindex', 'gemini_physics_analysis_results_elasticity_relative_20samples_icl', 'gemini_physics_analysis_results_elasticity_relative_20samples_oracle', 'gemini_physics_analysis_results_elasticity_relative_20samples_concat']
    total_complete = 0
    total_comparisons = 0
    print('\n' + '=' * 100)
    print(f"{'Experiment name':<60} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'total':<10}")
    print('=' * 100)
    for exp_dir in experiment_dirs:
        if not os.path.exists(exp_dir):
            print(f"{exp_dir:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        result_files = glob.glob(os.path.join(exp_dir, '*relative_results.txt'))
        if not result_files:
            print(f"{exp_dir:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        test_results = {}
        for result_file in result_files:
            filename = os.path.basename(result_file)
            if 'test1' in filename:
                test_name = 'Test-1'
            elif 'test2' in filename:
                test_name = 'Test-2'
            elif 'test3' in filename:
                test_name = 'Test-3'
            else:
                continue
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            pattern = 'Predicted results:(.*?)\\n'
            predictions = re.findall(pattern, content)
            valid_count = sum((1 for pred in predictions if pred != 'None'))
            total_count = len(predictions)
            test_results[test_name] = f'{valid_count}/{total_count}'
            total_complete += valid_count
            total_comparisons += total_count
        test1_result = test_results.get('Test-1', 'N/A')
        test2_result = test_results.get('Test-2', 'N/A')
        test3_result = test_results.get('Test-3', 'N/A')
        total_valid = 0
        total_all = 0
        for test_result in test_results.values():
            if test_result != 'N/A':
                valid, all_count = map(int, test_result.split('/'))
                total_valid += valid
                total_all += all_count
        total_result = f'{total_valid}/{total_all}' if total_all > 0 else 'N/A'
        print(f'{exp_dir:<60} {test1_result:<10} {test2_result:<10} {test3_result:<10} {total_result:<10}')
    print('=' * 100)
    print(f"{'overall progress':<60} {'':<10} {'':<10} {'':<10} {total_complete}/{total_comparisons}")
    print(f'Completion rate:{total_complete / total_comparisons * 100:.1f}%' if total_comparisons > 0 else 'Completion rate: N/A')
    failed_comparisons = total_comparisons - total_complete
    if failed_comparisons > 0:
        print(f'Still need to retest:{failed_comparisons}comparison')
        print(f'Estimated time:{failed_comparisons * 15 / 60:.1f}minute')
    else:
        print('🎉 All comparisons are done!')
if __name__ == '__main__':
    check_experiment_progress()
