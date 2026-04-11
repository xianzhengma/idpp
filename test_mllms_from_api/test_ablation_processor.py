import os
import glob
import pandas as pd
import numpy as np

def get_experiment_name_from_dir(dir_path):
    dir_name = os.path.basename(dir_path)
    experiment_mapping = {'gemini_physics_analysis_results': 'Baseline', 'gemini_physics_analysis_results_icl-example': 'ICL', 'gemini_physics_analysis_results_follow-oracle-test': 'Oracle', 'gemini_physics_analysis_results_icl-example_follow-oracle-test': 'ICL+Oracle', 'gemini_physics_analysis_results_concat-mode': 'Concat', 'gemini_physics_analysis_results_frame-index': 'FrameIndex', 'gemini_physics_analysis_results_icl-example_concat-mode': 'ICL+Concat', 'gemini_physics_analysis_results_follow-oracle-test_concat-mode': 'Oracle+Concat', 'gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index': 'ICL+Oracle+FrameIndex', 'gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index': 'AllFeatures'}
    return experiment_mapping.get(dir_name, dir_name)

def main():
    print('Test the function of batch processing experimental results...')
    pattern = 'gemini_physics_analysis_results*'
    base_dirs = glob.glob(pattern)
    print(f'Search mode:{pattern}')
    print(f'Current working directory:{os.getcwd()}')
    if not base_dirs:
        print(f'No matching directory found:{pattern}')
        all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        print(f'All directories under the current directory:{all_dirs}')
        related_dirs = [d for d in all_dirs if 'gemini' in d.lower() or 'physics' in d.lower() or 'analysis' in d.lower()]
        if related_dirs:
            print(f'Find relevant directories:{related_dirs}')
        return
    base_dirs.sort()
    print(f'\nDiscover{len(base_dirs)}Experiment directory:')
    for dir_path in base_dirs:
        experiment_name = get_experiment_name_from_dir(dir_path)
        print(f' - {dir_path} -> {experiment_name}')
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            result_files = [f for f in files if f.endswith('_results.txt')]
            print(f'Include{len(result_files)}result files')
        else:
            print(f'Warning: not a directory')
    print(f'\nGenerate simulated experiment comparison table...')
    experiments = [get_experiment_name_from_dir(d) for d in base_dirs]
    properties = ['viscosity', 'elasticity']
    test_sets = ['test1', 'test2', 'test3']
    np.random.seed(42)
    print(f'\nSimulation relative comparison result (AUC):')
    print('-' * 60)
    for prop in properties:
        print(f'\n{prop.upper()}- Relative comparison (AUC):')
        data = {}
        for exp in experiments:
            data[exp] = [np.random.uniform(0.5, 0.9) for _ in test_sets]
        df = pd.DataFrame(data, index=['Test-1', 'Test-2', 'Test-3'])
        df = df.T
        df['Average'] = df.mean(axis=1)
        df = df.round(4)
        df = df.sort_values('Average', ascending=False)
        print(df.to_string())
    print(f'\nScript function verification completed!')
    print(f'For actual use, please run: python process_individual_model_results_ablation.py')
if __name__ == '__main__':
    main()