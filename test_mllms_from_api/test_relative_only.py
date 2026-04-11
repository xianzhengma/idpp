import os
import sys
import argparse
from datetime import datetime
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def get_default_output_dir(model_version: str) -> str:
    model_name = model_version.lower().replace('-', '_').replace('.', '_')
    if model_name.startswith('gpt'):
        if 'mini' in model_name:
            return 'gpt_mini_physics_analysis_results'
        elif 'turbo' in model_name:
            return 'gpt_turbo_physics_analysis_results'
        else:
            return 'gpt_physics_analysis_results'
    elif model_name.startswith('gemini'):
        return 'gemini_physics_analysis_results'
    elif model_name.startswith('claude'):
        return 'claude_physics_analysis_results'
    else:
        model_prefix = model_name.split('_')[0]
        return f'{model_prefix}_physics_analysis_results'

def test_relative_version(analyzer, version_name, version_config, max_samples=3, frame_size=(320, 240), model_version='gpt-4o', output_dir=None):
    print(f"\n{'=' * 80}")
    print(f'Start testing version:{version_name}')
    print(f"property:{version_config['property']}")
    print(f"type:{version_config['type']}")
    print(f'Maximum number of pairs per test set:{max_samples}')
    print(f'GPT model version:{model_version}')
    print(f'Output directory:{output_dir}')
    print(f"{'=' * 80}")
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    all_results = []
    for test_id, data_dir in version_config['data_dirs'].items():
        print(f'\n--- Test set:{test_id} ---')
        print(f'Data directory:{data_dir}')
        try:
            if analyzer.is_test_completed(output_dir, version_name, test_id, 'relative'):
                print(f'✓ Test{version_name}_{test_id}_relative Completed, skipped')
                continue
            result = analyzer.run_relative_test(version_name, version_config['property'], test_id, data_dir, dataset_class, frame_size, max_samples, model_version)
            all_results.append(result)
            analyzer.save_test_result(result, output_dir)
            success_rate = result['successful_analyses'] / result['total_samples'] * 100 if result['total_samples'] > 0 else 0
            print(f"✓ Test completed:{result['successful_analyses']}/{result['total_samples']}success ({success_rate:.1f}%)")
        except Exception as e:
            print(f'✗ Test set{test_id}fail:{str(e)}')
            continue
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Script dedicated to testing relative comparisons')
    parser.add_argument('--version', type=str, help='Specify the version to be tested (such as v125_vjepa). If not specified, all relative versions will be tested.')
    parser.add_argument('--samples', type=int, default=3, help='Maximum number of pairs per test set (default: 3)')
    parser.add_argument('--model-version', type=str, default='gpt-4o', help='GPT model version (default: gpt-4o)')
    parser.add_argument('--output-dir', type=str, default=None, help='Results output directory (default: automatically generated based on model version)')
    parser.add_argument('--frame-width', type=int, default=320, help='Video frame width (default: 320)')
    parser.add_argument('--frame-height', type=int, default=240, help='Video frame height (default: 240)')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.model_version)
    print('Initializing GPT physical property analyzer...')
    try:
        analyzer = GPTPhysicsAnalyzer()
        print(f'Loaded configuration:{len(analyzer.config)}test version')
        print('✓ Analyzer initialization successful')
    except Exception as e:
        print(f'✗ Parser initialization failed:{e}')
        return 1
    frame_size = (args.frame_width, args.frame_height)
    relative_versions = {name: config for name, config in analyzer.config.items() if config['type'] == 'relative'}
    if not relative_versions:
        print('✗ No relative test version found')
        return 1
    print(f'\nturn up{len(relative_versions)}Relative test version:')
    for name, config in relative_versions.items():
print(f" - {name}: {config['property']}")
    if args.version:
        if args.version not in relative_versions:
            print(f'✗ Version{args.version}Does not exist or is not a relative beta version')
            print(f'Available relative test versions:{list(relative_versions.keys())}')
            return 1
        versions_to_test = {args.version: relative_versions[args.version]}
        print(f'\nOnly test the specified Version:{args.version}')
    else:
        versions_to_test = relative_versions
        print(f'\nTest all relative Versions')
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    for version_name, version_config in versions_to_test.items():
        results = test_relative_version(analyzer, version_name, version_config, args.samples, frame_size, args.model_version, args.output_dir)
        all_results.extend(results)
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(args.output_dir, f'relative_test_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('=== Relative test summary results ===\n')
            f.write(f"Test time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f'GPT model version:{args.model_version}\n')
            f.write(f"Total number of test versions:{len(set((r['version'] for r in all_results)))}\n")
            f.write(f'Total number of test sets:{len(all_results)}\n')
            f.write(f"Total number of pairs:{sum((r['total_samples'] for r in all_results))}\n")
            f.write(f"Number of successful analyses:{sum((r['successful_analyses'] for r in all_results))}\n")
            f.write(f"Number of failed analyses:{sum((r['failed_analyses'] for r in all_results))}\n")
            f.write('\n')
            for result in all_results:
                f.write(f"Version:{result['version']}, test set:{result['test_set']}, ")
                f.write(f"success:{result['successful_analyses']}/{result['total_samples']}\n")
        print(f'\n✓ Summary results have been saved to:{summary_file}')
    print(f"\n{'=' * 80}")
    print('Relative testing completed!')
    print(f'Results have been saved to:{args.output_dir}')
    print(f"{'=' * 80}")
    return 0
if __name__ == '__main__':
    sys.exit(main())