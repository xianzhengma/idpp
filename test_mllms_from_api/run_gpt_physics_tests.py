import os
import sys
import argparse
from datetime import datetime
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def get_default_output_dir(model_version: str, concat_mode: bool=False, frame_index: bool=False, icl_example: bool=False, follow_oracle_test: bool=False) -> str:
    model_name = model_version.lower().replace('-', '_').replace('.', '_')
    if model_name.startswith('gpt'):
        if 'mini' in model_name:
            base_dir = 'gpt_mini_physics_analysis_results'
        elif 'turbo' in model_name:
            base_dir = 'gpt_turbo_physics_analysis_results'
        else:
            base_dir = 'gpt_physics_analysis_results'
    elif model_name.startswith('qwen'):
        if 'max' in model_name:
            base_dir = 'qwen_max_physics_analysis_results'
        elif 'plus' in model_name:
            base_dir = 'qwen_plus_physics_analysis_results'
        elif 'turbo' in model_name:
            base_dir = 'qwen_turbo_physics_analysis_results'
        else:
            base_dir = 'qwen_physics_analysis_results'
    elif model_name.startswith('gemini'):
        base_dir = 'gemini_physics_analysis_results'
    elif model_name.startswith('claude'):
        base_dir = 'claude_physics_analysis_results'
    else:
        model_prefix = model_name.split('_')[0]
        base_dir = f'{model_prefix}_physics_analysis_results'
    suffixes = []
    if concat_mode:
        suffixes.append('concat')
    if frame_index:
        suffixes.append('frameindex')
    if icl_example:
        suffixes.append('icl')
    if follow_oracle_test:
        suffixes.append('oracle')
    if suffixes:
        suffix_str = '_'.join(suffixes)
        return f'{base_dir}_{suffix_str}'
    else:
        return base_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal model physical property analysis - complete test runner')
    parser.add_argument('--absolute-samples', type=int, default=50, help='Number of samples per test set for absolute value testing (default: 50)')
    parser.add_argument('--relative-pairs', type=int, default=50, help='Number of paired videos per test set for relative comparison testing (default: 50)')
    parser.add_argument('--model-version', type=str, default='gpt-4o', help='Model version (supported: gpt-4o, qwen-vl-max, qwen-vl-plus, qwen-vl-turbo, default: gpt-4o)')
    parser.add_argument('--frame-width', type=int, default=640, help='Video frame width (default: 640)')
    parser.add_argument('--frame-height', type=int, default=480, help='Video frame height (default: 480)')
    parser.add_argument('--output-dir', type=str, default=None, help='Results output directory (default: automatically generated based on model version)')
    parser.add_argument('--dry-run', action='store_true', help='Only shows the test plan, does not execute the actual test')
    parser.add_argument('--auto-confirm', action='store_true', help='Automatically confirm tests without user input')
    parser.add_argument('--concat-mode', action='store_true', help='Use stitching mode for relative comparison (add a black frame to separate the two videos)')
    parser.add_argument('--properties', type=str, nargs='*', default=None, choices=['friction', 'viscosity', 'elasticity'], help='Specify the physical property to test (optional: friction, viscosity, elasticity). All properties are tested by default. Example: --properties friction viscosity')
    parser.add_argument('--frame-index', action='store_true', help='Add an explicit frame index identifier to each video frame (eg: frame1: image, frame2: image)')
    parser.add_argument('--icl-example', action='store_true', help='Enable in-context learning and sample 3 examples from the training set as input-output pairs')
    parser.add_argument('--follow-oracle-test', action='store_true', help='Enable problem-solving step guidance and tell the model specific analysis methods and visual clues')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries per API call (default: 5)')
    parser.add_argument('--retry-delay', type=int, default=15, help='Retry interval in seconds (default: 15)')
    parser.add_argument('--validation-retries', type=int, default=3, help='Number of retries for result verification. If no valid value can be extracted, the API will be called again (default: 3)')
    parser.add_argument('--aggressive-retry', action='store_true', help='Enable aggressive retry mode: increase the number of retries and verifications to ensure more valid data on the first run')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(model_version=args.model_version, concat_mode=args.concat_mode, frame_index=args.frame_index, icl_example=args.icl_example, follow_oracle_test=args.follow_oracle_test)
    if args.properties is not None:
        if len(args.properties) == 0:
            print('Error: --properties argument requires specifying at least one physical property')
            print('Optional attributes: friction, viscosity, elasticity')
            print('Example: --properties friction viscosity')
            return 1
        valid_properties = ['friction', 'viscosity', 'elasticity']
        invalid_properties = [p for p in args.properties if p not in valid_properties]
        if invalid_properties:
            print(f"Error: Invalid property name:{', '.join(invalid_properties)}")
            print(f"Optional attributes:{', '.join(valid_properties)}")
            return 1
        print(f"Specify test properties:{', '.join(args.properties)}")
    else:
        print('Test all physical properties')
    print('Initializing multimodal model physical property analyzer...')
    try:
        analyzer = GPTPhysicsAnalyzer()
        print(f'Loaded configuration:{len(analyzer.config)}test version')
        print('✓ Analyzer initialization successful')
    except Exception as e:
        print(f'✗ Parser initialization failed:{e}')
        return 1
    if args.aggressive_retry:
        print('🚀 Aggressive retry mode enabled')
        if not any(('--max-retries' in arg for arg in sys.argv)):
            args.max_retries = 8
        if not any(('--validation-retries' in arg for arg in sys.argv)):
            args.validation_retries = 5
        if not any(('--retry-delay' in arg for arg in sys.argv)):
            args.retry_delay = 20
    print(f'\n================================================================================')
    print(f'Multimodal Model Physical Property Analysis - Test Plan')
    print(f'================================================================================')
    print(f'Model version:{args.model_version}')
    print(f'Number of absolute value test samples:{args.absolute_samples}')
    print(f'Relative comparison test for pairs:{args.relative_pairs}')
    print(f'Video frame size:{args.frame_width}x{args.frame_height}')
    print(f"Splicing mode:{('enable' if args.concat_mode else 'Disable')}")
    print(f"Frame index mode:{('enable' if args.frame_index else 'Disable')}")
    print(f"ICL example:{('enable' if args.icl_example else 'Disable')}")
    print(f"Problem solving step guide:{('enable' if args.follow_oracle_test else 'Disable')}")
    print(f'API retries:{args.max_retries}')
    print(f'Retry delay:{args.retry_delay}Second')
    print(f'Number of verification retries:{args.validation_retries}')
    print(f"Aggressive retry mode:{('enable' if args.aggressive_retry else 'Disable')}")
    if args.properties:
        print(f"Test properties:{', '.join(args.properties)}")
    else:
        print(f'Test properties: all properties (friction, viscosity, elasticity)')
    print(f'Output directory:{args.output_dir}')
    print()
    frame_size = (args.frame_width, args.frame_height)
    max_samples = max(args.absolute_samples, args.relative_pairs)
    print('Test configuration details:')
    print()
    total_absolute_tests = 0
    total_relative_tests = 0
    for version_name, version_config in analyzer.config.items():
        if args.properties is not None and version_config['property'] not in args.properties:
            continue
        if version_config['type'] == 'absolute' and args.absolute_samples == 0:
            continue
        if version_config['type'] == 'relative' and args.relative_pairs == 0:
            continue
        test_sets = [k for k in version_config['data_dirs'].keys() if k != 'train']
        print(f'{version_name}:')
        print(f"property:{version_config['property']}")
        print(f"type:{version_config['type']}")
        print(f'Test set:{test_sets}')
        if version_config['type'] == 'absolute':
            print(f'Number of samples in each test set:{args.absolute_samples}')
            print(f'Total number of tests:{len(test_sets)}')
            total_absolute_tests += len(test_sets)
        else:
            print(f'Number of ensemble pairs per test:{args.relative_pairs}')
            print(f'Total number of tests:{len(test_sets)}')
            total_relative_tests += len(test_sets)
        print()
    print('total:')
    print(f'Absolute value test:{total_absolute_tests}')
    print(f'Relative comparison test:{total_relative_tests}')
    print(f'Total number of tests:{total_absolute_tests + total_relative_tests}')
    estimated_minutes = (total_absolute_tests + total_relative_tests) * 0.5
    print(f'Estimated time:{int(estimated_minutes // 60)}Hour{int(estimated_minutes % 60)}minute')
    print('================================================================================')
    print()
    if args.dry_run:
        print('💡 Dry run mode: no actual testing is performed')
        return 0
    print('Ready to start testing...')
    if not args.auto_confirm:
        response = input('Continue? (y/n, default n):').lower().strip()
        if response not in ['y', 'yes', 'yes']:
            print('Test canceled')
            return 0
    os.makedirs(args.output_dir, exist_ok=True)
    print('Start running tests...')
    print()
    try:
        results = analyzer.run_all_tests(output_dir=args.output_dir, frame_size=frame_size, max_samples=max_samples, model_version=args.model_version, concat_mode=args.concat_mode, properties_filter=args.properties, frame_index=args.frame_index, icl_example=args.icl_example, follow_oracle_test=args.follow_oracle_test, absolute_samples=args.absolute_samples, relative_pairs=args.relative_pairs, max_retries=args.max_retries, retry_delay=args.retry_delay, validation_retries=args.validation_retries, aggressive_retry=args.aggressive_retry)
        print('\n' + '=' * 50)
        print('✅ All tests completed!')
        print(f'Results have been saved to:{args.output_dir}')
        print('=' * 50)
        return 0
    except Exception as e:
        print(f'✗ Run failed:{e}')
        return 1
if __name__ == '__main__':
    sys.exit(main())
