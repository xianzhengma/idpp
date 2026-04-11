import os
import sys
import json
import time
import argparse
from datetime import datetime
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def get_default_output_dir(model_version: str) -> str:
    model_name = model_version.lower().replace('-', '_').replace('.', '_')
    if model_name.startswith('gpt'):
        if 'mini' in model_name:
            return 'gpt_mini_quick_test_results'
        elif 'turbo' in model_name:
            return 'gpt_turbo_quick_test_results'
        else:
            return 'gpt_quick_test_results'
    elif model_name.startswith('gemini'):
        return 'gemini_quick_test_results'
    elif model_name.startswith('claude'):
        return 'claude_quick_test_results'
    else:
        model_prefix = model_name.split('_')[0]
        return f'{model_prefix}_quick_test_results'

def save_individual_result(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{result['version']}_{result['test_set']}_{result['type']}_result.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Quick Test Result - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('=' * 60 + '\n\n')
        f.write(f"Version: {result['version']}\n")
        f.write(f"Property: {result['property']}\n")
        f.write(f"Type: {result['type']}\n")
        f.write(f"Test Set: {result['test_set']}\n")
        f.write(f"Data Directory: {result['data_dir']}\n")
        f.write(f"Total Samples: {result['total_samples']}\n")
        f.write(f"Successful Analyses: {result['successful_analyses']}\n")
        f.write(f"Failed Analyses: {result['failed_analyses']}\n")
        if 'model_version' in result:
            f.write(f"Model Version: {result['model_version']}\n")
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
        if result['type'] == 'absolute' and 'individual_results' in result:
            f.write('\nIndividual Results:\n')
            for item in result['individual_results']:
                if 'error' in item:
                    f.write(f"  - {item['video']}: ERROR - {item['error']}\n")
                else:
                    f.write(f"  - {item['video']}: predicted={item['predicted']}, gt={item['gt']}\n")
                    if 'gpt_response' in item:
                        f.write(f"    GPT Response: {item['gpt_response'][:200]}...\n")
        elif result['type'] == 'relative' and 'comparison_results' in result:
            f.write('\nComparison Results:\n')
            for item in result['comparison_results']:
                if 'error' in item:
                    f.write(f"  - {item['video_pair']}: ERROR - {item['error']}\n")
                else:
                    f.write(f"  - {item['video_pair']}: predicted={item['predicted']}, gt={item['gt']}\n")
                    if 'gt_values' in item:
                        f.write(f"    GT Values: {item['gt_values']}\n")
        f.write('\n' + '=' * 60 + '\n')
    print(f'✓ Results saved to:{filepath}')

def test_single_config(analyzer, version_name, version_config, output_dir, frame_size=(320, 240), max_samples=1, model_version='gpt-4o'):
    results = []
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    if not dataset_class:
        print(f'jump over{version_name}: The corresponding data set class was not found')
        return results
    for test_id, data_dir in version_config['data_dirs'].items():
        print(f'\n--- Test set:{test_id} ---')
        print(f'Data directory:{data_dir}')
        print(f'Maximum number of samples:{max_samples}')
        print(f'Model version:{model_version}')
        try:
            gt_data = analyzer.get_gt_data(data_dir, dataset_class, version_config['property'], version=version_name)
            if version_config['type'] == 'absolute':
                sampled_videos = analyzer.sample_videos(data_dir, max_samples=max_samples)
                if not sampled_videos:
                    raise ValueError(f'No available video file found')
                test_video = sampled_videos[0]
                result = test_absolute_single(analyzer, version_name, version_config['property'], test_id, data_dir, dataset_class, test_video, frame_size, gt_data, model_version)
            else:
                required_videos = max_samples * 2
                sampled_videos = analyzer.sample_videos(data_dir, max_samples=required_videos)
                if len(sampled_videos) < 2:
                    raise ValueError(f'Relative comparison testing requires at least 2 video files')
                test_video = sampled_videos[0]
                test_video2 = sampled_videos[1]
                result = test_relative_single(analyzer, version_name, version_config['property'], test_id, data_dir, dataset_class, test_video, test_video2, frame_size, gt_data, model_version)
            result['model_version'] = model_version
            results.append(result)
            save_individual_result(result, output_dir)
            if result['successful_analyses'] > 0:
                print(f"✓ Test successful:{result['successful_analyses']}/{result['total_samples']}")
            else:
                print(f"✗ Test failed:{result['failed_analyses']}/{result['total_samples']}")
        except Exception as e:
            print(f'✗ Test failed:{str(e)}')
            failed_result = {'version': version_name, 'property': version_config['property'], 'type': version_config['type'], 'test_set': test_id, 'data_dir': data_dir, 'total_samples': max_samples if version_config['type'] == 'absolute' else 1, 'successful_analyses': 0, 'failed_analyses': max_samples if version_config['type'] == 'absolute' else 1, 'error': str(e), 'model_version': model_version}
            if version_config['type'] == 'absolute':
                failed_result['individual_results'] = []
            else:
                failed_result['comparison_results'] = []
            results.append(failed_result)
            save_individual_result(failed_result, output_dir)
        time.sleep(5)
    return results

def test_absolute_single(analyzer, version, property_type, test_id, data_dir, dataset_class, video_file, frame_size, gt_data, model_version):
    print(f'Start absolute value testing:{property_type}')
    try:
        gt_value = analyzer.get_gt_value(video_file, gt_data, dataset_class, property_type)
        print(f'GT value:{gt_value}')
        video_path = os.path.join(data_dir, video_file)
        print(f'Analysis video:{video_file}')
        gpt_result = analyzer.analyze_single_video_with_gpt(video_path, property_type, max_frames=16, frame_size=frame_size, model_version=model_version)
        predicted_value = analyzer.extract_predicted_value(gpt_result, property_type)
        individual_results = [{'video': video_file, 'predicted': predicted_value, 'gt': gt_value, 'gpt_response': gpt_result}]
        return {'version': version, 'property': property_type, 'type': 'absolute', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': 1, 'successful_analyses': 1 if predicted_value is not None else 0, 'failed_analyses': 0 if predicted_value is not None else 1, 'individual_results': individual_results}
    except Exception as e:
        print(f'Analysis failed:{str(e)}')
        return {'version': version, 'property': property_type, 'type': 'absolute', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': 1, 'successful_analyses': 0, 'failed_analyses': 1, 'individual_results': [{'video': video_file, 'predicted': None, 'gt': None, 'error': str(e)}]}

def test_relative_single(analyzer, version, property_type, test_id, data_dir, dataset_class, video_file1, video_file2, frame_size, gt_data, model_version):
    print(f'Start the relative comparison test:{property_type}')
    try:
        gt_value1 = analyzer.get_gt_value(video_file1, gt_data, dataset_class, property_type)
        gt_value2 = analyzer.get_gt_value(video_file2, gt_data, dataset_class, property_type)
        gt_comparison = 1 if gt_value1 > gt_value2 else 0
        print(f'Compare video pairs:{video_file1} vs {video_file2}')
        print(f'GT value:{gt_value1} vs {gt_value2}, GT comparison:{gt_comparison}')
        video_path1 = os.path.join(data_dir, video_file1)
video_path2 = os.path.join(data_dir, video_file2)
        gpt_result = analyzer.compare_two_videos_with_gpt(video_path1, video_path2, property_type, max_frames_per_video=16, frame_size=frame_size, model_version=model_version)
        predicted_comparison = analyzer.extract_comparison_result(gpt_result)
        comparison_results = [{'video_pair': f'{video_file1} vs {video_file2}', 'predicted': predicted_comparison, 'gt': gt_comparison, 'gt_values': f'{gt_value1} vs {gt_value2}', 'gpt_response': gpt_result}]
        return {'version': version, 'property': property_type, 'type': 'relative', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': 2, 'successful_analyses': 1 if predicted_comparison is not None else 0, 'failed_analyses': 0 if predicted_comparison is not None else 1, 'comparison_results': comparison_results}
    except Exception as e:
        print(f'Comparison failed:{str(e)}')
        return {'version': version, 'property': property_type, 'type': 'relative', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': 2, 'successful_analyses': 0, 'failed_analyses': 1, 'comparison_results': [{'video_pair': f'{video_file1} vs {video_file2}', 'predicted': None, 'gt': None, 'error': str(e)}]}

def save_summary_results(all_results, output_dir, model_version):
    summary_file = os.path.join(output_dir, f"quick_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Quick Test Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('=' * 80 + '\n\n')
        f.write(f'Model Version: {model_version}\n')
        total_configs = len(all_results)
        successful_configs = sum((1 for r in all_results if r['successful_analyses'] > 0))
        failed_configs = total_configs - successful_configs
        f.write(f'Total number of configurations:{total_configs}\n')
        f.write(f'Number of successful configurations:{successful_configs}\n')
        f.write(f'Number of failed configurations:{failed_configs}\n')
        f.write(f'Success rate:{successful_configs / total_configs * 100:.1f}%\n\n')
        properties = ['friction', 'viscosity', 'elasticity']
        for prop in properties:
            prop_results = [r for r in all_results if r['property'] == prop]
            if prop_results:
                f.write(f'{prop.upper()}Test results:\n')
                for result in prop_results:
                    status = '✓' if result['successful_analyses'] > 0 else '✗'
                    f.write(f" {status} {result['version']} ({result['type']}) - {result['test_set']}: {result['successful_analyses']}/{result['total_samples']}\n")
                f.write('\n')
        f.write('Detailed results:\n')
        f.write('-' * 60 + '\n')
        for result in all_results:
            f.write(f"\nConfiguration: {result['version']} - {result['test_set']}\n")
            f.write(f"property:{result['property']} ({result['type']})\n")
            f.write(f"state:{('success' if result['successful_analyses'] > 0 else 'fail')}\n")
            if 'error' in result:
                f.write(f"mistake:{result['error']}\n")
    print(f'✓ Summary results have been saved to:{summary_file}')

def main():
    parser = argparse.ArgumentParser(description='Quickly test all configuration scripts')
    parser.add_argument('--model-version', type=str, default='gpt-4o', help='GPT model version (default: gpt-4o)')
    parser.add_argument('--frame-width', type=int, default=320, help='Video frame width (default: 320)')
    parser.add_argument('--frame-height', type=int, default=240, help='Video frame height (default: 240)')
    parser.add_argument('--output-dir', type=str, default=None, help='Results output directory (default: automatically generated based on model version)')
    parser.add_argument('--max-samples', type=int, default=1, help='Maximum number of samples per test set (default: 1)')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.model_version)
    print('Start a quick test of all configurations...')
    print(f'Model version:{args.model_version}')    print(f'Output directory:{args.output_dir}')
    print(f'Maximum number of samples per test set:{args.max_samples}')
    frame_size = (args.frame_width, args.frame_height)
    analyzer = GPTPhysicsAnalyzer()
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    for version_name, version_config in analyzer.config.items():
        print(f"\n{'=' * 80}")
        print(f'Test configuration:{version_name}')
        print(f"property:{version_config['property']}, type:{version_config['type']}")
        print(f'Model version:{args.model_version}')
        print(f"{'=' * 80}")
        results = test_single_config(analyzer, version_name, version_config, args.output_dir, frame_size, args.max_samples, args.model_version)
        all_results.extend(results)
    save_summary_results(all_results, args.output_dir, args.model_version)
    total_configs = len(all_results)
    successful_configs = sum((1 for r in all_results if r['successful_analyses'] > 0))
    print(f"\n{'=' * 80}")
    print('Quick test completed!')
    print(f'Model version:{args.model_version}')
    print(f'Total number of configurations:{total_configs}')
    print(f'Number of successful configurations:{successful_configs}')
    print(f'Success rate:{successful_configs / total_configs * 100:.1f}%')
    print(f'Results have been saved to:{args.output_dir}')
    print(f"{'=' * 80}")
    return 0
if __name__ == '__main__':
    import sys
    exit_code = main()
    sys.exit(exit_code)
