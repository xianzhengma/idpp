import os
importsys
import json
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def test_single_video():
    print('=' * 80)
    print('Single video test - for debugging GPT reply and extraction logic')
    print('=' * 80)
    try:
        analyzer = GPTPhysicsAnalyzer()
        print('✓ Analyzer initialization successful')
    except Exception as e:
        print(f'✗ Parser initialization failed:{e}')
        return
    config = analyzer.config
    print(f'\nAvailable test configurations:')
    for i, (version, version_config) in enumerate(config.items()):
        print(f"{i + 1}. {version}: {version_config['property']} ({version_config['type']})")
    version_name = list(config.keys())[0]
    version_config = config[version_name]
    print(f'\nUse configuration: {version_name}')
    print(f"property:{version_config['property']}")
    print(f"type:{version_config['type']}")
    test_id = list(version_config['data_dirs'].keys())[0]
    data_dir = version_config['data_dirs'][test_id]
    print(f'Test set:{test_id}')
    print(f'Data directory:{data_dir}')
    if not os.path.exists(data_dir):
        print(f'✗ The data directory does not exist:{data_dir}')
        return
    videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
    if not videos:
        print(f'✗ The rgb.mp4 file was not found in the directory')
        return
    video_file = videos[0]
    video_path = os.path.join(data_dir, video_file)
    print(f'\nTest video:{video_file}')
    print(f'Full path:{video_path}')
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    print(f'Dataset class:{dataset_class}')
    gt_data = analyzer.get_gt_data(data_dir, dataset_class, version_config['property'])
    gt_value = analyzer.get_gt_value(video_file, gt_data, dataset_class, version_config['property'])
    print(f'GT value:{gt_value}')
    print('\n' + '=' * 80)
    print('Start GPT analysis...')
    print('=' * 80)
    try:
        gpt_result = analyzer.analyze_single_video_with_gpt(video_path, version_config['property'], max_frames=8, frame_size=(320, 240), model_version='gemini-2.5-pro')
        print(f'\n' + '=' * 80)
        print('Extract predicted values...')
        print('=' * 80)
        predicted_value = analyzer.extract_predicted_value(gpt_result, version_config['property'])
        print(f'\n' + '=' * 80)
        print('Summary of test results')
        print('=' * 80)
        print(f'video:{video_file}')
        print(f"property:{version_config['property']}")
        print(f'GT value:{gt_value}')
        print(f'Predicted value:{predicted_value}')
        if predicted_value is not None:
            print('✓ Successfully extracted predicted values')
        else:
            print('✗ Failed to extract predicted values')
    except Exception as e:
        print(f'✗ Analysis failed:{e}')
        import traceback
        traceback.print_exc()

def main():
    test_single_video()
if __name__ == '__main__':
    main()