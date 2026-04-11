import json
import os
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def test_gt_data_reading():
    analyzer = GPTPhysicsAnalyzer()
    test_cases = [{'version': 'v110_vjepa', 'data_dir': '/path/to/shared/genesis_data/6.2_viscosity_test3', 'property_type': 'viscosity', 'dataset_class': 'WebVidv3ViscosityTest', 'sample_video': 'video_0_id_0_friction_sim_box_objFriction_2.0_cola_1_1_rgb.mp4'}, {'version': 'v120_vjepa', 'data_dir': '/path/to/shared/genesis_data/5.2_viscosity_synthetic_test1', 'property_type': 'friction', 'dataset_class': 'WebVidv2Viscosity', 'sample_video': 'bounce_ball_E_0.005039669168531562_color_0.81-1.00-0_camangle_0.38_camheight_-0.41_camlookat_0.36_-0.23_-0.11_height_0.50_rgb.mp4'}]
    for test_case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Test version:{test_case['version']}")
        print(f"Data directory:{test_case['data_dir']}")
        print(f"Physical properties:{test_case['property_type']}")
        print(f"Dataset class:{test_case['dataset_class']}")
        print(f"Sample video:{test_case['sample_video']}")
        print(f"{'=' * 60}")
        if not os.path.exists(test_case['data_dir']):
            print(f"✗ The data directory does not exist:{test_case['data_dir']}")
            continue
        gt_data = analyzer.get_gt_data(test_case['data_dir'], test_case['dataset_class'], test_case['property_type'], test_case['version'])
        if gt_data is None:
            print('✗ Unable to load GT data')
            continue
        print(f'✓ Successfully loaded GT data, including{len(gt_data)}items')
        print(f'GT data key example:{list(gt_data.keys())[:5]}')
        gt_value = analyzer.get_gt_value(test_case['sample_video'], gt_data, test_case['dataset_class'], test_case['property_type'])
        print(f'Sample video GT value:{gt_value}')
        basename = test_case['sample_video'].replace('_rgb.mp4', '')
        print(f'File name basename:{basename}')
        if basename in gt_data:
            print(f'✓ Find the corresponding entry in GT data:{gt_data[basename]}')
        else:
            print(f'✗ No corresponding entry found in GT data')
            parsed_value = analyzer.extract_property_from_filename(test_case['sample_video'], test_case['property_type'], test_case['dataset_class'])
            print(f'File name parsing results:{parsed_value}')
            similar_keys = [key for key in gt_data.keys() if basename.split('_')[0] in key]
            print(f'Similar keys:{similar_keys[:5]}')
        gt_values = []
        for key, value in gt_data.items():
            if isinstance(value, dict):
                if 'gt_restitution' in value:
                    gt_values.append(value['gt_restitution'])
            else:
                gt_values.append(value)
        zero_count = sum((1 for v in gt_values if v == 0.0))
        non_zero_count = len(gt_values) - zero_count
        print(f'GT value statistics: Total ={len(gt_values)}, nonzero={non_zero_count}, zero value ={zero_count}')
        if gt_values:
            print(f'GT value range:{min(gt_values)} - {max(gt_values)}')
        print(f'\nCheck all video files in the directory...')
        video_files = [f for f in os.listdir(test_case['data_dir']) if f.endswith('_rgb.mp4')]
        print(f'Total number of video files:{len(video_files)}')
        matched_count = 0
        for video_file in video_files[:10]:
            basename = video_file.replace('_rgb.mp4', '')
            if basename in gt_data:
                matched_count += 1
        print(f'The first 10 video files with GT data:{matched_count}/10')
if __name__ == '__main__':
    test_gt_data_reading()