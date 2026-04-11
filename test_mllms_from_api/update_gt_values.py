import os
import re
import json
from typing import Dict, Any, Tuple, Optional
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def extract_gt_value_from_filename(filename: str, gt_data: Dict[str, Any]) -> Optional[float]:
    basename = filename.replace('.mp4', '')
    if basename in gt_data:
        return gt_data[basename].get('gt_restitution', 0.0)
    friction_match = re.search('objFriction_([0-9.]+)', basename)
    if friction_match:
        return float(friction_match.group(1))
    return None

def update_results_file(file_path: str, config: Dict[str, Any]) -> None:
    print(f'Updating file:{file_path}')
    if not os.path.exists(file_path):
        print(f'File does not exist:{file_path}')
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    version_match = re.search('Version:(\\w+)', content)
    data_dir_match = re.search('Data directory:([^\\n]+)', content)
    test_id_match = re.search('Test set:(\\w+)', content)
    if not version_match or not data_dir_match or (not test_id_match):
        print('Unable to extract version, data directory or test set information from file')
        return
    version = version_match.group(1)
    data_dir = data_dir_match.group(1)
    test_id = test_id_match.group(1)
    if version not in config:
        print(f'Version not found in configuration:{version}')
        return
    version_config = config[version]
    analyzer = GPTPhysicsAnalyzer()
    gt_data = analyzer.get_gt_data(data_dir=data_dir, dataset_class=version_config['dataset_class'], property_type=version_config['property'], version=version)
    if not gt_data:
        print(f'Unable to load GT data for version:{version}')
        return
    print(f'GT data has been loaded, including{len(gt_data)}items')
    lines = content.split('\n')
    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('Video pair:'):
            video_pair_match = re.search('Video pair: (.+?) vs (.+?)$', line)
            if video_pair_match:
                video1, video2 = video_pair_match.groups()
                gt_value1 = extract_gt_value_from_filename(video1, gt_data)
                gt_value2 = extract_gt_value_from_filename(video2, gt_data)
                if gt_value1 is not None and gt_value2 is not None:
                    gt_result = 1 if gt_value1 > gt_value2 else 0
                    updated_lines.append(line)
                    i += 1
                    if i < len(lines):
                        updated_lines.append(lines[i])
                    i += 1
                    if i < len(lines):
                        updated_lines.append(f'GT results:{gt_result}')
                    i += 1
                    if i < len(lines):
                        updated_lines.append(f'GT value:{gt_value1} vs {gt_value2}')
                    print(f'renew:{video1} ({gt_value1}) vs {video2} ({gt_value2}) → GT results:{gt_result}')
                else:
                    updated_lines.append(line)
                    if gt_value1 is None:
                        print(f'Warning: Unable to obtain GT value for{video1}')
                    if gt_value2 is None:
                        print(f'Warning: Unable to obtain GT value for{video2}')
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
        i += 1
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))
    print(f'File update completed:{file_path}')

def main():
    config_file = 'vjepa_data_dirs.json'
    if not os.path.exists(config_file):
        print(f'Configuration file does not exist:{config_file}')
        return
    with open(config_file, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    dataset_class_mapping = {'v110_vjepa': 'WebVidv3ViscosityTest', 'v120_vjepa': 'WebVidv2Viscosity', 'v119_vjepa': 'WebVidv3ViscosityTest', 'v121_vjepa': 'WebVidv2Viscosity', 'v124_vjepa': 'WebVidv2Viscosity', 'v125_vjepa': 'WebVidv2Viscosity'}
    config = {}
    for version, version_config in base_config.items():
        config[version] = {'property': version_config['property'], 'type': version_config['type'], 'data_dirs': version_config['data_dirs'], 'dataset_class': dataset_class_mapping.get(version, 'WebVidv3ViscosityTest')}
    print(f'Loaded configuration:{len(config)}test version')
    result_files = []
    gpt_results_dir = 'gpt_physics_analysis_results'
    if os.path.exists(gpt_results_dir):
        for file in os.listdir(gpt_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(gpt_results_dir, file))
    gemini_results_dir = 'gemini_physics_analysis_results'
    if os.path.exists(gemini_results_dir):
        for file in os.listdir(gemini_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(gemini_results_dir, file))
    qwen_results_dir = 'qwen_physics_analysis_results'
    if os.path.exists(qwen_results_dir):
        for file in os.listdir(qwen_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(qwen_results_dir, file))
    print(f'turn up{len(result_files)}Results files need to be updated:')
    for file in result_files:
        print(f'  - {file}')
    for file_path in result_files:
        try:
            update_results_file(file_path, config)
        except Exception as e:
            print(f'update file{file_path}An error occurred:{e}')
    print('All files updated!')
if __name__ == '__main__':
    main()
