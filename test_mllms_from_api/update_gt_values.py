#!/usr/bin/env python3
"""
更新测试结果文件中的GT值
根据相对比较规则：第一个GT值 > 第二个GT值 → GT结果为1，否则为0
"""

import os
import re
import json
from typing import Dict, Any, Tuple, Optional
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def extract_gt_value_from_filename(filename: str, gt_data: Dict[str, Any]) -> Optional[float]:
    """从视频文件名中提取GT值"""
    # 移除.mp4扩展名
    basename = filename.replace('.mp4', '')
    
    # 直接查找GT数据
    if basename in gt_data:
        return gt_data[basename].get('gt_restitution', 0.0)
    
    # 如果直接查找失败，尝试提取objFriction值
    friction_match = re.search(r'objFriction_([0-9.]+)', basename)
    if friction_match:
        return float(friction_match.group(1))
    
    return None

def update_results_file(file_path: str, config: Dict[str, Any]) -> None:
    """更新单个结果文件中的GT值"""
    print(f"正在更新文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 读取现有文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析测试版本和数据目录
    version_match = re.search(r'版本: (\w+)', content)
    data_dir_match = re.search(r'数据目录: ([^\n]+)', content)
    test_id_match = re.search(r'测试集: (\w+)', content)
    
    if not version_match or not data_dir_match or not test_id_match:
        print("无法从文件中提取版本、数据目录或测试集信息")
        return
    
    version = version_match.group(1)
    data_dir = data_dir_match.group(1)
    test_id = test_id_match.group(1)
    
    # 获取对应的配置
    if version not in config:
        print(f"配置中找不到版本: {version}")
        return
    
    version_config = config[version]
    
    # 初始化物理分析器来获取GT数据
    analyzer = GPTPhysicsAnalyzer()
    gt_data = analyzer.get_gt_data(
        data_dir=data_dir,
        dataset_class=version_config['dataset_class'],
        property_type=version_config['property'],
        version=version
    )
    
    if not gt_data:
        print(f"无法加载GT数据for版本: {version}")
        return
    
    print(f"已加载GT数据，包含 {len(gt_data)} 个条目")
    
    # 分割内容为行
    lines = content.split('\n')
    updated_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 查找比较对的行
        if line.startswith('视频对: '):
            # 提取视频文件名
            video_pair_match = re.search(r'视频对: (.+?) vs (.+?)$', line)
            if video_pair_match:
                video1, video2 = video_pair_match.groups()
                
                # 获取GT值
                gt_value1 = extract_gt_value_from_filename(video1, gt_data)
                gt_value2 = extract_gt_value_from_filename(video2, gt_data)
                
                if gt_value1 is not None and gt_value2 is not None:
                    # 计算GT结果（第一个GT值 > 第二个GT值 → GT结果为1，否则为0）
                    gt_result = 1 if gt_value1 > gt_value2 else 0
                    
                    # 更新当前行
                    updated_lines.append(line)
                    
                    # 跳过预测结果行
                    i += 1
                    if i < len(lines):
                        updated_lines.append(lines[i])
                    
                    # 更新GT结果行
                    i += 1
                    if i < len(lines):
                        updated_lines.append(f"GT结果: {gt_result}")
                    
                    # 更新GT值行
                    i += 1
                    if i < len(lines):
                        updated_lines.append(f"GT值: {gt_value1} vs {gt_value2}")
                    
                    print(f"更新: {video1} ({gt_value1}) vs {video2} ({gt_value2}) → GT结果: {gt_result}")
                else:
                    # 如果无法获取GT值，保持原样
                    updated_lines.append(line)
                    if gt_value1 is None:
                        print(f"警告: 无法获取GT值 for {video1}")
                    if gt_value2 is None:
                        print(f"警告: 无法获取GT值 for {video2}")
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
        
        i += 1
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"文件更新完成: {file_path}")

def main():
    """主函数：更新所有结果文件中的GT值"""
    # 加载配置
    config_file = 'vjepa_data_dirs.json'
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    # 添加dataset_class信息
    dataset_class_mapping = {
        "v110_vjepa": "WebVidv3ViscosityTest",
        "v120_vjepa": "WebVidv2Viscosity",
        "v119_vjepa": "WebVidv3ViscosityTest",
        "v121_vjepa": "WebVidv2Viscosity",
        "v124_vjepa": "WebVidv2Viscosity",
        "v125_vjepa": "WebVidv2Viscosity"
    }
    
    config = {}
    for version, version_config in base_config.items():
        config[version] = {
            "property": version_config["property"],
            "type": version_config["type"],
            "data_dirs": version_config["data_dirs"],
            "dataset_class": dataset_class_mapping.get(version, "WebVidv3ViscosityTest")
        }
    
    print(f"已加载配置: {len(config)} 个测试版本")
    
    # 查找所有结果文件
    result_files = []
    
    # GPT结果文件
    gpt_results_dir = 'gpt_physics_analysis_results'
    if os.path.exists(gpt_results_dir):
        for file in os.listdir(gpt_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(gpt_results_dir, file))
    
    # Gemini结果文件
    gemini_results_dir = 'gemini_physics_analysis_results'
    if os.path.exists(gemini_results_dir):
        for file in os.listdir(gemini_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(gemini_results_dir, file))
    
    # Qwen结果文件
    qwen_results_dir = 'qwen_physics_analysis_results'
    if os.path.exists(qwen_results_dir):
        for file in os.listdir(qwen_results_dir):
            if file.endswith('_results.txt'):
                result_files.append(os.path.join(qwen_results_dir, file))
    
    print(f"找到 {len(result_files)} 个结果文件需要更新:")
    for file in result_files:
        print(f"  - {file}")
    
    # 更新每个文件
    for file_path in result_files:
        try:
            update_results_file(file_path, config)
        except Exception as e:
            print(f"更新文件 {file_path} 时出错: {e}")
    
    print("所有文件更新完成!")

if __name__ == "__main__":
    main() 