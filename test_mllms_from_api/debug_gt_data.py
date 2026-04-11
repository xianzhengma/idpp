#!/usr/bin/env python3
"""
调试GT数据读取逻辑
"""

import json
import os
from gpt_physics_analyzer import GPTPhysicsAnalyzer


def test_gt_data_reading():
    """测试GT数据读取逻辑"""
    
    # 创建分析器
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试案例
    test_cases = [
        {
            "version": "v110_vjepa",
            "data_dir": "/scratch/shared/beegfs/guanqi/genesis_data/6.2_viscosity_test3",
            "property_type": "viscosity",
            "dataset_class": "WebVidv3ViscosityTest",
            "sample_video": "video_0_id_0_friction_sim_box_objFriction_2.0_cola_1_1_rgb.mp4"
        },
        {
            "version": "v120_vjepa", 
            "data_dir": "/scratch/shared/beegfs/guanqi/genesis_data/5.2_viscosity_synthetic_test1",
            "property_type": "friction",
            "dataset_class": "WebVidv2Viscosity",
            "sample_video": "bounce_ball_E_0.005039669168531562_color_0.81-1.00-0_camangle_0.38_camheight_-0.41_camlookat_0.36_-0.23_-0.11_height_0.50_rgb.mp4"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试版本: {test_case['version']}")
        print(f"数据目录: {test_case['data_dir']}")
        print(f"物理属性: {test_case['property_type']}")
        print(f"数据集类: {test_case['dataset_class']}")
        print(f"样本视频: {test_case['sample_video']}")
        print(f"{'='*60}")
        
        # 检查数据目录
        if not os.path.exists(test_case['data_dir']):
            print(f"✗ 数据目录不存在: {test_case['data_dir']}")
            continue
        
        # 加载GT数据
        gt_data = analyzer.get_gt_data(
            test_case['data_dir'], 
            test_case['dataset_class'], 
            test_case['property_type'],
            test_case['version']
        )
        
        if gt_data is None:
            print("✗ 无法加载GT数据")
            continue
        
        print(f"✓ 成功加载GT数据，包含 {len(gt_data)} 个条目")
        
        # 显示GT数据的前几个键
        print(f"GT数据键示例: {list(gt_data.keys())[:5]}")
        
        # 测试样本视频的GT值获取
        gt_value = analyzer.get_gt_value(
            test_case['sample_video'],
            gt_data,
            test_case['dataset_class'],
            test_case['property_type']
        )
        
        print(f"样本视频GT值: {gt_value}")
        
        # 测试文件名解析
        basename = test_case['sample_video'].replace('_rgb.mp4', '')
        print(f"文件名basename: {basename}")
        
        if basename in gt_data:
            print(f"✓ 在GT数据中找到对应条目: {gt_data[basename]}")
        else:
            print(f"✗ 在GT数据中未找到对应条目")
            
            # 尝试文件名解析
            parsed_value = analyzer.extract_property_from_filename(
                test_case['sample_video'],
                test_case['property_type'],
                test_case['dataset_class']
            )
            print(f"文件名解析结果: {parsed_value}")
            
            # 寻找相似的键
            similar_keys = [key for key in gt_data.keys() if basename.split('_')[0] in key]
            print(f"相似的键: {similar_keys[:5]}")
        
        # 统计GT数据中的值分布
        gt_values = []
        for key, value in gt_data.items():
            if isinstance(value, dict):
                if 'gt_restitution' in value:
                    gt_values.append(value['gt_restitution'])
            else:
                gt_values.append(value)
        
        zero_count = sum(1 for v in gt_values if v == 0.0)
        non_zero_count = len(gt_values) - zero_count
        
        print(f"GT值统计: 总数={len(gt_values)}, 非零={non_zero_count}, 零值={zero_count}")
        if gt_values:
            print(f"GT值范围: {min(gt_values)} - {max(gt_values)}")
        
        # 检查所有视频文件
        print(f"\n检查目录中的所有视频文件...")
        video_files = [f for f in os.listdir(test_case['data_dir']) if f.endswith('_rgb.mp4')]
        print(f"视频文件总数: {len(video_files)}")
        
        # 检查有多少视频文件有GT数据
        matched_count = 0
        for video_file in video_files[:10]:  # 只检查前10个
            basename = video_file.replace('_rgb.mp4', '')
            if basename in gt_data:
                matched_count += 1
        
        print(f"前10个视频文件中有GT数据的: {matched_count}/10")


if __name__ == "__main__":
    test_gt_data_reading() 