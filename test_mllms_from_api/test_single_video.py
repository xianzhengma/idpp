#!/usr/bin/env python3
"""
单个视频测试脚本
用于调试GPT回复内容和提取逻辑
"""

import os
import sys
import json
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def test_single_video():
    """测试单个视频"""
    print("=" * 80)
    print("单个视频测试 - 用于调试GPT回复和提取逻辑")
    print("=" * 80)
    
    # 初始化分析器
    try:
        analyzer = GPTPhysicsAnalyzer()
        print("✓ 分析器初始化成功")
    except Exception as e:
        print(f"✗ 分析器初始化失败: {e}")
        return
    
    # 选择一个测试配置
    config = analyzer.config
    print(f"\n可用的测试配置:")
    for i, (version, version_config) in enumerate(config.items()):
        print(f"{i+1}. {version}: {version_config['property']} ({version_config['type']})")
    
    # 选择第一个配置进行测试
    version_name = list(config.keys())[0]
    version_config = config[version_name]
    
    print(f"\n使用配置: {version_name}")
    print(f"属性: {version_config['property']}")
    print(f"类型: {version_config['type']}")
    
    # 选择第一个测试集
    test_id = list(version_config['data_dirs'].keys())[0]
    data_dir = version_config['data_dirs'][test_id]
    
    print(f"测试集: {test_id}")
    print(f"数据目录: {data_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"✗ 数据目录不存在: {data_dir}")
        return
    
    # 获取一个视频文件
    videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
    if not videos:
        print(f"✗ 目录中没有找到rgb.mp4文件")
        return
    
    # 选择第一个视频
    video_file = videos[0]
    video_path = os.path.join(data_dir, video_file)
    
    print(f"\n测试视频: {video_file}")
    print(f"完整路径: {video_path}")
    
    # 获取数据集类和GT数据
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    print(f"数据集类: {dataset_class}")
    
    gt_data = analyzer.get_gt_data(data_dir, dataset_class, version_config['property'])
    gt_value = analyzer.get_gt_value(video_file, gt_data, dataset_class, version_config['property'])
    print(f"GT值: {gt_value}")
    
    print("\n" + "=" * 80)
    print("开始GPT分析...")
    print("=" * 80)
    
    try:
        # 进行GPT分析
        gpt_result = analyzer.analyze_single_video_with_gpt(
            video_path, 
            version_config['property'],
            max_frames=8,  # 少量帧用于测试
            frame_size=(320, 240),  # 小分辨率
            # model_version="gpt-4o"  # 使用默认模型
            model_version="gemini-2.5-pro"
        )
        
        print(f"\n" + "=" * 80)
        print("提取预测值...")
        print("=" * 80)
        
        # 提取预测值（这里会打印调试信息）
        predicted_value = analyzer.extract_predicted_value(gpt_result, version_config['property'])
        
        print(f"\n" + "=" * 80)
        print("测试结果总结")
        print("=" * 80)
        print(f"视频: {video_file}")
        print(f"属性: {version_config['property']}")
        print(f"GT值: {gt_value}")
        print(f"预测值: {predicted_value}")
        
        if predicted_value is not None:
            print("✓ 成功提取预测值")
        else:
            print("✗ 未能提取预测值")
            
    except Exception as e:
        print(f"✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    test_single_video()

if __name__ == "__main__":
    main() 