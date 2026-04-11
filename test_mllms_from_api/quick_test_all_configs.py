#!/usr/bin/env python3
"""
快速测试所有配置脚本
每个配置只测试1个视频，验证所有配置是否正常工作
实时保存结果，避免中途失败导致结果丢失
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def get_default_output_dir(model_version: str) -> str:
    """
    根据模型版本生成默认输出目录名
    
    参数:
        model_version: GPT模型版本
        
    返回:
        默认输出目录名
    """
    # 将模型名称转换为目录友好的格式
    model_name = model_version.lower().replace("-", "_").replace(".", "_")
    
    # 根据模型前缀生成目录名
    if model_name.startswith("gpt"):
        if "mini" in model_name:
            return "gpt_mini_quick_test_results"
        elif "turbo" in model_name:
            return "gpt_turbo_quick_test_results"
        else:
            return "gpt_quick_test_results"
    elif model_name.startswith("gemini"):
        return "gemini_quick_test_results"
    elif model_name.startswith("claude"):
        return "claude_quick_test_results"
    else:
        # 其他模型使用通用格式
        model_prefix = model_name.split("_")[0]
        return f"{model_prefix}_quick_test_results"

def save_individual_result(result, output_dir):
    """
    保存单个测试结果
    
    参数:
        result: 测试结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建以版本和测试集命名的文件
    filename = f"{result['version']}_{result['test_set']}_{result['type']}_result.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Quick Test Result - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
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
            f.write("\nIndividual Results:\n")
            for item in result['individual_results']:
                if 'error' in item:
                    f.write(f"  - {item['video']}: ERROR - {item['error']}\n")
                else:
                    f.write(f"  - {item['video']}: predicted={item['predicted']}, gt={item['gt']}\n")
                    # 保存完整GPT回复用于调试
                    if 'gpt_response' in item:
                        f.write(f"    GPT Response: {item['gpt_response'][:200]}...\n")
        
        elif result['type'] == 'relative' and 'comparison_results' in result:
            f.write("\nComparison Results:\n")
            for item in result['comparison_results']:
                if 'error' in item:
                    f.write(f"  - {item['video_pair']}: ERROR - {item['error']}\n")
                else:
                    f.write(f"  - {item['video_pair']}: predicted={item['predicted']}, gt={item['gt']}\n")
                    if 'gt_values' in item:
                        f.write(f"    GT Values: {item['gt_values']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ 结果已保存到: {filepath}")

def test_single_config(analyzer, version_name, version_config, output_dir, frame_size=(320, 240), max_samples=1, model_version="gpt-4o"):
    """
    测试单个配置
    
    参数:
        analyzer: GPT物理分析器
        version_name: 版本名称
        version_config: 版本配置
        output_dir: 输出目录
        frame_size: 视频帧大小
        max_samples: 每个测试集的最大样本数量
        model_version: GPT模型版本
    """
    results = []
    
    # 获取数据集类名
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    if not dataset_class:
        print(f"跳过 {version_name}: 未找到对应的数据集类")
        return results
    
    for test_id, data_dir in version_config['data_dirs'].items():
        print(f"\n--- 测试集: {test_id} ---")
        print(f"数据目录: {data_dir}")
        print(f"最大样本数: {max_samples}")
        print(f"模型版本: {model_version}")
        
        try:
            # 加载GT数据
            gt_data = analyzer.get_gt_data(data_dir, dataset_class, version_config['property'], version=version_name)
            
            # 采样视频
            if version_config['type'] == 'absolute':
                sampled_videos = analyzer.sample_videos(data_dir, max_samples=max_samples)
                if not sampled_videos:
                    raise ValueError(f"未找到可用的视频文件")
                
                # 测试第一个视频
                test_video = sampled_videos[0]
                result = test_absolute_single(
                    analyzer, version_name, version_config['property'], test_id,
                    data_dir, dataset_class, test_video, frame_size, gt_data, model_version
                )
                
            else:  # relative
                # 对于相对比较，需要至少2个视频
                required_videos = max_samples * 2
                sampled_videos = analyzer.sample_videos(data_dir, max_samples=required_videos)
                if len(sampled_videos) < 2:
                    raise ValueError(f"相对比较测试需要至少2个视频文件")
                
                # 测试第一对视频
                test_video = sampled_videos[0]
                test_video2 = sampled_videos[1]
                result = test_relative_single(
                    analyzer, version_name, version_config['property'], test_id,
                    data_dir, dataset_class, test_video, test_video2, frame_size, gt_data, model_version
                )
            
            # 在结果中添加模型版本信息
            result['model_version'] = model_version
            results.append(result)
            
            # 实时保存结果
            save_individual_result(result, output_dir)
            
            # 显示测试结果摘要
            if result['successful_analyses'] > 0:
                print(f"✓ 测试成功: {result['successful_analyses']}/{result['total_samples']}")
            else:
                print(f"✗ 测试失败: {result['failed_analyses']}/{result['total_samples']}")
            
        except Exception as e:
            print(f"✗ 测试失败: {str(e)}")
            
            # 创建失败结果记录
            failed_result = {
                'version': version_name,
                'property': version_config['property'],
                'type': version_config['type'],
                'test_set': test_id,
                'data_dir': data_dir,
                'total_samples': max_samples if version_config['type'] == 'absolute' else 1,
                'successful_analyses': 0,
                'failed_analyses': max_samples if version_config['type'] == 'absolute' else 1,
                'error': str(e),
                'model_version': model_version
            }
            
            if version_config['type'] == 'absolute':
                failed_result['individual_results'] = []
            else:
                failed_result['comparison_results'] = []
            
            results.append(failed_result)
            
            # 也保存失败结果
            save_individual_result(failed_result, output_dir)
        
        # 添加延迟避免API限制
        time.sleep(5)
    
    return results

def test_absolute_single(analyzer, version, property_type, test_id, data_dir, dataset_class, video_file, frame_size, gt_data, model_version):
    """测试单个absolute视频"""
    print(f"开始绝对值测试: {property_type}")
    
    try:
        # 获取GT值
        gt_value = analyzer.get_gt_value(video_file, gt_data, dataset_class, property_type)
        print(f"GT值: {gt_value}")
        
        # GPT分析
        video_path = os.path.join(data_dir, video_file)
        print(f"分析视频: {video_file}")
        
        gpt_result = analyzer.analyze_single_video_with_gpt(
            video_path, property_type, max_frames=16, frame_size=frame_size, model_version=model_version
        )
        
        # 提取预测值
        predicted_value = analyzer.extract_predicted_value(gpt_result, property_type)
        
        individual_results = [{
            'video': video_file,
            'predicted': predicted_value,
            'gt': gt_value,
            'gpt_response': gpt_result
        }]
        
        return {
            'version': version,
            'property': property_type,
            'type': 'absolute',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': 1,
            'successful_analyses': 1 if predicted_value is not None else 0,
            'failed_analyses': 0 if predicted_value is not None else 1,
            'individual_results': individual_results
        }
        
    except Exception as e:
        print(f"分析失败: {str(e)}")
        return {
            'version': version,
            'property': property_type,
            'type': 'absolute',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': 1,
            'successful_analyses': 0,
            'failed_analyses': 1,
            'individual_results': [{
                'video': video_file,
                'predicted': None,
                'gt': None,
                'error': str(e)
            }]
        }

def test_relative_single(analyzer, version, property_type, test_id, data_dir, dataset_class, video_file1, video_file2, frame_size, gt_data, model_version):
    """测试单个relative视频对"""
    print(f"开始相对比较测试: {property_type}")
    
    try:
        # 获取两个视频的GT值
        gt_value1 = analyzer.get_gt_value(video_file1, gt_data, dataset_class, property_type)
        gt_value2 = analyzer.get_gt_value(video_file2, gt_data, dataset_class, property_type)
        
        gt_comparison = 1 if gt_value1 > gt_value2 else 0
        
        print(f"比较视频对: {video_file1} vs {video_file2}")
        print(f"GT值: {gt_value1} vs {gt_value2}, GT比较: {gt_comparison}")
        
        # GPT比较
        video_path1 = os.path.join(data_dir, video_file1)
        video_path2 = os.path.join(data_dir, video_file2)
        
        gpt_result = analyzer.compare_two_videos_with_gpt(
            video_path1, video_path2, property_type, max_frames_per_video=16, frame_size=frame_size, model_version=model_version
        )
        
        # 提取比较结果
        predicted_comparison = analyzer.extract_comparison_result(gpt_result)
        
        comparison_results = [{
            'video_pair': f"{video_file1} vs {video_file2}",
            'predicted': predicted_comparison,
            'gt': gt_comparison,
            'gt_values': f"{gt_value1} vs {gt_value2}",
            'gpt_response': gpt_result
        }]
        
        return {
            'version': version,
            'property': property_type,
            'type': 'relative',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': 2,
            'successful_analyses': 1 if predicted_comparison is not None else 0,
            'failed_analyses': 0 if predicted_comparison is not None else 1,
            'comparison_results': comparison_results
        }
        
    except Exception as e:
        print(f"比较失败: {str(e)}")
        return {
            'version': version,
            'property': property_type,
            'type': 'relative',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': 2,
            'successful_analyses': 0,
            'failed_analyses': 1,
            'comparison_results': [{
                'video_pair': f"{video_file1} vs {video_file2}",
                'predicted': None,
                'gt': None,
                'error': str(e)
            }]
        }

def save_summary_results(all_results, output_dir, model_version):
    """保存汇总结果"""
    summary_file = os.path.join(output_dir, f"quick_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Quick Test Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Version: {model_version}\n")
        
        # 统计信息
        total_configs = len(all_results)
        successful_configs = sum(1 for r in all_results if r['successful_analyses'] > 0)
        failed_configs = total_configs - successful_configs
        
        f.write(f"总配置数: {total_configs}\n")
        f.write(f"成功配置数: {successful_configs}\n")
        f.write(f"失败配置数: {failed_configs}\n")
        f.write(f"成功率: {(successful_configs/total_configs*100):.1f}%\n\n")
        
        # 按属性分类
        properties = ['friction', 'viscosity', 'elasticity']
        for prop in properties:
            prop_results = [r for r in all_results if r['property'] == prop]
            if prop_results:
                f.write(f"{prop.upper()} 测试结果:\n")
                for result in prop_results:
                    status = "✓" if result['successful_analyses'] > 0 else "✗"
                    f.write(f"  {status} {result['version']} ({result['type']}) - {result['test_set']}: {result['successful_analyses']}/{result['total_samples']}\n")
                f.write("\n")
        
        # 详细结果
        f.write("详细结果:\n")
        f.write("-" * 60 + "\n")
        for result in all_results:
            f.write(f"\n配置: {result['version']} - {result['test_set']}\n")
            f.write(f"属性: {result['property']} ({result['type']})\n")
            f.write(f"状态: {'成功' if result['successful_analyses'] > 0 else '失败'}\n")
            if 'error' in result:
                f.write(f"错误: {result['error']}\n")
    
    print(f"✓ 汇总结果已保存到: {summary_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速测试所有配置脚本")
    parser.add_argument(
        "--model-version",
        type=str,
        default="gpt-4o",
        help="GPT模型版本（默认：gpt-4o）"
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="视频帧宽度（默认：320）"
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=240,
        help="视频帧高度（默认：240）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录（默认：根据模型版本自动生成）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="每个测试集的最大样本数量（默认：1）"
    )
    
    args = parser.parse_args()
    
    # 根据模型版本生成输出目录（如果用户没有指定）
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.model_version)
    
    print("开始快速测试所有配置...")
    print(f"模型版本: {args.model_version}")
    print(f"输出目录: {args.output_dir}")
    print(f"每个测试集最大样本数: {args.max_samples}")
    
    # 设置参数
    frame_size = (args.frame_width, args.frame_height)
    
    # 创建分析器
    analyzer = GPTPhysicsAnalyzer()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # 测试所有配置
    for version_name, version_config in analyzer.config.items():
        print(f"\n{'='*80}")
        print(f"测试配置: {version_name}")
        print(f"属性: {version_config['property']}, 类型: {version_config['type']}")
        print(f"模型版本: {args.model_version}")
        print(f"{'='*80}")
        
        results = test_single_config(
            analyzer, version_name, version_config, args.output_dir, 
            frame_size, args.max_samples, args.model_version
        )
        all_results.extend(results)
    
    # 保存汇总结果
    save_summary_results(all_results, args.output_dir, args.model_version)
    
    # 显示最终统计
    total_configs = len(all_results)
    successful_configs = sum(1 for r in all_results if r['successful_analyses'] > 0)
    
    print(f"\n{'='*80}")
    print("快速测试完成！")
    print(f"模型版本: {args.model_version}")
    print(f"总配置数: {total_configs}")
    print(f"成功配置数: {successful_configs}")
    print(f"成功率: {(successful_configs/total_configs*100):.1f}%")
    print(f"结果已保存到: {args.output_dir}")
    print(f"{'='*80}")
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 