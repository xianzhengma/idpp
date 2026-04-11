#!/usr/bin/env python3
"""
专门测试相对比较的脚本
"""

import os
import sys
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
            return "gpt_mini_physics_analysis_results"
        elif "turbo" in model_name:
            return "gpt_turbo_physics_analysis_results"
        else:
            return "gpt_physics_analysis_results"
    elif model_name.startswith("gemini"):
        return "gemini_physics_analysis_results"
    elif model_name.startswith("claude"):
        return "claude_physics_analysis_results"
    else:
        # 其他模型使用通用格式
        model_prefix = model_name.split("_")[0]
        return f"{model_prefix}_physics_analysis_results"


def test_relative_version(analyzer, version_name, version_config, max_samples=3, frame_size=(320, 240), model_version="gpt-4o", output_dir=None):
    """
    测试单个相对比较版本
    
    参数:
        analyzer: GPTPhysicsAnalyzer实例
        version_name: 版本名称
        version_config: 版本配置
        max_samples: 最大样本数（配对数）
        frame_size: 帧大小
        model_version: GPT模型版本
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"开始测试版本: {version_name}")
    print(f"属性: {version_config['property']}")
    print(f"类型: {version_config['type']}")
    print(f"每个测试集最大配对数: {max_samples}")
    print(f"GPT模型版本: {model_version}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")
    
    # 确定数据集类
    dataset_class = analyzer.version_dataset_mapping.get(version_name)
    
    all_results = []
    
    # 测试每个数据集
    for test_id, data_dir in version_config['data_dirs'].items():
        print(f"\n--- 测试集: {test_id} ---")
        print(f"数据目录: {data_dir}")
        
        try:
            # 检查测试是否已完成
            if analyzer.is_test_completed(output_dir, version_name, test_id, 'relative'):
                print(f"✓ 测试 {version_name}_{test_id}_relative 已完成，跳过")
                continue
            
            # 运行相对测试
            result = analyzer.run_relative_test(
                version_name, 
                version_config['property'], 
                test_id, 
                data_dir, 
                dataset_class, 
                frame_size, 
                max_samples,
                model_version
            )
            
            all_results.append(result)
            
            # 保存结果
            analyzer.save_test_result(result, output_dir)
            
            # 显示结果摘要
            success_rate = result['successful_analyses'] / result['total_samples'] * 100 if result['total_samples'] > 0 else 0
            print(f"✓ 测试完成: {result['successful_analyses']}/{result['total_samples']} 成功 ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"✗ 测试集 {test_id} 失败: {str(e)}")
            continue
    
    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="专门测试相对比较的脚本")
    parser.add_argument(
        "--version", 
        type=str, 
        help="指定要测试的版本（如v125_vjepa），不指定则测试所有相对版本"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=3,
        help="每个测试集的最大配对数（默认：3）"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="gpt-4o",
        help="GPT模型版本（默认：gpt-4o）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录（默认：根据模型版本自动生成）"
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
    
    args = parser.parse_args()
    
    # 根据模型版本生成输出目录（如果用户没有指定）
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.model_version)
    
    # 创建分析器
    print("正在初始化GPT物理属性分析器...")
    try:
        analyzer = GPTPhysicsAnalyzer()
        print(f"已加载配置: {len(analyzer.config)} 个测试版本")
        print("✓ 分析器初始化成功")
    except Exception as e:
        print(f"✗ 分析器初始化失败: {e}")
        return 1
    
    # 设置帧大小
    frame_size = (args.frame_width, args.frame_height)
    
    # 过滤相对测试版本
    relative_versions = {
        name: config for name, config in analyzer.config.items() 
        if config['type'] == 'relative'
    }
    
    if not relative_versions:
        print("✗ 没有找到相对测试版本")
        return 1
    
    print(f"\n找到 {len(relative_versions)} 个相对测试版本:")
    for name, config in relative_versions.items():
        print(f"  - {name}: {config['property']}")
    
    # 如果指定了版本，只测试该版本
    if args.version:
        if args.version not in relative_versions:
            print(f"✗ 版本 {args.version} 不存在或不是相对测试版本")
            print(f"可用的相对测试版本: {list(relative_versions.keys())}")
            return 1
        
        versions_to_test = {args.version: relative_versions[args.version]}
        print(f"\n只测试指定版本: {args.version}")
    else:
        versions_to_test = relative_versions
        print(f"\n测试所有相对版本")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # 测试选定的版本
    for version_name, version_config in versions_to_test.items():
        results = test_relative_version(
            analyzer, 
            version_name, 
            version_config, 
            args.samples, 
            frame_size,
            args.model_version,
            args.output_dir
        )
        all_results.extend(results)
    
    # 保存汇总结果
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"relative_test_summary_{timestamp}.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 相对测试汇总结果 ===\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPT模型版本: {args.model_version}\n")
            f.write(f"总测试版本数: {len(set(r['version'] for r in all_results))}\n")
            f.write(f"总测试集数: {len(all_results)}\n")
            f.write(f"总配对数: {sum(r['total_samples'] for r in all_results)}\n")
            f.write(f"成功分析数: {sum(r['successful_analyses'] for r in all_results)}\n")
            f.write(f"失败分析数: {sum(r['failed_analyses'] for r in all_results)}\n")
            f.write("\n")
            
            for result in all_results:
                f.write(f"版本: {result['version']}, 测试集: {result['test_set']}, ")
                f.write(f"成功: {result['successful_analyses']}/{result['total_samples']}\n")
        
        print(f"\n✓ 汇总结果已保存到: {summary_file}")
    
    print(f"\n{'='*80}")
    print("相对测试完成！")
    print(f"结果已保存到: {args.output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 