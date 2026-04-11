#!/usr/bin/env python3
"""
多模态模型物理属性分析 - 完整测试运行器

使用说明:
- 支持GPT和Qwen模型进行物理属性分析
- 支持设置绝对值测试和相对比较测试的样本数量
- 绝对值测试：每个测试集测试N个视频
- 相对比较测试：每个测试集测试N组成对的视频
"""

import os
import sys
import argparse
from datetime import datetime
from gpt_physics_analyzer import GPTPhysicsAnalyzer


def get_default_output_dir(model_version: str, concat_mode: bool = False, frame_index: bool = False, icl_example: bool = False, follow_oracle_test: bool = False) -> str:
    """
    根据模型版本和启用的功能生成默认输出目录名
    
    参数:
        model_version: 模型版本（支持GPT、Qwen等）
        concat_mode: 是否启用拼接模式
        frame_index: 是否启用帧索引
        icl_example: 是否启用ICL示例
        follow_oracle_test: 是否启用解题步骤指导
        
    返回:
        默认输出目录名（包含功能后缀）
    """
    # 将模型名称转换为目录友好的格式
    model_name = model_version.lower().replace("-", "_").replace(".", "_")
    
    # 根据模型前缀生成基础目录名
    if model_name.startswith("gpt"):
        if "mini" in model_name:
            base_dir = "gpt_mini_physics_analysis_results"
        elif "turbo" in model_name:
            base_dir = "gpt_turbo_physics_analysis_results"
        else:
            base_dir = "gpt_physics_analysis_results"
    elif model_name.startswith("qwen"):
        if "max" in model_name:
            base_dir = "qwen_max_physics_analysis_results"
        elif "plus" in model_name:
            base_dir = "qwen_plus_physics_analysis_results"
        elif "turbo" in model_name:
            base_dir = "qwen_turbo_physics_analysis_results"
        else:
            base_dir = "qwen_physics_analysis_results"
    elif model_name.startswith("gemini"):
        base_dir = "gemini_physics_analysis_results"
    elif model_name.startswith("claude"):
        base_dir = "claude_physics_analysis_results"
    else:
        # 其他模型使用通用格式
        model_prefix = model_name.split("_")[0]
        base_dir = f"{model_prefix}_physics_analysis_results"
    
    # 根据启用的功能添加后缀
    suffixes = []
    if concat_mode:
        suffixes.append("concat")
    if frame_index:
        suffixes.append("frameindex")
    if icl_example:
        suffixes.append("icl")
    if follow_oracle_test:
        suffixes.append("oracle")
    
    if suffixes:
        # 将后缀添加到基础目录名
        suffix_str = "_".join(suffixes)
        return f"{base_dir}_{suffix_str}"
    else:
        return base_dir


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态模型物理属性分析 - 完整测试运行器")
    
    parser.add_argument(
        "--absolute-samples", 
        type=int, 
        default=50,
        help="绝对值测试每个测试集的样本数量（默认：50）"
    )
    
    parser.add_argument(
        "--relative-pairs", 
        type=int, 
        default=50,
        help="相对比较测试每个测试集的成对视频数量（默认：50）"
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
        default="gpt-4o",
        help="模型版本（支持：gpt-4o, qwen-vl-max, qwen-vl-plus, qwen-vl-turbo，默认：gpt-4o）"
    )
    
    parser.add_argument(
        "--frame-width", 
        type=int, 
        default=640,
        help="视频帧宽度（默认：640）"
    )
    
    parser.add_argument(
        "--frame-height", 
        type=int, 
        default=480,
        help="视频帧高度（默认：480）"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,  # 改为None，稍后根据model_version动态生成
        help="结果输出目录（默认：根据模型版本自动生成）"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="仅显示测试计划，不执行实际测试"
    )
    
    parser.add_argument(
        "--auto-confirm", 
        action="store_true",
        help="自动确认测试，无需用户输入"
    )
    
    parser.add_argument(
        "--concat-mode", 
        action="store_true",
        help="使用拼接模式进行相对比较（在两个视频间加黑帧分隔）"
    )
    
    parser.add_argument(
        "--properties",
        type=str,
        nargs="*",
        default=None,
        choices=["friction", "viscosity", "elasticity"],
        help="指定要测试的物理属性（可选：friction, viscosity, elasticity）。默认测试所有属性。示例：--properties friction viscosity"
    )
    
    parser.add_argument(
        "--frame-index",
        action="store_true",
        help="为每个视频帧添加显式的帧索引标识（如：frame1: image, frame2: image）"
    )
    
    parser.add_argument(
        "--icl-example",
        action="store_true",
        help="启用in-context learning，从训练集中采样3个示例作为输入输出对"
    )
    
    parser.add_argument(
        "--follow-oracle-test",
        action="store_true",
        help="启用解题步骤指导，告诉模型具体的分析方法和视觉线索"
    )
    
    # 新增重试配置参数
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="每次API调用的最大重试次数（默认：5）"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=15,
        help="重试间隔时间，单位秒（默认：15）"
    )
    
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=3,
        help="结果验证重试次数，如果提取不到有效值会重新调用API（默认：3）"
    )
    
    parser.add_argument(
        "--aggressive-retry",
        action="store_true",
        help="启用激进重试模式：增加重试次数和验证次数，确保第一次运行时获得更多有效数据"
    )
    
    return parser.parse_args()



def main():
    """主函数"""
    
    # 解析参数
    args = parse_args()
    
    # 根据模型版本和功能生成输出目录（如果用户没有指定）
    if args.output_dir is None:
        args.output_dir = get_default_output_dir(
            model_version=args.model_version,
            concat_mode=args.concat_mode,
            frame_index=args.frame_index,
            icl_example=args.icl_example,
            follow_oracle_test=args.follow_oracle_test
        )
    
    # 处理和验证属性参数
    if args.properties is not None:
        if len(args.properties) == 0:
            # 如果指定了--properties但没有提供参数，显示帮助
            print("错误：--properties 参数需要指定至少一个物理属性")
            print("可选属性：friction, viscosity, elasticity")
            print("示例：--properties friction viscosity")
            return 1
        
        # 验证属性名称（虽然argparse已经做了choices验证，但这里再确认一下）
        valid_properties = ["friction", "viscosity", "elasticity"]
        invalid_properties = [p for p in args.properties if p not in valid_properties]
        if invalid_properties:
            print(f"错误：无效的属性名称: {', '.join(invalid_properties)}")
            print(f"可选属性：{', '.join(valid_properties)}")
            return 1
        
        print(f"指定测试属性: {', '.join(args.properties)}")
    else:
        print("测试所有物理属性")
    
    # 创建分析器
    print("正在初始化多模态模型物理属性分析器...")
    try:
        analyzer = GPTPhysicsAnalyzer()
        print(f"已加载配置: {len(analyzer.config)} 个测试版本")
        print("✓ 分析器初始化成功")
    except Exception as e:
        print(f"✗ 分析器初始化失败: {e}")
        return 1
    
    # 处理激进重试模式
    if args.aggressive_retry:
        print("🚀 激进重试模式已启用")
        # 在激进模式下，如果用户没有明确设置，则使用更高的默认值
        if not any('--max-retries' in arg for arg in sys.argv):
            args.max_retries = 8
        if not any('--validation-retries' in arg for arg in sys.argv):
            args.validation_retries = 5
        if not any('--retry-delay' in arg for arg in sys.argv):
            args.retry_delay = 20
    
    # 输出参数信息
    print(f"\n================================================================================")
    print(f"多模态模型物理属性分析 - 测试计划")
    print(f"================================================================================")
    print(f"模型版本: {args.model_version}")
    print(f"绝对值测试样本数: {args.absolute_samples}")
    print(f"相对比较测试成对数: {args.relative_pairs}")
    print(f"视频帧尺寸: {args.frame_width}x{args.frame_height}")
    print(f"拼接模式: {'启用' if args.concat_mode else '禁用'}")
    print(f"帧索引模式: {'启用' if args.frame_index else '禁用'}")
    print(f"ICL示例: {'启用' if args.icl_example else '禁用'}")
    print(f"解题步骤指导: {'启用' if args.follow_oracle_test else '禁用'}")
    print(f"API重试次数: {args.max_retries}")
    print(f"重试延迟: {args.retry_delay}秒")
    print(f"验证重试次数: {args.validation_retries}")
    print(f"激进重试模式: {'启用' if args.aggressive_retry else '禁用'}")
    if args.properties:
        print(f"测试属性: {', '.join(args.properties)}")
    else:
        print(f"测试属性: 所有属性 (friction, viscosity, elasticity)")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 设置测试参数
    frame_size = (args.frame_width, args.frame_height)
    
    # 根据测试类型选择最大样本数
    max_samples = max(args.absolute_samples, args.relative_pairs)
    
    # 显示测试配置详情
    print("测试配置详情:")
    print()
    
    total_absolute_tests = 0
    total_relative_tests = 0
    
    for version_name, version_config in analyzer.config.items():
        # 如果指定了属性过滤，只显示符合条件的测试
        if args.properties is not None and version_config['property'] not in args.properties:
            continue
        
        # 根据样本数量跳过不需要的测试类型
        if version_config['type'] == 'absolute' and args.absolute_samples == 0:
            continue
        
        if version_config['type'] == 'relative' and args.relative_pairs == 0:
            continue
            
        # 过滤掉训练集，只显示测试集
        test_sets = [k for k in version_config['data_dirs'].keys() if k != 'train']
        
        print(f"{version_name}:")
        print(f"  属性: {version_config['property']}")
        print(f"  类型: {version_config['type']}")
        print(f"  测试集: {test_sets}")
        
        if version_config['type'] == 'absolute':
            print(f"  每个测试集样本数: {args.absolute_samples}")
            print(f"  总测试数: {len(test_sets)}")
            total_absolute_tests += len(test_sets)
        else:
            print(f"  每个测试集成对数: {args.relative_pairs}")
            print(f"  总测试数: {len(test_sets)}")
            total_relative_tests += len(test_sets)
        print()
    
    print("总计:")
    print(f"  绝对值测试: {total_absolute_tests}")
    print(f"  相对比较测试: {total_relative_tests}")
    print(f"  总测试数: {total_absolute_tests + total_relative_tests}")
    
    # 估计时间（每个测试约30秒）
    estimated_minutes = (total_absolute_tests + total_relative_tests) * 0.5
    print(f"  预计用时: {int(estimated_minutes//60)}小时{int(estimated_minutes%60)}分钟")
    print("================================================================================")
    print()
    
    if args.dry_run:
        print("💡 干运行模式：不执行实际测试")
        return 0
    
    # 询问用户确认
    print("准备开始测试...")
    if not args.auto_confirm:
        response = input("是否继续? (y/n, 默认n): ").lower().strip()
        if response not in ['y', 'yes', '是']:
            print("测试已取消")
            return 0
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行测试
    print("开始运行测试...")
    print()
    
    # 使用新的run_all_tests方法，并为不同类型的测试设置不同的参数
    try:
        # 运行所有测试
        results = analyzer.run_all_tests(
            output_dir=args.output_dir,
            frame_size=frame_size,
            max_samples=max_samples,  # 保持向后兼容
            model_version=args.model_version,
            concat_mode=args.concat_mode,
            properties_filter=args.properties,
            frame_index=args.frame_index,
            icl_example=args.icl_example,
            follow_oracle_test=args.follow_oracle_test,
            absolute_samples=args.absolute_samples,
            relative_pairs=args.relative_pairs,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            validation_retries=args.validation_retries,
            aggressive_retry=args.aggressive_retry
        )
        
        print("\n" + "="*50)
        print("✅ 所有测试完成!")
        print(f"结果已保存到: {args.output_dir}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 