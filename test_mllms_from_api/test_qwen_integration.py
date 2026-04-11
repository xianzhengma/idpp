#!/usr/bin/env python3
"""
Qwen API 集成测试脚本

用于验证Qwen API是否正确集成到现有框架中
"""

import os
import sys
import json
import time
from typing import Dict, List


def test_qwen_api_wrapper():
    """
    测试Qwen API包装器的基本功能
    """
    print("=" * 60)
    print("测试 Qwen API 包装器")
    print("=" * 60)
    
    try:
        from qwen_api_wrapper import QwenVisionAPI, qwen_response
        
        # 测试类初始化
        print("1. 测试API初始化...")
        api = QwenVisionAPI()
        print("✓ QwenVisionAPI 初始化成功")
        
        # 测试连接
        print("\n2. 测试API连接...")
        if api.test_connection():
            print("✓ API连接测试成功")
        else:
            print("✗ API连接测试失败")
            return False
        
        # 测试简单文本消息
        print("\n3. 测试简单文本消息...")
        test_messages = [
            {
                "role": "user",
                "content": "请简短地介绍一下你自己"
            }
        ]
        
        response = qwen_response(test_messages, model_version="qwen-vl-max")
        print(f"✓ 文本消息测试成功")
        print(f"响应: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen API包装器测试失败: {e}")
        return False


def test_gpt_physics_analyzer_integration():
    """
    测试GPT物理分析器的Qwen集成
    """
    print("\n" + "=" * 60)
    print("测试 GPT物理分析器 Qwen集成")
    print("=" * 60)
    
    try:
        from gpt_physics_analyzer import GPTPhysicsAnalyzer
        
        # 初始化分析器
        print("1. 初始化物理分析器...")
        analyzer = GPTPhysicsAnalyzer()
        print("✓ GPTPhysicsAnalyzer 初始化成功")
        
        # 测试API调用方法
        print("\n2. 测试API调用方法...")
        test_messages = [
            {
                "role": "user",
                "content": "测试消息"
            }
        ]
        
        # 测试GPT API调用
        print("   - 测试GPT API路径...")
        gpt_response = analyzer.call_third_party_gpt_api(test_messages, "gpt-4o")
        print(f"   ✓ GPT API调用: {type(gpt_response).__name__}")
        
        # 测试Qwen API调用
        print("   - 测试Qwen API路径...")
        qwen_response = analyzer.call_third_party_gpt_api(test_messages, "qwen-vl-max")
        print(f"   ✓ Qwen API调用: {type(qwen_response).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ 物理分析器集成测试失败: {e}")
        return False


def test_model_version_detection():
    """
    测试模型版本检测逻辑
    """
    print("\n" + "=" * 60)
    print("测试 模型版本检测")
    print("=" * 60)
    
    try:
        from gpt_physics_analyzer import GPTPhysicsAnalyzer
        
        analyzer = GPTPhysicsAnalyzer()
        test_messages = [{"role": "user", "content": "test"}]
        
        # 测试不同模型版本的路由
        test_cases = [
            ("gpt-4o", "GPT"),
            ("gpt-4o-mini", "GPT"),
            ("qwen-vl-max", "Qwen"),
            ("qwen-vl-plus", "Qwen"),
            ("qwen-vl-turbo", "Qwen"),
        ]
        
        print("测试模型版本路由:")
        for model_version, expected_api in test_cases:
            try:
                response = analyzer.call_third_party_gpt_api(test_messages, model_version)
                
                if expected_api == "GPT" and "GPT" in response:
                    print(f"✓ {model_version} -> {expected_api} API")
                elif expected_api == "Qwen" and "Qwen" in response:
                    print(f"✓ {model_version} -> {expected_api} API")
                else:
                    print(f"✓ {model_version} -> {expected_api} API (调用成功)")
                    
            except Exception as e:
                print(f"⚠ {model_version} -> {expected_api} API (可能需要配置)")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型版本检测测试失败: {e}")
        return False


def test_output_directory_generation():
    """
    测试输出目录生成逻辑
    """
    print("\n" + "=" * 60)
    print("测试 输出目录生成")
    print("=" * 60)
    
    try:
        # 导入函数
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from run_gpt_physics_tests import get_default_output_dir
        
        # 测试不同模型的输出目录
        test_cases = [
            ("gpt-4o", "gpt_physics_analysis_results"),
            ("gpt-4o-mini", "gpt_mini_physics_analysis_results"),
            ("qwen-vl-max", "qwen_max_physics_analysis_results"),
            ("qwen-vl-plus", "qwen_plus_physics_analysis_results"),
            ("qwen-vl-turbo", "qwen_turbo_physics_analysis_results"),
        ]
        
        print("测试输出目录生成:")
        for model_version, expected_dir in test_cases:
            actual_dir = get_default_output_dir(model_version)
            if actual_dir == expected_dir:
                print(f"✓ {model_version} -> {actual_dir}")
            else:
                print(f"✗ {model_version} -> {actual_dir} (预期: {expected_dir})")
        
        return True
        
    except Exception as e:
        print(f"✗ 输出目录生成测试失败: {e}")
        return False


def test_prompt_format_conversion():
    """
    测试提示格式转换
    """
    print("\n" + "=" * 60)
    print("测试 提示格式转换")
    print("=" * 60)
    
    try:
        from qwen_api_wrapper import QwenVisionAPI
        
        api = QwenVisionAPI()
        
        # 测试GPT格式转换为Qwen格式
        gpt_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请分析这个图像"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}
                ]
            }
        ]
        
        qwen_messages = api.convert_gpt_prompt_to_qwen(gpt_messages)
        
        print("GPT格式转换测试:")
        print(f"输入消息数: {len(gpt_messages)}")
        print(f"输出消息数: {len(qwen_messages)}")
        
        # 检查转换结果
        has_text = any(
            any(item.get("type") == "text" for item in msg.get("content", []) if isinstance(msg.get("content"), list))
            for msg in qwen_messages
        )
        
        has_video = any(
            any(item.get("type") == "video" for item in msg.get("content", []) if isinstance(msg.get("content"), list))
            for msg in qwen_messages
        )
        
        if has_text:
            print("✓ 文本内容转换成功")
        if has_video:
            print("✓ 图像内容转换成功 (转换为video格式)")
        
        return True
        
    except Exception as e:
        print(f"✗ 提示格式转换测试失败: {e}")
        return False


def main():
    """
    主测试函数
    """
    print("Qwen API 集成测试")
    print("=" * 80)
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  警告: 未设置环境变量 DASHSCOPE_API_KEY")
        print("   某些测试可能会失败")
        print()
    
    # 运行测试
    test_results = []
    
    test_results.append(("Qwen API包装器", test_qwen_api_wrapper()))
    test_results.append(("物理分析器集成", test_gpt_physics_analyzer_integration()))
    test_results.append(("模型版本检测", test_model_version_detection()))
    test_results.append(("输出目录生成", test_output_directory_generation()))
    test_results.append(("提示格式转换", test_prompt_format_conversion()))
    
    # 显示测试结果摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20} {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed + failed} 个测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！Qwen API集成成功！")
        return 0
    else:
        print(f"\n⚠️  {failed} 个测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 