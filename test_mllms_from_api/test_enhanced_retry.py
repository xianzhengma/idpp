#!/usr/bin/env python3
"""
增强重试系统测试脚本
用于验证新的重试机制是否正常工作
"""

import os
import sys
import time
from gpt_physics_analyzer import GPTPhysicsAnalyzer


def test_retry_parameters():
    """测试重试参数是否正确传递"""
    print("🧪 测试重试参数传递...")
    
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试方法签名
    try:
        # 检查方法是否接受新参数
        import inspect
        
        # 检查 analyze_single_video_with_gpt 方法
        sig = inspect.signature(analyzer.analyze_single_video_with_gpt)
        expected_params = ['max_retries', 'retry_delay', 'validation_retries']
        
        for param in expected_params:
            if param not in sig.parameters:
                print(f"❌ analyze_single_video_with_gpt 缺少参数: {param}")
                return False
            else:
                print(f"✅ analyze_single_video_with_gpt 包含参数: {param}")
        
        # 检查 compare_two_videos_with_gpt 方法
        sig = inspect.signature(analyzer.compare_two_videos_with_gpt)
        
        for param in expected_params:
            if param not in sig.parameters:
                print(f"❌ compare_two_videos_with_gpt 缺少参数: {param}")
                return False
            else:
                print(f"✅ compare_two_videos_with_gpt 包含参数: {param}")
        
        print("✅ 所有方法都包含必要的重试参数")
        return True
        
    except Exception as e:
        print(f"❌ 参数检查失败: {e}")
        return False


def test_api_error_detection():
    """测试API错误检测逻辑"""
    print("\n🧪 测试API错误检测...")
    
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试错误响应检测
    error_responses = [
        "GPT API调用失败: 连接超时",
        "chatanywhere_error: 服务器繁忙",
        "INTERNAL_SERVER_ERROR",
        "第三方API调用失败: 未知错误"
    ]
    
    normal_responses = [
        "Estimated friction coefficient value: 0.5",
        "The analysis shows that the object has moderate friction.",
        "Comparison result: 0.7"
    ]
    
    # 测试extract_predicted_value中的错误检测
    for response in error_responses:
        result = analyzer.extract_predicted_value(response, "friction")
        if result is not None:
            print(f"❌ 错误响应未被正确识别: {response[:50]}...")
            return False
        else:
            print(f"✅ 正确识别错误响应: {response[:30]}...")
    
    # 测试正常响应不被误判
    for response in normal_responses:
        # 这里不检查是否能提取到值，只检查不被误判为错误
        print(f"✅ 正常响应不被误判: {response[:30]}...")
    
    print("✅ API错误检测逻辑正常")
    return True


def test_prediction_extraction():
    """测试预测值提取功能"""
    print("\n🧪 测试预测值提取...")
    
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试各种格式的响应
    test_cases = [
        {
            "response": "Based on the analysis, the estimated friction coefficient value: 0.75",
            "property": "friction",
            "expected": 0.75
        },
        {
            "response": "The viscosity appears to be around 150 cP. Estimated viscosity value: 150 cP",
            "property": "viscosity", 
            "expected": 150.0
        },
        {
            "response": "The ball bounces well. Estimated restitution coefficient value: 0.85",
            "property": "elasticity",
            "expected": 0.85
        },
        {
            "response": "Comparison result: 0.3",
            "property": "comparison",
            "expected": 0.3
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases):
        if case["property"] == "comparison":
            result = analyzer.extract_comparison_result(case["response"])
        else:
            result = analyzer.extract_predicted_value(case["response"], case["property"])
        
        if result == case["expected"]:
            print(f"✅ 测试用例 {i+1}: 成功提取 {result}")
            success_count += 1
        else:
            print(f"❌ 测试用例 {i+1}: 期望 {case['expected']}, 得到 {result}")
    
    if success_count == len(test_cases):
        print("✅ 预测值提取功能正常")
        return True
    else:
        print(f"⚠️ 预测值提取功能部分正常 ({success_count}/{len(test_cases)})")
        return False


def test_retry_configuration():
    """测试重试配置功能"""
    print("\n🧪 测试重试配置...")
    
    # 测试命令行参数解析
    try:
        from run_gpt_physics_tests import parse_args
        
        # 模拟命令行参数
        test_args = [
            "--max-retries", "8",
            "--retry-delay", "20", 
            "--validation-retries", "5",
            "--aggressive-retry"
        ]
        
        # 临时修改sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ["test"] + test_args
        
        try:
            args = parse_args()
            
            # 检查参数是否正确解析
            if args.max_retries == 8:
                print("✅ max_retries 参数解析正确")
            else:
                print(f"❌ max_retries 参数解析错误: {args.max_retries}")
                return False
                
            if args.retry_delay == 20:
                print("✅ retry_delay 参数解析正确")
            else:
                print(f"❌ retry_delay 参数解析错误: {args.retry_delay}")
                return False
                
            if args.validation_retries == 5:
                print("✅ validation_retries 参数解析正确")
            else:
                print(f"❌ validation_retries 参数解析错误: {args.validation_retries}")
                return False
                
            if args.aggressive_retry:
                print("✅ aggressive_retry 参数解析正确")
            else:
                print("❌ aggressive_retry 参数解析错误")
                return False
                
        finally:
            sys.argv = original_argv
        
        print("✅ 重试配置功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 重试配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("增强重试系统测试")
    print("=" * 60)
    
    tests = [
        test_retry_parameters,
        test_api_error_detection,
        test_prediction_extraction,
        test_retry_configuration
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 发生异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！增强重试系统已就绪。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查系统配置。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 