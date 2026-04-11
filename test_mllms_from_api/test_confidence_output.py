#!/usr/bin/env python3
"""
测试置信度输出功能
演示相对测试中新的0-1置信度分数输出
"""

import re
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_physics_analyzer import GPTPhysicsAnalyzer


def test_confidence_extraction():
    """
    测试置信度分数提取功能
    """
    print("=" * 60)
    print("测试置信度分数提取功能")
    print("=" * 60)
    
    # 创建分析器实例
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试用例：不同置信度分数的GPT响应
    test_cases = [
        {
            "response": "After analyzing both videos carefully, I observe that VIDEO 1 shows higher friction than VIDEO 2. Comparison result: 0.8",
            "expected": 0.8,
            "description": "标准置信度格式 - 高置信度偏向VIDEO 1"
        },
        {
            "response": "The motion in VIDEO 2 appears to have lower friction compared to VIDEO 1. Comparison result: 0.2",
            "expected": 0.2,
            "description": "标准置信度格式 - 高置信度偏向VIDEO 2"
        },
        {
            "response": "Both videos show similar friction levels, it's difficult to determine a clear winner. Comparison result: 0.5",
            "expected": 0.5,
            "description": "标准置信度格式 - 不确定/相等"
        },
        {
            "response": "Analyzing the videos, I believe VIDEO 1 has quite high friction. Comparison result: 0.9",
            "expected": 0.9,
            "description": "标准置信度格式 - 很高置信度偏向VIDEO 1"
        },
        {
            "response": "VIDEO 2 clearly shows much lower friction than VIDEO 1. Comparison result: 0.1",
            "expected": 0.1,
            "description": "标准置信度格式 - 很高置信度偏向VIDEO 2"
        },
        {
            "response": "I can see that VIDEO 1 has higher friction levels than VIDEO 2.",
            "expected": 0.8,
            "description": "备用提取 - 从文本推断偏向VIDEO 1"
        },
        {
            "response": "The analysis shows that VIDEO 2 has higher friction properties.",
            "expected": 0.2,
            "description": "备用提取 - 从文本推断偏向VIDEO 2"
        },
        {
            "response": "Both videos appear to have equal or similar friction characteristics.",
            "expected": 0.5,
            "description": "备用提取 - 从equal关键词推断"
        },
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case['description']}")
        print(f"输入响应: {case['response']}")
        
        # 测试提取功能
        extracted_confidence = analyzer.extract_comparison_result(case['response'])
        
        print(f"期望置信度: {case['expected']}")
        print(f"提取置信度: {extracted_confidence}")
        
        # 检查结果
        if extracted_confidence is not None and abs(extracted_confidence - case['expected']) < 0.01:
            print("✓ 测试通过")
            success_count += 1
        elif extracted_confidence == case['expected']:
            print("✓ 测试通过")
            success_count += 1
        else:
            print("✗ 测试失败")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_count} 通过")
    print("=" * 60)
    
    return success_count == total_count


def test_prompt_generation():
    """
    测试新的提示词生成功能
    """
    print("\n" + "=" * 60)
    print("测试新提示词生成功能")
    print("=" * 60)
    
    analyzer = GPTPhysicsAnalyzer()
    
    # 测试相对比较的提示词
    prompt = analyzer.get_analysis_prompt("friction", "relative")
    
    print("相对比较提示词（friction）:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # 检查关键词
    key_phrases = [
        "confidence score between 0 and 1",
        "Comparison result: X",
        "X = 1.0: Very confident that VIDEO 1",
        "X = 0.0: Very confident that VIDEO 2",
        "closer to 1.0"
    ]
    
    missing_phrases = []
    for phrase in key_phrases:
        if phrase not in prompt:
            missing_phrases.append(phrase)
    
    if not missing_phrases:
        print("✓ 提示词包含所有关键短语")
        return True
    else:
        print("✗ 提示词缺少以下关键短语:")
        for phrase in missing_phrases:
            print(f"  - {phrase}")
        return False


def main():
    """
    主测试函数
    """
    print("置信度输出功能测试")
    print("=" * 80)
    
    try:
        # 测试提取功能
        extraction_passed = test_confidence_extraction()
        
        # 测试提示词生成
        prompt_passed = test_prompt_generation()
        
        # 总结
        print("\n" + "=" * 80)
        print("测试总结:")
        print(f"置信度提取功能: {'✓ 通过' if extraction_passed else '✗ 失败'}")
        print(f"提示词生成功能: {'✓ 通过' if prompt_passed else '✗ 失败'}")
        
        if extraction_passed and prompt_passed:
            print("\n🎉 所有测试通过！置信度输出功能已成功实现")
            print("\n现在相对测试将输出0-1之间的置信度值，可用于计算AUC分数")
            print("  - 0.0-0.5: 倾向于VIDEO 2有更高的物理属性值") 
            print("  - 0.5-1.0: 倾向于VIDEO 1有更高的物理属性值")
            print("  - 越接近0或1，置信度越高")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 