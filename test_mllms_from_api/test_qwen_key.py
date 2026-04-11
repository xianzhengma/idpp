#!/usr/bin/env python3
"""
测试Qwen API密钥是否正确
"""

import os
from openai import OpenAI


def test_qwen_api_key():
    """测试Qwen API密钥"""
    
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return False
    print("测试Qwen API密钥...")
    print("API密钥: (来自环境变量，已隐藏)")
    print(f"API基础URL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    
    try:
        # 创建OpenAI客户端（使用阿里云DashScope）
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        # 发送简单的文本请求
        print("\n发送测试请求...")
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, please respond in Chinese."
                }
            ],
            max_tokens=50
        )
        
        print("✓ API测试成功!")
        print(f"响应: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ API测试失败: {e}")
        
        # 检查错误类型
        if "401" in str(e) or "invalid_api_key" in str(e):
            print("\n可能的原因:")
            print("1. 这个API密钥不是阿里云DashScope的密钥")
            print("2. API密钥格式不正确")
            print("3. API密钥已过期或被禁用")
            print("\n解决方案:")
            print("1. 请到阿里云DashScope控制台获取正确的API密钥")
            print("   地址: https://dashscope.console.aliyun.com/")
            print("2. 确认您的API密钥是DashScope服务的密钥")
        elif "403" in str(e):
            print("\n可能的原因:")
            print("1. API密钥没有权限访问qwen-vl-max模型")
            print("2. 账户余额不足")
        else:
            print(f"\n其他错误: {e}")
        
        return False


def test_alternative_endpoints():
    """测试不同的API端点"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("请设置 DASHSCOPE_API_KEY")
        return
    
    # 可能的端点
    endpoints = [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-intl.aliyuncs.com/api/v1",
        "https://dashscope.aliyuncs.com/api/v1"
    ]
    
    print("\n测试不同的API端点...")
    
    for endpoint in endpoints:
        print(f"\n尝试端点: {endpoint}")
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=endpoint
            )
            
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            
            print(f"✓ 端点 {endpoint} 工作正常")
            return True
            
        except Exception as e:
            print(f"✗ 端点 {endpoint} 失败: {str(e)[:100]}...")
    
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen API密钥测试")
    print("=" * 60)
    
    # 测试主要API
    success = test_qwen_api_key()
    
    if not success:
        # 如果主要API失败，测试其他端点
        success = test_alternative_endpoints()
    
    if success:
        print("\n🎉 API密钥测试成功，可以使用！")
    else:
        print("\n❌ 所有测试都失败了")
        print("\n建议:")
        print("1. 确认您的API密钥是从阿里云DashScope获取的")
        print("2. 检查账户状态和余额")
        print("3. 尝试重新生成API密钥") 