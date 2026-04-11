import os
import time
import json
from typing import Dict, List, Optional
from openai import OpenAI


class QwenVisionAPI:
    """
    Qwen Vision API调用封装类
    
    提供与现有GPT API兼容的接口，支持视频帧分析
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化Qwen Vision API客户端
        
        参数:
            api_key: API密钥，默认从环境变量DASHSCOPE_API_KEY获取
            base_url: API基础URL，默认使用阿里云DashScope
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or "sk-34a6731731ed411f91b4c759d1c871d9"
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not self.api_key:
            raise ValueError("API密钥未设置，请设置环境变量DASHSCOPE_API_KEY或直接传入api_key参数")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 支持的模型列表
        self.supported_models = [
            "qwen-vl-max",
            "qwen-vl-plus", 
            "qwen-vl-turbo"
        ]
    
    def convert_gpt_prompt_to_qwen(self, prompt_messages: List[Dict]) -> List[Dict]:
        """
        将GPT格式的提示消息转换为Qwen格式
        
        参数:
            prompt_messages: GPT格式的提示消息列表
            
        返回:
            Qwen格式的提示消息列表
        """
        qwen_messages = []
        
        for message in prompt_messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if isinstance(content, str):
                # 纯文本消息
                qwen_messages.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(content, list):
                # 多模态消息（包含文本和图像）
                qwen_content = []
                image_urls = []
                
                for item in content:
                    if item.get("type") == "text":
                        qwen_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image_url":
                        # 收集图像URL，稍后统一处理
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            image_urls.append(image_url)
                
                # 如果有图像，添加video类型的内容（Qwen使用video字段处理图像序列）
                if image_urls:
                    qwen_content.append({
                        "type": "video",
                        "video": image_urls
                    })
                
                qwen_messages.append({
                    "role": role,
                    "content": qwen_content
                })
        
        return qwen_messages
    
    def qwen_response(self, prompt_messages: List[Dict], model_version: str = "qwen-vl-max", max_retries: int = 5) -> str:
        """
        调用Qwen Vision API（兼容GPT API接口）
        
        参数:
            prompt_messages: 提示消息列表（GPT格式）
            model_version: 模型版本
            max_retries: 最大重试次数
            
        返回:
            API响应内容或错误信息
        """
        # 验证模型版本
        if model_version not in self.supported_models:
            print(f"警告: 模型 {model_version} 不在支持列表中，使用默认模型 qwen-vl-max")
            model_version = "qwen-vl-max"
        
        # 转换提示消息格式
        qwen_messages = self.convert_gpt_prompt_to_qwen(prompt_messages)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                print(f"正在调用Qwen API... (尝试 {attempt + 1}/{max_retries})")
                
                # 调用Qwen API
                response = self.client.chat.completions.create(
                    model=model_version,
                    messages=qwen_messages,
                    temperature=0.0,
                    max_tokens=2000
                )
                
                # 提取响应内容
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    print(f"✓ Qwen API调用成功")
                    return content
                else:
                    print(f"✗ Qwen API返回空响应")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "Qwen API调用失败: 返回空响应"
                
            except Exception as e:
                print(f"重试 {attempt + 1}/{max_retries} 失败: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    return f"Qwen API调用失败: 所有重试都失败了 - {str(e)}"
        
        return "Qwen API调用失败: 未知错误"
    
    def test_connection(self) -> bool:
        """
        测试API连接是否正常
        
        返回:
            连接是否成功
        """
        try:
            test_messages = [{
                "role": "user",
                "content": "Hello, test connection"
            }]
            
            response = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=test_messages,
                max_tokens=10
            )
            
            return response.choices and len(response.choices) > 0
            
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False


# 全局函数，兼容现有框架
def qwen_response(prompt_messages: List[Dict], model_version: str = "qwen-vl-max", max_retries: int = 5) -> str:
    """
    Qwen Vision API调用函数（兼容GPT API接口）
    
    参数:
        prompt_messages: 提示消息列表
        model_version: 模型版本
        max_retries: 最大重试次数
        
    返回:
        API响应内容或错误信息
    """
    try:
        # 创建API实例
        qwen_api = QwenVisionAPI()
        
        # 调用API
        return qwen_api.qwen_response(prompt_messages, model_version, max_retries)
        
    except Exception as e:
        return f"Qwen API初始化失败: {str(e)}"


# 测试代码
def test_qwen_api():
    """
    测试Qwen API调用
    """
    try:
        # 测试连接
        qwen_api = QwenVisionAPI()
        print("测试API连接...")
        
        if qwen_api.test_connection():
            print("✓ API连接成功")
        else:
            print("✗ API连接失败")
            return
        
        # 测试文本提示
        test_prompt = [{
            "role": "user",
            "content": "请简短地介绍一下你自己"
        }]
        
        print("\n测试文本提示...")
        response = qwen_response(test_prompt)
        print(f"响应: {response}")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_qwen_api() 