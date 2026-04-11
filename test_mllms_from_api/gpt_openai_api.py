import os
import time
import re
import base64
import cv2
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional, Tuple


class GPTOpenAIAnalyzer:
    """
    基于OpenAI官方API的GPT视频分析器
    参考gpt_api_video.py中的原版调用方式
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        初始化GPT OpenAI分析器
        
        参数:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的GPT模型名称
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未设置OPENAI_API_KEY环境变量或未提供api_key参数")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def read_video_frames(self, video_path: str, frame_step: int = 5, frame_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        读取视频文件并将帧转换为base64编码，每隔frame_step帧取一帧
        
        参数:
            video_path: 视频文件的路径
            frame_step: 每隔多少帧取一帧
            frame_size: 调整帧大小的元组 (宽度, 高度)，None表示保持原始大小
            
        返回:
            base64编码的视频帧列表
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        base64_frames = []
        frame_count = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            # 每隔frame_step帧取一帧
            if frame_count % frame_step == 0:
                # 如果指定了帧大小，则调整帧的大小
                if frame_size is not None:
                    frame = cv2.resize(frame, frame_size)
                    
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
                
            frame_count += 1

        video.release()
        
        if frame_size:
            print(f"总共 {frame_count} 帧，采样后得到 {len(base64_frames)} 帧，调整大小为 {frame_size}。")
        else:
            print(f"总共 {frame_count} 帧，采样后得到 {len(base64_frames)} 帧，保持原始大小。")
            
        return base64_frames
    
    def get_analysis_prompt(self, property_type: str, analysis_type: str) -> str:
        """
        根据物理属性类型和分析类型生成相应的提示词
        
        参数:
            property_type: 物理属性类型 ("friction", "viscosity", "elasticity")
            analysis_type: 分析类型 ("absolute", "relative")
            
        返回:
            生成的提示词
        """
        if analysis_type == "absolute":
            if property_type == "friction":
                return """
                You are analyzing frames from a video showing objects with friction interaction. Your task is to estimate the friction coefficient.
                
                IMPORTANT: You CAN analyze these images. You MUST provide a specific friction coefficient estimate.
                
                For reference:
                - Ice on ice: 0.02-0.03
                - Steel on steel: 0.4-0.6
                - Rubber on concrete: 0.6-0.85
                - Wood on wood: 0.3-0.5
                
                Analyze the sliding motion, deceleration rate, and surface interaction.
                
                YOUR RESPONSE MUST END WITH A SPECIFIC FRICTION COEFFICIENT VALUE IN THIS FORMAT:
                "Estimated friction coefficient value: X"
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
            elif property_type == "viscosity":
                return """
                You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.
                
                IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.
                
                For reference:
                - Water: 1 cP
                - Vegetable oil: 100 cP
                - Motor oil: 300 cP
                - Honey: 5000 cP
                
                Analyze the liquid's flow speed, surface waves, and interaction with objects.
                
                YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:
                "Estimated viscosity value: X cP"
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
            elif property_type == "elasticity":
                return """
                Please analyze these frames from a video of a bouncing object and estimate its restitution coefficient.
                
                The restitution coefficient is the ratio of the velocity after collision to the velocity before collision.
                
                Please:
                1. Track the object's position in each frame
                2. Determine its velocity before and after impact
                3. Calculate the restitution coefficient
                4. Provide a detailed analysis of your methodology
                
                YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:
                "Estimated restitution coefficient value: X"
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
        else:  # relative
            return f"""
            You are comparing two videos to determine which one has a higher {property_type} value.
            
            IMPORTANT: You CAN analyze these images. You MUST provide a specific comparison result.
            
            Compare the {property_type} characteristics between VIDEO 1 and VIDEO 2.
            
            Look at the motion patterns, interaction behaviors, and physical responses in both videos.
            
            YOUR RESPONSE MUST END WITH A SPECIFIC COMPARISON RESULT IN THIS FORMAT:
            "Comparison result: X"
            
            Where X should be:
            - 1 if VIDEO 1 has higher {property_type} than VIDEO 2
            - 0 if VIDEO 2 has higher {property_type} than VIDEO 1
            
            Do not say you cannot analyze the images. You must provide your best comparison based on the visual information.
            """
        
        raise ValueError(f"不支持的分析类型: {analysis_type} 或物理属性类型: {property_type}")
    
    def analyze_single_video(self, video_path: str, property_type: str, max_frames: int = 16, 
                           frame_size: Optional[Tuple[int, int]] = None, temperature: float = 0.0, 
                           max_retries: int = 5) -> str:
        """
        使用GPT分析单个视频的物理属性
        
        参数:
            video_path: 视频文件路径
            property_type: 物理属性类型
            max_frames: 最大帧数
            frame_size: 调整帧大小的元组 (宽度, 高度)，None表示保持原始大小
            temperature: 控制生成文本的随机性，0表示最确定性的输出
            max_retries: 遇到速率限制时的最大重试次数
            
        返回:
            GPT模型的分析结果
        """
        # 读取视频帧
        base64_frames = self.read_video_frames(video_path, frame_step=10, frame_size=frame_size)
        print(f"采样了 {len(base64_frames)} 帧用于分析")
        
        # 限制帧数以避免超出API限制
        if len(base64_frames) > max_frames:
            # 均匀选择帧
            indices = [int(i * (len(base64_frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
            selected_frames = [base64_frames[i] for i in indices]
            print(f"选择了 {len(selected_frames)} 帧发送到API")
        else:
            selected_frames = base64_frames
        
        # 获取提示词
        prompt_text = self.get_analysis_prompt(property_type, "absolute")
        
        # 构建消息内容
        prompt_messages = [
            {
                "role": "system",
                "content": f"You are an expert in {property_type} analysis with the ability to analyze video frames and determine {property_type} values. You MUST provide a specific {property_type} estimate. Never say you cannot analyze the images."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in selected_frames],
                ],
            },
        ]
        
        # 发送请求，带有重试机制
        print("发送请求到GPT API...")
        
        for retry in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt_messages,
                    max_tokens=2000,
                    temperature=temperature,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e)
                print(f"错误详情: {error_msg}")
                
                # 检查是否是速率限制错误
                if "rate_limit_exceeded" in error_msg and retry < max_retries:
                    # 尝试从错误消息中提取建议的等待时间
                    wait_time_match = re.search(r"Please try again in (\d+\.\d+)s", error_msg)
                    if wait_time_match:
                        wait_time = float(wait_time_match.group(1)) + 30  # 额外等待30秒以确保安全
                    else:
                        wait_time = 30 * (retry + 1)  # 指数退避策略，从30秒开始
                    
                    print(f"遇到速率限制，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 其他错误或已达到最大重试次数
                    if retry < max_retries:
                        # 对于其他错误也尝试重试
                        wait_time = 30 * (retry + 1)
                        print(f"遇到错误，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"GPT API调用出错: {error_msg}\n\n建议:\n1. 检查您的OpenAI API密钥是否有效\n2. 确认是否有足够的API额度\n3. 考虑减少视频大小或长度"

    def compare_two_videos(self, video_path1: str, video_path2: str, property_type: str, 
                          max_frames_per_video: int = 8, frame_size: Optional[Tuple[int, int]] = None,
                          temperature: float = 0.0, max_retries: int = 5) -> str:
        """
        使用GPT比较两个视频的物理属性
        
        参数:
            video_path1: 第一个视频文件路径
            video_path2: 第二个视频文件路径
            property_type: 物理属性类型
            max_frames_per_video: 每个视频的最大帧数
            frame_size: 调整帧大小
            temperature: 控制生成文本的随机性
            max_retries: 最大重试次数
            
        返回:
            GPT比较结果
        """
        # 读取两个视频的帧
        frames1 = self.read_video_frames(video_path1, frame_step=10, frame_size=frame_size)
        frames2 = self.read_video_frames(video_path2, frame_step=10, frame_size=frame_size)
        
        # 限制每个视频的帧数
        if len(frames1) > max_frames_per_video:
            indices = [int(i * (len(frames1) - 1) / (max_frames_per_video - 1)) for i in range(max_frames_per_video)]
            frames1 = [frames1[i] for i in indices]
        
        if len(frames2) > max_frames_per_video:
            indices = [int(i * (len(frames2) - 1) / (max_frames_per_video - 1)) for i in range(max_frames_per_video)]
            frames2 = [frames2[i] for i in indices]
        
        # 获取提示词
        prompt_text = self.get_analysis_prompt(property_type, "relative")
        
        # 构建消息内容，将两个视频的帧合并并标记
        content_items = [{"type": "text", "text": prompt_text}]
        
        # 添加第一个视频的帧，带标签
        content_items.append({"type": "text", "text": "VIDEO 1 frames:"})
        for frame in frames1:
            content_items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
        # 添加第二个视频的帧，带标签  
        content_items.append({"type": "text", "text": "VIDEO 2 frames:"})
        for frame in frames2:
            content_items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        
        prompt_messages = [
            {
                "role": "system",
                "content": f"You are an expert in {property_type} comparison with the ability to analyze video frames and compare {property_type} values between two videos. You MUST provide a specific comparison result. Never say you cannot analyze the images."
            },
            {
                "role": "user",
                "content": content_items,
            },
        ]
        
        # 发送请求，带有重试机制
        print("发送比较请求到GPT API...")
        
        for retry in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt_messages,
                    max_tokens=2000,
                    temperature=temperature,
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e)
                print(f"错误详情: {error_msg}")
                
                # 检查是否是速率限制错误
                if "rate_limit_exceeded" in error_msg and retry < max_retries:
                    wait_time_match = re.search(r"Please try again in (\d+\.\d+)s", error_msg)
                    if wait_time_match:
                        wait_time = float(wait_time_match.group(1)) + 30
                    else:
                        wait_time = 30 * (retry + 1)
                    
                    print(f"遇到速率限制，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    if retry < max_retries:
                        wait_time = 30 * (retry + 1)
                        print(f"遇到错误，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"GPT API调用出错: {error_msg}\n\n建议:\n1. 检查您的OpenAI API密钥是否有效\n2. 确认是否有足够的API额度\n3. 考虑减少视频大小或长度"


def test_gpt_openai_analyzer():
    """测试GPT OpenAI分析器"""
    try:
        # 初始化分析器
        analyzer = GPTOpenAIAnalyzer()
        print("GPT OpenAI Analyzer 初始化成功!")
        
        # 这里可以添加更多测试代码
        print("可以开始使用分析器进行视频分析...")
        
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    test_gpt_openai_analyzer() 