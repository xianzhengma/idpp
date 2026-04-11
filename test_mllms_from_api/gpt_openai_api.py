import os
import time
import re
import base64
import cv2
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional, Tuple

class GPTOpenAIAnalyzer:

    def __init__(self, api_key: Optional[str]=None, model: str='gpt-4o'):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('The OPENAI_API_KEY environment variable is not set or the api_key parameter is not provided')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def read_video_frames(self, video_path: str, frame_step: int=5, frame_size: Optional[Tuple[int, int]]=None) -> List[str]:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f'Unable to open video file:{video_path}')
        base64_frames = []
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_count % frame_step == 0:
                if frame_size is not None:
                    frame = cv2.resize(frame, frame_size)
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
            frame_count += 1
        video.release()
        if frame_size:
            print(f'total{frame_count}Frame, obtained after sampling{len(base64_frames)}frame, resized to{frame_size}。')
        else:
            print(f'total{frame_count}Frame, obtained after sampling{len(base64_frames)}Frame, keeping original size.')
        return base64_frames

    def get_analysis_prompt(self, property_type: str, analysis_type: str) -> str:
        if analysis_type == 'absolute':
            if property_type == 'friction':
                return '\n                You are analyzing frames from a video showing objects with friction interaction. Your task is to estimate the friction coefficient.\n                \n                IMPORTANT: You CAN analyze these images. You MUST provide a specific friction coefficient estimate.\n                \n                For reference:\n                - Ice on ice: 0.02-0.03\n                - Steel on steel: 0.4-0.6\n                - Rubber on concrete: 0.6-0.85\n                - Wood on wood: 0.3-0.5\n                \n                Analyze the sliding motion, deceleration rate, and surface interaction.\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC FRICTION COEFFICIENT VALUE IN THIS FORMAT:\n                "Estimated friction coefficient value: X"\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                '
            elif property_type == 'viscosity':
                return '\n                You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.\n                \n                IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.\n                \n                For reference:\n                - Water: 1 cP\n                - Vegetable oil: 100 cP\n                - Motor oil: 300 cP\n                - Honey: 5000 cP\n                \n                Analyze the liquid\'s flow speed, surface waves, and interaction with objects.\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:\n                "Estimated viscosity value: X cP"\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                '
            elif property_type == 'elasticity':
                return '\n                Please analyze these frames from a video of a bouncing object and estimate its restitution coefficient.\n                \n                The restitution coefficient is the ratio of the velocity after collision to the velocity before collision.\n                \n                Please:\n                1. Track the object\'s position in each frame\n                2. Determine its velocity before and after impact\n                3. Calculate the restitution coefficient\n                4. Provide a detailed analysis of your methodology\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:\n                "Estimated restitution coefficient value: X"\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                '
        else:
            return f'\n            You are comparing two videos to determine which one has a higher {property_type} value.\n            \n            IMPORTANT: You CAN analyze these images. You MUST provide a specific comparison result.\n            \n            Compare the {property_type} characteristics between VIDEO 1 and VIDEO 2.\n            \n            Look at the motion patterns, interaction behaviors, and physical responses in both videos.\n            \n            YOUR RESPONSE MUST END WITH A SPECIFIC COMPARISON RESULT IN THIS FORMAT:\n            "Comparison result: X"\n            \n            Where X should be:\n            - 1 if VIDEO 1 has higher {property_type} than VIDEO 2\n            - 0 if VIDEO 2 has higher {property_type} than VIDEO 1\n            \n            Do not say you cannot analyze the images. You must provide your best comparison based on the visual information.\n            '
        raise ValueError(f'Unsupported analysis types:{analysis_type}or physical property type:{property_type}')

    def analyze_single_video(self, video_path: str, property_type: str, max_frames: int=16, frame_size: Optional[Tuple[int, int]]=None, temperature: float=0.0, max_retries: int=5) -> str:
        base64_frames = self.read_video_frames(video_path, frame_step=10, frame_size=frame_size)
        print(f'Sampled{len(base64_frames)}frames used for analysis')
        if len(base64_frames) > max_frames:
            indices = [int(i * (len(base64_frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
            selected_frames = [base64_frames[i] for i in indices]
            print(f'selected{len(selected_frames)}Frame sent to API')
        else:
            selected_frames = base64_frames
        prompt_text = self.get_analysis_prompt(property_type, 'absolute')
        prompt_messages = [{'role': 'system', 'content': f'You are an expert in {property_type} analysis with the ability to analyze video frames and determine {property_type} values. You MUST provide a specific {property_type} estimate. Never say you cannot analyze the images.'}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}, *[{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}} for frame in selected_frames]]}]
        print('Send request to GPT API...')
        for retry in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=prompt_messages, max_tokens=2000, temperature=temperature)
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                print(f'Error details:{error_msg}')
                if 'rate_limit_exceeded' in error_msg and retry < max_retries:
                    wait_time_match = re.search('Please try again in (\\d+\\.\\d+)s', error_msg)
                    if wait_time_match:
                        wait_time = float(wait_time_match.group(1)) + 30
                    else:
                        wait_time = 30 * (retry + 1)
                    print(f'Rate limit encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                    time.sleep(wait_time)
continue
                elif retry < max_retries:
                    wait_time = 30 * (retry + 1)
                    print(f'Error encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
                else:
                    return f'GPT API call error:{error_msg}\n\nSuggestions:\n1. Check that your OpenAI API key yesno is valid\n2. Confirm yesno has sufficient API credit\n3. Consider reducing the video size or length'

    def compare_two_videos(self, video_path1: str, video_path2: str, property_type: str, max_frames_per_video: int=8, frame_size: Optional[Tuple[int, int]]=None, temperature: float=0.0, max_retries: int=5) -> str:
        frames1 = self.read_video_frames(video_path1, frame_step=10, frame_size=frame_size)
        frames2 = self.read_video_frames(video_path2, frame_step=10, frame_size=frame_size)
        if len(frames1) > max_frames_per_video:
            indices = [int(i * (len(frames1) - 1) / (max_frames_per_video - 1)) for i in range(max_frames_per_video)]
            frames1 = [frames1[i] for i in indices]
        if len(frames2) > max_frames_per_video:
            indices = [int(i * (len(frames2) - 1) / (max_frames_per_video - 1)) for i in range(max_frames_per_video)]
            frames2 = [frames2[i] for i in indices]
        prompt_text = self.get_analysis_prompt(property_type, 'relative')
        content_items = [{'type': 'text', 'text': prompt_text}]
        content_items.append({'type': 'text', 'text': 'VIDEO 1 frames:'})
        for frame in frames1:
            content_items.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        content_items.append({'type': 'text', 'text': 'VIDEO 2 frames:'})
        for frame in frames2:
            content_items.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        prompt_messages = [{'role': 'system', 'content': f'You are an expert in {property_type} comparison with the ability to analyze video frames and compare {property_type} values between two videos. You MUST provide a specific comparison result. Never say you cannot analyze the images.'}, {'role': 'user', 'content': content_items}]
        print('Send comparison request to GPT API...')
        for retry in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=prompt_messages, max_tokens=2000, temperature=temperature)
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                print(f'Error details:{error_msg}')
                if 'rate_limit_exceeded' in error_msg and retry < max_retries:
                    wait_time_match = re.search('Please try again in (\\d+\\.\\d+)s', error_msg)
                    if wait_time_match:
                        wait_time = float(wait_time_match.group(1)) + 30
                    else:
                        wait_time = 30 * (retry + 1)
                    print(f'Rate limit encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
                elif retry < max_retries:
                    wait_time = 30 * (retry + 1)
                    print(f'Error encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
                else:
                    return f'GPT API call error:{error_msg}\n\nSuggestions:\n1. Check that your OpenAI API key yesno is valid\n2. Confirm yesno has sufficient API credit\n3. Consider reducing the video size or length'

def test_gpt_openai_analyzer():
    try:
        analyzer = GPTOpenAIAnalyzer()
        print('GPT OpenAI Analyzer initialized successfully!')
        print('You can start using the analyzer for video analysis...')
    except Exception as e:
        print(f'Initialization failed:{e}')
if __name__ == '__main__':
    test_gpt_openai_analyzer()