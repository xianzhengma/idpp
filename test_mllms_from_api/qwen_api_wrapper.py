import os
import time
import json
from typing import Dict, List, Optional
from openai import OpenAI

class QwenVisionAPI:

    def __init__(self, api_key: Optional[str]=None, base_url: Optional[str]=None):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY') or ''
        if not self.api_key:
            raise ValueError('Please set DASHSCOPE_API_KEY or pass in api_key')
        self.base_url = base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        if not self.api_key:
            raise ValueError('The API key is not set. Please set the environment variable DASHSCOPE_API_KEY or directly pass in the api_key parameter.')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.supported_models = ['qwen-vl-max', 'qwen-vl-plus', 'qwen-vl-turbo']

    def convert_gpt_prompt_to_qwen(self, prompt_messages: List[Dict]) -> List[Dict]:
        qwen_messages = []
        for message in prompt_messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if isinstance(content, str):
                qwen_messages.append({'role': role, 'content': content})
            elif isinstance(content, list):
                qwen_content = []
                image_urls = []
                for item in content:
                    if item.get('type') == 'text':
                        qwen_content.append({'type': 'text', 'text': item.get('text', '')})
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {}).get('url', '')
                        if image_url:
                            image_urls.append(image_url)
                if image_urls:
                    qwen_content.append({'type': 'video', 'video': image_urls})
                qwen_messages.append({'role': role, 'content': qwen_content})
        return qwen_messages

    def qwen_response(self, prompt_messages: List[Dict], model_version: str='qwen-vl-max', max_retries: int=5) -> str:
        if model_version not in self.supported_models:
            print(f'Warning: Model{model_version}Not in the support list, uses default model qwen-vl-max')
            model_version = 'qwen-vl-max'
        qwen_messages = self.convert_gpt_prompt_to_qwen(prompt_messages)
        for attempt in range(max_retries):
            try:
                print(f'Calling Qwen API... (try{attempt + 1}/{max_retries})')
                response = self.client.chat.completions.create(model=model_version, messages=qwen_messages, temperature=0.0, max_tokens=2000)
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    print(f'✓ Qwen API call successful')
                    return content
                else:
                    print(f'✗ Qwen API returns empty response')
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        print(f'wait{wait_time}Try again in seconds...')
                        time.sleep(wait_time)
                        continue
                    else:
                        return 'Qwen API call failed: returned empty response'
            except Exception as e:
                print(f'Try again{attempt + 1}/{max_retries}fail:{e}')
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f'wait{wait_time}Try again in seconds...')
                    time.sleep(wait_time)
                else:
                    return f'Qwen API call failed: All retries failed -{str(e)}'
        return 'Qwen API call failed: unknown error'

    def test_connection(self) -> bool:
        try:
            test_messages = [{'role': 'user', 'content': 'Hello, test connection'}]
            response = self.client.chat.completions.create(model='qwen-vl-max', messages=test_messages, max_tokens=10)
            return response.choices and len(response.choices) > 0
        except Exception as e:
            print(f'Connection test failed:{e}')
            return False

def qwen_response(prompt_messages: List[Dict], model_version: str='qwen-vl-max', max_retries: int=5) -> str:
    try:
        qwen_api = QwenVisionAPI()
        return qwen_api.qwen_response(prompt_messages, model_version, max_retries)
    except Exception as e:
        return f'Qwen API initialization failed:{str(e)}'

def test_qwen_api():
    try:
        qwen_api = QwenVisionAPI()
        print('Test API connection...')
        if qwen_api.test_connection():
            print('✓ API connection successful')
        else:
            print('✗ API connection failed')
            return
        test_prompt = [{'role': 'user', 'content': 'Please briefly introduce yourself'}]
        print('\nTest text prompt...')
        response = qwen_response(test_prompt)
        print(f'response:{response}')
    except Exception as e:
        print(f'Test failed:{e}')
if __name__ == '__main__':
    test_qwen_api()