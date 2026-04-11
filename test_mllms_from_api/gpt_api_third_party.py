import json
import requests
import base64
import time
import os

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_response(prompt, model_version=None, max_retries=5):
    key_path = os.environ.get('GPT_KEY_JSON', os.path.join(os.environ.get('SVD_ROOT', '.'), 'secrets', 'gpt_key.json'))
    for idx in range(max_retries):
        try:
            with open(key_path, 'r', encoding='utf-8') as file:
                gpt_key = json.load(file)
            key = gpt_key['key']
            model_version = gpt_key['version'] if model_version == None else model_version
            ip_address = gpt_key['ip_address']
            header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
            post_dict = {'model': model_version, 'messages': prompt}
            r = requests.post(ip_address, json=post_dict, headers=header)
            json_r = r.json()
            message = json_r['choices'][0]['message']['content']
            return message
        except Exception as e:
            print(f'Try again{idx + 1}/{max_retries}fail:{e}')
            if hasattr(locals(), 'r'):
                print(f'HTTP status code:{r.status_code}')
            if hasattr(locals(), 'json_r'):
                print(f'Response content:{json_r}')
            if idx < max_retries - 1:
                wait_time = 30 * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
    return 'GPT API call failed: All retries failed, please check the API service status'

def gptv_response(prompt, model_version=None, max_retries=8, base_delay=30):
    for idx in range(max_retries):
        try:
            key = os.environ.get('OPENAI_API_KEY', '').strip()
            if not key:
                return 'GPT Vision API call failed: Please set the environment variable OPENAI_API_KEY'
            print('Use environment variable OPENAI_API_KEY (full content hidden)')
            model_version = 'gpt-4o' if model_version == None else model_version
            ip_address = 'https://api.chatanywhere.tech/v1/chat/completions'
            header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
            post_dict = {'model': model_version, 'messages': prompt, 'max_tokens': 2000, 'temperature': 0.0}
            r = requests.post(ip_address, json=post_dict, headers=header, timeout=(30, 120))
            print(f'<Response [{r.status_code}]>')
            if r.status_code == 200:
                json_r = r.json()
                if 'choices' in json_r and len(json_r['choices']) > 0:
                    message = json_r['choices'][0]['message']['content']
                    print(f'✓ API call successful')
                    return message
                else:
                    print(f'Response format error:{json_r}')
                    if idx < max_retries - 1:
                        wait_time = base_delay * 2 ** min(idx, 4)
                        print(f'wait{wait_time}Try again in seconds ({idx + 1}/{max_retries})...')
                        time.sleep(wait_time)
                        continue
            elif r.status_code == 429:
                print(f'Rate limit encountered (429)')
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after) + 10
                else:
                    wait_time = base_delay * 2 ** min(idx, 4)
                if idx < max_retries - 1:
                    print(f'wait{wait_time}Try again in seconds ({idx + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
            elif r.status_code in [500, 502, 503, 504]:
                print(f'Server error:{r.status_code}')
                if idx < max_retries - 1:
                    wait_time = base_delay * 2 ** min(idx, 3)
                    print(f'wait{wait_time}Try again in seconds ({idx + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
            else:
                print(f'HTTP error:{r.status_code}')
                print(f'Response content:{r.text[:200]}...')
                if idx < max_retries - 1:
                    wait_time = base_delay * (idx + 1)
                    print(f'wait{wait_time}Try again in seconds ({idx + 1}/{max_retries})...')
                    time.sleep(wait_time)
                    continue
        except requests.exceptions.Timeout as e:
            print(f'Request timeout ({idx + 1}/{max_retries}): {e}')
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
                continue
        except requests.exceptions.ConnectionError as e:
            print(f'Connection error ({idx + 1}/{max_retries}): {e}')
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
                continue
        except requests.exceptions.RequestException as e:
            print(f'Request exception ({idx + 1}/{max_retries}): {e}')
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
                continue
        except json.JSONDecodeError as e:
            print(f'JSON parsing error ({idx + 1}/{max_retries}): {e}')
            if 'r' in locals():
                print(f'Original response:{r.text[:200]}...')
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
                continue
        except Exception as e:
            print(f'Unknown error ({idx + 1}/{max_retries}): {e}')
            if 'r' in locals():
                print(f'HTTP status code:{r.status_code}')
                print(f'Response content:{r.text[:200]}...')
            if 'json_r' in locals():
                print(f'JSON response:{json_r}')
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f'wait{wait_time}Try again in seconds...')
                time.sleep(wait_time)
                continue
    return 'GPT Vision API call failed: All retries failed, please check the API service status or try again later'

def test_gptv_with_image():
    try:
        image_path = os.environ.get('GPT_SAMPLE_IMAGE', 'cover.png')
        base64_image = encode_image(image_path)
        print('Image encoded successfully.')
        prompt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Please describe what this image is about.'}, {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}]}]
        print('Sending request to GPT-V model...')
        response = gptv_response(prompt)
        print('Response from GPT-V model:')
        print(response)
    except Exception as e:
        print(f'An error occurred: {e}')
if __name__ == '__main__':
    test_gptv_with_image()
