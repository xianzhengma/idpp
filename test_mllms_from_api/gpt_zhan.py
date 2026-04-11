import json
import os
import requests
from tqdm import tqdm
filename = 'flickr_test_captions_list_5.10.json'
with open(filename, 'r', encoding='utf-8') as handle:
    data = json.load(handle)

def gpt_response(prompt, model_version=None):
    proxy_key = os.environ.get('GPT_PROXY_KEY', '')
    proxy_url = os.environ.get('GPT_PROXY_URL', 'https://api.chatanywhere.tech/v1/chat/completions')
    if not proxy_key:
        raise RuntimeError('Please set the environment variable GPT_PROXY_KEY')
    for idx in range(99999):
        try:
            model_version = model_version or 'gpt-4o'
            header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {proxy_key}'}
            post_dict = {'model': model_version, 'messages': prompt, 'n': 5, 'temperature': 1}
            response = requests.post(proxy_url, json=post_dict, headers=header, timeout=120)
            json_r = response.json()['choices']
            return json_r
        except (requests.RequestException, KeyError, ValueError) as exc:
            print(f'Try again{idx + 1}: {exc}')
            if idx > 100:
                raise
if __name__ == '__main__':
    print('Configure GPT_PROXY_KEY and data file before running.')
