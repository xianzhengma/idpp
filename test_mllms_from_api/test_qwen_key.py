import os
from openai import OpenAI

def test_qwen_api_key():
    api_key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not api_key:
        print('Please set the environment variable DASHSCOPE_API_KEY')
        return False
    print('Test Qwen API key...')
    print('API key: (from environment variables, hidden)')
    print(f'API base URL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
    try:
        client = OpenAI(api_key=api_key, base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
        print('\nSend test request...')
        response = client.chat.completions.create(model='qwen-vl-max', messages=[{'role': 'user', 'content': 'Hello, please respond in Chinese.'}], max_tokens=50)
        print('✓ API test successful!')
        print(f'response:{response.choices[0].message.content}')
        return True
    except Exception as e:
        print(f'✗ API test failed:{e}')
        if '401' in str(e) or 'invalid_api_key' in str(e):
            print('\nPossible reasons:')
            print('1. This API key is not the key of Alibaba Cloud DashScope')
            print('2. API key format is incorrect')
            print('3. API key has expired or been disabled')
            print('\nSolution:')
            print('1. Please go to Alibaba Cloud DashScope console to obtain the correct API key.')
            print('Address: https://dashscope.console.aliyun.com/')
            print('2. Confirm that your API key is the key for the DashScope service')
        elif '403' in str(e):
            print('\nPossible reasons:')
            print('1. The API key does not have permission to access the qwen-vl-max model')
            print('2. Insufficient account balance')
        else:
            print(f'\nOther mistake:{e}')
        return False

def test_alternative_endpoints():
    api_key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not api_key:
        print('Please set DASHSCOPE_API_KEY')
        return
    endpoints = ['https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'https://dashscope-intl.aliyuncs.com/api/v1', 'https://dashscope.aliyuncs.com/api/v1']
    print('\nTesting different API endpoints...')
    for endpoint in endpoints:
        print(f'\nTry endpoint: {endpoint}')
        try:
            client = OpenAI(api_key=api_key, base_url=endpoint)
            response = client.chat.completions.create(model='qwen-vl-max', messages=[{'role': 'user', 'content': 'Test'}], max_tokens=10)
            print(f'✓ Endpoint{endpoint}working fine')
            return True
        except Exception as e:
            print(f'✗ Endpoint{endpoint}fail:{str(e)[:100]}...')
    return False
if __name__ == '__main__':
    print('=' * 60)
    print('Qwen API key test')
    print('=' * 60)
    success = test_qwen_api_key()
    if not success:
        success = test_alternative_endpoints()
    if success:
        print('\n🎉 API key test success, can be used!')
    else:
        print('\n❌ All tests failed')
        print('\nSuggestion:')
        print('1. Confirm that your API key is obtained from Alibaba Cloud DashScope')
        print('2. Check account status and balance')
        print('3. Try to regenerate API key')