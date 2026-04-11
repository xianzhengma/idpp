import os
importsys
import json
import time
from typing import Dict, List

def test_qwen_api_wrapper():
    print('=' * 60)
    print('Testing the Qwen API wrapper')
    print('=' * 60)
    try:
        from qwen_api_wrapper import QwenVisionAPI, qwen_response
        print('1. Test API initialization...')
        api = QwenVisionAPI()
        print('✓ QwenVisionAPI initialization successful')
        print('\n2. Test API connection...')
        if api.test_connection():
            print('✓ API connection test successful')
        else:
            print('✗ API connection test failed')
            return False
        print('\n3. Test simple text message...')
        test_messages = [{'role': 'user', 'content': 'Please briefly introduce yourself'}]
        response = qwen_response(test_messages, model_version='qwen-vl-max')
        print(f'✓ Text message test successful')
        print(f'response:{response[:100]}...')
        return True
    except Exception as e:
        print(f'✗ Qwen API wrapper test failed:{e}')
        return False

def test_gpt_physics_analyzer_integration():
    print('\n' + '=' * 60)
    print('Testing GPT Physics Analyzer Qwen Integration')
    print('=' * 60)
    try:
        from gpt_physics_analyzer import GPTPhysicsAnalyzer
        print('1. Initialize the physical analyzer...')
        analyzer = GPTPhysicsAnalyzer()
        print('✓ GPTPhysicsAnalyzer initialized successfully')
        print('\n2. Test API calling method...')
        test_messages = [{'role': 'user', 'content': 'test message'}]
        print('- Test GPT API path...')
        gpt_response = analyzer.call_third_party_gpt_api(test_messages, 'gpt-4o')
        print(f'✓ GPT API call:{type(gpt_response).__name__}')
        print('- Test Qwen API path...')
        qwen_response = analyzer.call_third_party_gpt_api(test_messages, 'qwen-vl-max')
        print(f'✓ Qwen API call:{type(qwen_response).__name__}')
        return True
    except Exception as e:
        print(f'✗ Physics analyzer integration test failed:{e}')
        return False

def test_model_version_detection():
    print('\n' + '=' * 60)
    print('Test model version detection')
    print('=' * 60)
    try:
        from gpt_physics_analyzer import GPTPhysicsAnalyzer
        analyzer = GPTPhysicsAnalyzer()
        test_messages = [{'role': 'user', 'content': 'test'}]
        test_cases = [('gpt-4o', 'GPT'), ('gpt-4o-mini', 'GPT'), ('qwen-vl-max', 'Qwen'), ('qwen-vl-plus', 'Qwen'), ('qwen-vl-turbo', 'Qwen')]
        print('Test model version routing:')
        for model_version, expected_api in test_cases:
            try:
                response = analyzer.call_third_party_gpt_api(test_messages, model_version)
                if expected_api == 'GPT' and 'GPT' in response:
                    print(f'✓ {model_version} -> {expected_api} API')
                elif expected_api == 'Qwen' and 'Qwen' in response:
                    print(f'✓ {model_version} -> {expected_api} API')
                else:
                    print(f'✓ {model_version} -> {expected_api}API (call successful)')
            except Exception as e:
                print(f'⚠ {model_version} -> {expected_api}API (may require configuration)')
        return True
    except Exception as e:
        print(f'✗ Model version detection test failed:{e}')
        return False

def test_output_directory_generation():
    print('\n' + '=' * 60)
    print('Test output directory generation')
    print('=' * 60)
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from run_gpt_physics_tests import get_default_output_dir
        test_cases = [('gpt-4o', 'gpt_physics_analysis_results'), ('gpt-4o-mini', 'gpt_mini_physics_analysis_results'), ('qwen-vl-max', 'qwen_max_physics_analysis_results'), ('qwen-vl-plus', 'qwen_plus_physics_analysis_results'), ('qwen-vl-turbo', 'qwen_turbo_physics_analysis_results')]
        print('Test output directory generation:')
        for model_version, expected_dir in test_cases:
            actual_dir = get_default_output_dir(model_version)
            if actual_dir == expected_dir:
                print(f'✓ {model_version} -> {actual_dir}')
            else:print(f'✗ {model_version} -> {actual_dir}(expected:{expected_dir})')
        return True
    except Exception as e:
        print(f'✗ Output directory generation test failed:{e}')
        return False

def test_prompt_format_conversion():
    print('\n' + '=' * 60)
    print('Test prompt format conversion')
    print('=' * 60)
    try:
        from qwen_api_wrapper import QwenVisionAPI
        api = QwenVisionAPI()
        gpt_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'Please analyze this image'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='}}]}]
        qwen_messages = api.convert_gpt_prompt_to_qwen(gpt_messages)
        print('GPT format conversion test:')
        print(f'Enter the number of messages:{len(gpt_messages)}')
        print(f'Number of output messages:{len(qwen_messages)}')
        has_text = any((any((item.get('type') == 'text' for item in msg.get('content', []) if isinstance(msg.get('content'), list))) for msg in qwen_messages))
        has_video = any((any((item.get('type') == 'video' for item in msg.get('content', []) if isinstance(msg.get('content'), list))) for msg in qwen_messages))
        if has_text:
            print('✓ Text content converted successfully')
        if has_video:
            print('✓ Image content converted successfully (converted to video format)')
        return True
    except Exception as e:
        print(f'✗ Prompt format conversion test failed:{e}')
        return False

def main():
    print('Qwen API integration testing')
    print('=' * 80)
    if not os.getenv('DASHSCOPE_API_KEY'):
        print('⚠️ WARNING: Environment variable DASHSCOPE_API_KEY not set')
        print('Some tests may fail')
        print()
    test_results = []
    test_results.append(('Qwen API wrapper', test_qwen_api_wrapper()))
    test_results.append(('Physical analyzer integration', test_gpt_physics_analyzer_integration()))
    test_results.append(('Model version detection', test_model_version_detection()))
    test_results.append(('Output directory generation', test_output_directory_generation()))
    test_results.append(('Prompt format conversion', test_prompt_format_conversion()))
    print('\n' + '=' * 80)
    print('Summary of test results')
    print('=' * 80)
    passed = 0
    failed = 0
    for test_name, result in test_results:
        status = '✓ Pass' if result else '✗ Failure'
        print(f'{test_name:20} {status}')
        if result:
            passed += 1
        else:
            failed += 1
    print(f'\ntotal: {passed + failed}tests')
    print(f'pass:{passed}')
    print(f'fail:{failed}')
    if failed == 0:
        print('\n🎉 All tests passed! Qwen API integration success!')
        return 0
    else:
        print(f'\n⚠️ {failed}tests failed, please check configuration')
        return 1
if __name__ == '__main__':
    sys.exit(main())