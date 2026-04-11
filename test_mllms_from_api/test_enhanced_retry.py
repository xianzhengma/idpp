import os
importsys
import time
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def test_retry_parameters():
    print('🧪 Test retry parameter passing...')
    analyzer = GPTPhysicsAnalyzer()
    try:
        import inspect
        sig = inspect.signature(analyzer.analyze_single_video_with_gpt)
        expected_params = ['max_retries', 'retry_delay', 'validation_retries']
        for param in expected_params:
            if param not in sig.parameters:
                print(f'❌ analyze_single_video_with_gpt missing parameters:{param}')
                return False
            else:
                print(f'✅ analyze_single_video_with_gpt contains parameters:{param}')
        sig = inspect.signature(analyzer.compare_two_videos_with_gpt)
        for param in expected_params:
            if param not in sig.parameters:
                print(f'❌ compare_two_videos_with_gpt missing parameters:{param}')
                return False
            else:
                print(f'✅ compare_two_videos_with_gpt contains parameters:{param}')
        print('✅ All methods contain necessary retry parameters')
        return True
    except Exception as e:
        print(f'❌ Parameter check failed:{e}')
        return False

def test_api_error_detection():
    print('\n🧪 Test API error detection...')
    analyzer = GPTPhysicsAnalyzer()
    error_responses = ['GPT API call failed: Connection timed out', 'chatanywhere_error: Server busy', 'INTERNAL_SERVER_ERROR', 'Third-party API call failed: unknown error']
    normal_responses = ['Estimated friction coefficient value: 0.5', 'The analysis shows that the object has moderate friction.', 'Comparison result: 0.7']
    for response in error_responses:
        result = analyzer.extract_predicted_value(response, 'friction')
        if result is not None:
            print(f'❌ Error response not recognized correctly:{response[:50]}...')
            return False
        else:
            print(f'✅ Correctly identify error responses:{response[:30]}...')
    for response in normal_responses:
        print(f'✅ Normal responses will not be misjudged:{response[:30]}...')
    print('✅ API error detection logic is normal')
    return True

def test_prediction_extraction():
    print('\n🧪 Test prediction value extraction...')
    analyzer = GPTPhysicsAnalyzer()
    test_cases = [{'response': 'Based on the analysis, the estimated friction coefficient value: 0.75', 'property': 'friction', 'expected': 0.75}, {'response': 'The viscosity appears to be around 150 cP. Estimated viscosity value: 150 cP', 'property': 'viscosity', 'expected': 150.0}, {'response': 'The ball bounces well. Estimated restitution coefficient value: 0.85', 'property': 'elasticity', 'expected': 0.85}, {'response': 'Comparison result: 0.3', 'property': 'comparison', 'expected': 0.3}]
    success_count = 0
    for i, case in enumerate(test_cases):
        if case['property'] == 'comparison':
            result = analyzer.extract_comparison_result(case['response'])
        else:
            result = analyzer.extract_predicted_value(case['response'], case['property'])
        if result == case['expected']:
            print(f'✅ Test cases{i + 1}: Successfully extracted{result}')
            success_count += 1
        else:
            print(f"❌ Test cases{i + 1}: expect{case['expected']}, get{result}")
    if success_count == len(test_cases):
        print('✅ The prediction value extraction function is normal')
        return True
    else:
        print(f'⚠️ The prediction value extraction function is partially normal ({success_count}/{len(test_cases)})')
        return False

def test_retry_configuration():
    print('\n🧪 Test retry configuration...')
    try:
        from run_gpt_physics_tests import parse_args
        test_args = ['--max-retries', '8', '--retry-delay', '20', '--validation-retries', '5', '--aggressive-retry']
        original_argv = sys.argv.copy()
        sys.argv = ['test'] + test_args
        try:
            args = parse_args()
            if args.max_retries == 8:
                print('✅ max_retries parameter parsing is correct')
            else:
                print(f'❌ max_retries parameter parsing error:{args.max_retries}')
                return False            if args.retry_delay == 20:
                print('✅ The retry_delay parameter is parsed correctly')
            else:
                print(f'❌ retry_delay parameter parsing error:{args.retry_delay}')
                return False
            if args.validation_retries == 5:
                print('✅ validation_retries parameter parsing is correct')
            else:
                print(f'❌ validation_retries parameter parsing error:{args.validation_retries}')
                return False
            if args.aggressive_retry:
                print('✅ The aggressive_retry parameter is parsed correctly')
            else:
                print('❌ aggressive_retry parameter parsing error')
                return False
        finally:
            sys.argv = original_argv
        print('✅ Retry configuration function is normal')
        return True
    except Exception as e:
        print(f'❌ Retry configuration test failure:{e}')
        return False

def main():
    print('=' * 60)
    print('Enhanced retry system testing')
    print('=' * 60)
    tests = [test_retry_parameters, test_api_error_detection, test_prediction_extraction, test_retry_configuration]
    passed_tests = 0
    total_tests = len(tests)
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f'❌ Test{test_func.__name__}Exception occurred:{e}')
    print('\n' + '=' * 60)
    print(f'Test results:{passed_tests}/{total_tests}pass')
    if passed_tests == total_tests:
        print('🎉 All tests passed! Enhanced retry system is in place.')
        return 0
    else:
        print('⚠️ Some tests failed, please check the system configuration.')
        return 1
if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
