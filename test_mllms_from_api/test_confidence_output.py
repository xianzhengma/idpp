import re
importsys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpt_physics_analyzer import GPTPhysicsAnalyzer

def test_confidence_extraction():
    print('=' * 60)
    print('Test the confidence score extraction function')
    print('=' * 60)
    analyzer = GPTPhysicsAnalyzer()
    test_cases = [{'response': 'After analyzing both videos carefully, I observe that VIDEO 1 shows higher friction than VIDEO 2. Comparison result: 0.8', 'expected': 0.8, 'description': 'Standard Confidence Format - High Confidence Biased VIDEO 1'}, {'response': 'The motion in VIDEO 2 appears to have lower friction compared to VIDEO 1. Comparison result: 0.2', 'expected': 0.2, 'description': 'Standard Confidence Format - High Confidence Biased VIDEO 2'}, {'response': "Both videos show similar friction levels, it's difficult to determine a clear winner. Comparison result: 0.5", 'expected': 0.5, 'description': 'Standard Confidence Format - Uncertain/Equal'}, {'response': 'Analyzing the videos, I believe VIDEO 1 has quite high friction. Comparison result: 0.9', 'expected': 0.9, 'description': 'Standard Confidence Format - High Confidence Favors VIDEO 1'}, {'response': 'VIDEO 2 clearly shows much lower friction than VIDEO 1. Comparison result: 0.1', 'expected': 0.1, 'description': 'Standard Confidence Format - High Confidence Favors VIDEO 2'}, {'response': 'I can see that VIDEO 1 has higher friction levels than VIDEO 2.', 'expected': 0.8, 'description': 'Alternate Extraction - Inference from Text Biased VIDEO 1'}, {'response': 'The analysis shows that VIDEO 2 has higher friction properties.', 'expected': 0.2, 'description': 'Alternate extraction - inference from text biased VIDEO 2'}, {'response': 'Both videos appear to have equal or similar friction characteristics.', 'expected': 0.5, 'description': 'Alternate extraction - inferred from text equal keyword'}]
    success_count = 0
    total_count = len(test_cases)
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {case['description']}")
        print(f"Enter response:{case['response']}")
        extracted_confidence = analyzer.extract_comparison_result(case['response'])
        print(f"Expected confidence:{case['expected']}")
        print(f'Extract confidence:{extracted_confidence}')
        if extracted_confidence is not None and abs(extracted_confidence - case['expected']) < 0.01:
            print('✓ Test passed')
            success_count += 1
        elif extracted_confidence == case['expected']:
            print('✓ Test passed')
            success_count += 1
        else:
            print('✗ Test failed')
    print('\n' + '=' * 60)
    print(f'Test results:{success_count}/{total_count}pass')
    print('=' * 60)
    return success_count == total_count

def test_prompt_generation():
    print('\n' + '=' * 60)
    print('Test the new prompt word generation feature')
    print('=' * 60)
    analyzer = GPTPhysicsAnalyzer()
    prompt = analyzer.get_analysis_prompt('friction', 'relative')
    print('Relative comparison prompt word (friction):')
    print('-' * 40)
    print(prompt)
    print('-' * 40)
    key_phrases = ['confidence score between 0 and 1', 'Comparison result: X', 'X = 1.0: Very confident that VIDEO 1', 'X = 0.0: Very confident that VIDEO 2', 'closer to 1.0']
    missing_phrases = []
    for phrase in key_phrases:
        if phrase not in prompt:
            missing_phrases.append(phrase)
    if not missing_phrases:
        print('✓ Prompt words include all key phrases')
        return True
    else:
        print('✗ The prompt word is missing the following key phrases:')
        for phrase in missing_phrases:
            print(f' - {phrase}')
        return False

def main():
    print('Confidence output function test')
    print('=' * 80)
    try:
        extraction_passed = test_confidence_extraction()
        prompt_passed = test_prompt_generation()
        print('\n' + '=' * 80)
        print('Test summary:')
        print(f"Confidence extraction function:{('✓ Pass' if extraction_passed else '✗ Failure')}")
        print(f"Prompt word generation function:{('✓ Pass' if prompt_passed else '✗ Failure')}")if extraction_passed and prompt_passed:
            print('\n🎉 All tests passed! The confidence output function has been successfully implemented')
            print('\nNow the relative test will output a confidence value between 0-1, which can be used to calculate the AUC score')
            print('- 0.0-0.5: Favor VIDEO 2 with higher physical attribute values')
            print('- 0.5-1.0: Favor VIDEO 1 with higher physical attribute values')
            print('- The closer it is to 0 or 1, the higher the confidence level')
            return 0
        else:
            print('\n❌ Partial test fails')
            return 1
    except Exception as e:
        print(f'\n❌ Mistake occurred during the test: {e}')
        return 1
if __name__ == '__main__':
    sys.exit(main())