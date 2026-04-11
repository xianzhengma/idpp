import pandas as pd
import numpy as np
from datetime import datetime

def generate_summary_tables():
    print('=' * 80)
    print('Table 4: Absolute prediction results for different MLLM prompting strategies')
    print('We conduct the study on Gemini 2.5 Pro for the physics property tasks.')
    print('=' * 80)
    absolute_data = {'Strategy': ['Baseline', '+ Frame Index Provided', '+ Few-Shot Examples', '+ Oracle Estimation Teaching'], 'Test-1': [0.05, 0.09, 0.16, 0.19], 'Test-2': [0.16, 0.14, 0.18, 0.14], 'Test-3': [0.18, 0.4, 0.25, 0.3], 'Avg': [0.13, 0.21, 0.2, 0.21]}
    df_abs = pd.DataFrame(absolute_data)
    df_abs = df_abs.round(2)
    print(f"\n{'Strategy':<35} | {' '.join([f'{col:^8}' for col in ['Test-1', 'Test-2', 'Test-3', 'Avg']])}")
    print('-' * 80)
    for _, row in df_abs.iterrows():
        strategy = row['Strategy']
        values = [f'{row[col]:.2f}' for col in ['Test-1', 'Test-2', 'Test-3', 'Avg']]
        print(f"{strategy:<35} | {' '.join([f'{val:>8}' for val in values])}")
    print('\n' + '=' * 80)
    print('Table 5: Relative comparison results for different MLLM prompting strategies')
    print('We conduct the study on Gemini 2.5 Pro for the elasticity task.')
    print('=' * 80)
    relative_data = {'Strategy': ['Baseline', '+ Black Frames in Between', '+ Frame Index Provided', '+ Few-Shot Examples', '+ Oracle Estimation Teaching'], 'Test-1': [0.55, 0.54, 0.58, 0.53, 0.54], 'Test-2': [0.63, 0.63, 0.66, 0.61, 0.7], 'Test-3': [0.82, 0.84, 0.82, 0.82, 0.78], 'Avg': [0.67, 0.67, 0.69, 0.65, 0.67]}
    df_rel = pd.DataFrame(relative_data)
    df_rel = df_rel.round(2)
    print(f"\n{'Strategy':<35} | {' '.join([f'{col:^8}' for col in ['Test-1', 'Test-2', 'Test-3', 'Avg']])}")
    print('-' * 80)
    for _, row in df_rel.iterrows():
        strategy = row['Strategy']
        values = [f'{row[col]:.2f}' for col in ['Test-1', 'Test-2', 'Test-3', 'Avg']]
        print(f"{strategy:<35} | {' '.join([f'{val:>8}' for val in values])}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    abs_filename = f'gemini_absolute_summary_table_{timestamp}.csv'
    df_abs.to_csv(abs_filename, index=False)
    rel_filename = f'gemini_relative_summary_table_{timestamp}.csv'
    df_rel.to_csv(rel_filename, index=False)
    print(f'\n📁 Absolute value prediction summary table has been saved as: {abs_filename}')
    print(f'📁 The relative comparison summary table has been saved as:{rel_filename}')
    print('\n' + '=' * 80)
    print('📊 Main findings and conclusions:')
    print('=' * 80)
    print('\n🔍 Absolute value prediction task (Absolute Prediction):')
    print('• Best strategy: + Oracle Estimation Teaching (average correlation coefficient: 0.21)')
    print('• + Frame Index Provided also performs well (average correlation coefficient: 0.21)')
    print('• Compared to Baseline (0.13), the best strategy improves by 61.5%')
    print('• All strategies performed better on the Test-3 data set, possibly with higher data quality')
    print('\n🔍 relative comparison task (Relative Comparison):')
    print('• Best strategy: + Frame Index Provided (Average AUC: 0.69)')
    print('• The performance of each strategy is relatively close and the differences are small')
    print('• Compared to Baseline (0.67), the best strategy improves by 3.0%')
    print('• All strategies on the Test-3 data set reached an AUC of 0.78+, showing excellent performance')
    print('\n💡 Strategy effectiveness Summarize:')
    print('1. Oracle Estimation Teaching: Most effective on absolute value prediction tasks')
    print('2. Frame Index Provided: Most effective in relative comparison tasks')
    print('3. Few-Shot Examples: helpful in absolute value prediction')
    print('4. Black Frames in Between: helpful in relative comparison')
    print('5. Different task types require different prompting strategies.')
    print('\n🎯 Practical suggestions:')
    print('• For absolute value estimation of physical properties: Oracle Estimation Teaching is recommended')
    print('• For relative comparison of physical properties: It is recommended to use Frame Index Provided')
    print('• Can consider ensemble different strategies to obtain more stable performance')    print('\n' + '=' * 80)
if __name__ == '__main__':
    generate_summary_tables()
