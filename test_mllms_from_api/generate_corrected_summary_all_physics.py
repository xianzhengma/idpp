import pandas as pd
import numpy as np
from datetime import datetime

def generate_corrected_comprehensive_summary():
    print('=' * 100)
    print('🔬 GEMINI 2.5 PRO physical property prediction ablation experiment - revised comprehensive analysis report')
    print('⚠️ Note: Data quality filtering with GT value <0.1 has been applied to Elasticity')
    print('=' * 100)
    print('\n📊 Experimental settings:')
    print('• Model: Gemini 2.5 Pro')
    print('• Physical properties: Friction, Viscosity, Elasticity')
    print('• Task type: Absolute Prediction & Relative Comparison')
    print('• Number of samples: 100 samples per experiment (divided into 5 parts, 20 samples each)')
    print('• Test set: Test-1, Test-2, Test-3')
    print('• 🔧 Data quality control: Samples with Elasticity GT value <0.1 have been filtered')
    print(f"\n{'=' * 100}")
    print('Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties (Corrected)')
    print('=' * 100)
    absolute_results = {'Elasticity': {'Baseline': [-0.03, 0.26, 0.06, 0.1], '+ Frame Index Provided': [0.06, 0.35, 0.55, 0.32], '+ Few-Shot Examples': [0.39, 0.34, 0.24, 0.33], '+ Oracle Estimation Teaching': [0.19, 0.14, 0.24, 0.19]}, 'Friction': {'Baseline': [0.06, 0.09, 0.19, 0.11], '+ Frame Index Provided': [0.11, 0.05, -0.02, 0.05], '+ Few-Shot Examples': [-0.07, 0.11, 0.12, 0.05], '+ Oracle Estimation Teaching': [0.06, 0.17, 0.03, 0.09]}, 'Visibility': {'Baseline': [0.05, 0.12, 0.35, 0.17], '+ Frame Index Provided': [0.01, 0.02, 0.37, 0.13], '+ Few-Shot Examples': [0.07, -0.05, 0.29, 0.1], '+ Oracle Estimation Teaching': [0.15, 0.04, 0.29, 0.16]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in absolute_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            if property_name == 'Elasticity' and first_row:
                prop_display += ' 🔧'
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f'\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties (Corrected)')
    print('=' * 100)
    relative_results = {'Elasticity': {'Baseline': [0.34, 0.49, 0.55, 0.46], '+ Black Frames in Between': [0.51, 0.5, 0.63, 0.54], '+ Frame Index Provided': [0.47, 0.57, 0.52, 0.52], '+ Few-Shot Examples': [0.37, 0.4, 0.52, 0.43], '+ Oracle Estimation Teaching': [0.58, 0.54, 0.54, 0.55]}, 'Friction': {'Baseline': [0.46, 0.45, 0.97, 0.63], '+ Black Frames in Between': [0.47, 0.57, 0.93, 0.66], '+ Frame Index Provided': [0.54, 0.55, 0.98, 0.69], '+ Few-Shot Examples': [0.49, 0.47, 0.98, 0.65], '+ Oracle Estimation Teaching': [0.52, 0.59, 0.97, 0.69]}, 'Viscosity': {'Baseline': [0.67, 0.72, 0.95, 0.78], '+ Black Frames in Between': [0.59, 0.59, 0.94, 0.71], '+ Frame Index Provided': [0.68, 0.63, 0.97, 0.76], '+ Few-Shot Examples': [0.68, 0.71, 0.97, 0.79], '+ Oracle Estimation Teaching': [0.48, 0.71, 0.92, 0.7]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in relative_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            if property_name == 'Elasticity' and first_row:
                prop_display += ' 🔧'
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f"\n{'=' * 100}")
    print('🏆 Best strategy analysis (results after correction)')
    print('=' * 100)
    print(f'\n📈 Absolute value prediction task (Pearson Correlation):')
    print(f'🥇 Elasticity: + Few-Shot Examples (0.33) - 230.0% better than Baseline 🔧')
    print(f'🥇 Friction: Baseline (0.11) - Best benchmark performance')
    print(f'🥇 Viscosity: Baseline (0.17) - Best benchmark performance')print(f'\n📈 relative comparison task (AUC):')
    print(f'🥇 Elasticity: + Oracle Estimation Teaching (0.55) - 19.6% higher than Baseline 🔧')
    print(f'🥇 Friction: + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 9.5% higher than Baseline')
    print(f'🥇 Viscosity: + Few-Shot Examples (0.79) - 1.3% better than Baseline')
    print(f"\n{'=' * 100}")
    print('🔧Analysis of the impact of data quality filtering')
    print('=' * 100)
    print(f'\n📊 Elasticity filter statistics:')
    print(f'Absolute value task:')
    print(f'• Remove 3-5 samples with GT value <0.1 from each experiment')
    print(f'• Accounting for 3-5% of the total sample, the impact is relatively small')
    print(f'\n relative comparison task:')
    print(f'• Test-1: Remove 19 sample pairs (about 19%)')
    print(f'• Test-2: Removed 63 sample pairs (~63%!) - significant impact')
    print(f'• Test-3: Filtering effect is small')
    print(f'\n⚠️ Filter effect:')
    print(f'• Elasticity relative comparison AUC dropped to 0.55 from 0.65 previously')
    print(f'• It shows that samples with small GT values ​​are often easier to distinguish')
    print(f'• The filtered results can better reflect the true performance of the model on valid data')
    print(f"\n{'=' * 100}")
    print('💡 Cross-attribute analysis of strategy effectiveness (based on revised data)')
    print('=' * 100)
    print(f'\n🔍 Strategy performance Summarize:')
    print(f' 1. + Few-Shot Examples:')
    print(f'✅ Best performance in Elasticity absolute value prediction (0.33) 🔧')
    print(f'✅ Best performer on Viscosity relative comparison (0.79)')
    print(f'❌ Average performance on Friction task')
    print(f'📈 The overall most versatile strategy')
    print(f'\n 2. + Oracle Estimation Teaching:')
    print(f'✅ Best performer in Elasticity relative comparison (0.55) 🔧')
    print(f'✅ Best performance in Friction relative comparison (0.69)')
    print(f'⚖️Mediocre performance in absolute value predictions')
    print(f'📈 Expert in relative comparison tasks')
    print(f'\n 3. + Frame Index Provided:')
    print(f'✅ Best performance in Friction relative comparison (0.69)')
    print(f'✅ Performs well in Elasticity absolute value prediction (0.32) 🔧')
    print(f'⚖️ Overall performance is stable')
    print(f'\n 4. Baseline:')
    print(f'✅ Best performance in Friction and Viscosity absolute value prediction')
    print(f'❌ Poor performance on Elasticity tasks 🔧')
    print(f'💡 Effective for simple attributes, complex attributes require enhanced strategies')
    print(f"\n{'=' * 100}")
    print('🎯 Practical suggestions (based on revised data)')
    print('=' * 100)
    print(f'\n💼 Choose a strategy based on task type and data quality:')
    print(f'\n📏 Absolute value prediction:')
    print(f'- Elasticity (complex) → Use Few-Shot Examples (0.33)')
    print(f'- Friction/Viscosity (easy) → use Baseline')
    print(f'\n 🔄 relative comparison:')
    print(f'- Elasticity (Complex) → Use Oracle Estimation Teaching (0.55)')
    print(f'- Friction → Use Oracle/Frame Index (0.69)')
    print(f'- Viscosity → Use Few-Shot Examples (0.79)')
    print(f'\n 🔧 Data quality considerations:')
    print(f'• For data sets with a large number of low-quality samples, perform data filtering first')
    print(f'• Physical attributes such as Elasticity require special attention to the validity of GT values.')
    print(f'• Policy effectiveness may need to be re-evaluated after filtering')
    print(f'\n🚀 Advanced suggestions:')
    print(f'• Few-Shot Examples: The most versatile, suitable for complex attributes')
    print(f'• Oracle Teaching: Relative Comparison Expert')
    print(f'• Be careful when using simple attributes with overly complex prompts.')
    print(f'• Suggest ensemble Different strategy to gain robustness')
    print(f"\n{'=' * 100}")
    print('✅Comprehensive analysis completed after correction')
    print('=' * 100)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    abs_data = []
    for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])abs_filename = f'gemini_absolute_all_physics_corrected_summary_{timestamp}.csv'
    abs_df.to_csv(abs_filename, index=False)
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f'gemini_relative_all_physics_corrected_summary_{timestamp}.csv'
    rel_df.to_csv(rel_filename, index=False)
    print(f'\n📁 The corrected Absolute value prediction summary table has been saved as: {abs_filename}')
    print(f'📁 The revised relative comparison summary table has been saved as:{rel_filename}')
    print(f'\n🎉 The corrected analysis file has been generated, including:')
    print(f'• 3 single attribute detailed tables (elasticity🔧, friction, viscosity)')
    print(f'• 2 revised comprehensive summary tables')
    print(f'• Data quality filter statistics and impact analysis')
    print(f'• Strategy recommendations based on correction data')
    print(f'\n🔧 Note: Elasticity results have been filtered by GT value <0.1, which can better reflect the real model performance')
    print(f"{'=' * 100}")
if __name__ == '__main__':
    generate_corrected_comprehensive_summary()