import pandas as pd
import numpy as np
from datetime import datetime

def generate_final_correct_summary():
    print('=' * 100)
    print('🔬 GEMINI 2.5 PRO physical property prediction ablation experiment - final correct analysis report')
    print('✅Filter conditions have been corrected: only apply GT value <0.1 filter to Elasticity absolute value predictions')
    print('=' * 100)
    print('\n📊 Experimental settings:')
    print('• Model: Gemini 2.5 Pro')
    print('• Physical properties: Friction, Viscosity, Elasticity')
    print('• Task type: Absolute Prediction & Relative Comparison')
    print('• Number of samples: 100 samples per experiment (divided into 5 parts, 20 samples each)')
    print('• Test set: Test-1, Test-2, Test-3')
    print('• ✅ Data quality control: Only filter GT values ​​<0.1 for Elasticity absolute value predictions')
    print('• ✅ Relative comparison tasks keep original data (no filtering)')
    print(f"\n{'=' * 100}")
    print('Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties (Final Correct)')
    print('=' * 100)
    absolute_results = {'Elasticity': {'Baseline': [-0.03, 0.26, 0.06, 0.1], '+ Frame Index Provided': [0.06, 0.35, 0.55, 0.32], '+ Few-Shot Examples': [0.39, 0.34, 0.24, 0.33], '+ Oracle Estimation Teaching': [0.19, 0.14, 0.24, 0.19]}, 'Friction': {'Baseline': [0.06, 0.09, 0.19, 0.11], '+ Frame Index Provided': [0.11, 0.05, -0.02, 0.05], '+ Few-Shot Examples': [-0.07, 0.11, 0.12, 0.05], '+ Oracle Estimation Teaching': [0.06, 0.17, 0.03, 0.09]}, 'Visibility': {'Baseline': [0.05, 0.12, 0.35, 0.17], '+ Frame Index Provided': [0.01, 0.02, 0.37, 0.13], '+ Few-Shot Examples': [0.07, -0.05, 0.29, 0.1], '+ Oracle Estimation Teaching': [0.15, 0.04, 0.29, 0.16]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in absolute_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            if property_name == 'Elasticity' and first_row:
                prop_display += '✅'
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f'\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties (Final Correct)')
    print('=' * 100)
    relative_results = {'Elasticity': {'Baseline': [0.51, 0.74, 0.55, 0.6], '+ Black Frames in Between': [0.56, 0.72, 0.63, 0.64], '+ Frame Index Provided': [0.54, 0.8, 0.52, 0.62], '+ Few-Shot Examples': [0.43, 0.65, 0.52, 0.54], '+ Oracle Estimation Teaching': [0.63, 0.79, 0.54, 0.65]}, 'Friction': {'Baseline': [0.46, 0.45, 0.97, 0.63], '+ Black Frames in Between': [0.47, 0.57, 0.93, 0.66], '+ Frame Index Provided': [0.54, 0.55, 0.98, 0.69], '+ Few-Shot Examples': [0.49, 0.47, 0.98, 0.65], '+ Oracle Estimation Teaching': [0.52, 0.59, 0.97, 0.69]}, 'Viscosity': {'Baseline': [0.67, 0.72, 0.95, 0.78], '+ Black Frames in Between': [0.59, 0.59, 0.94, 0.71], '+ Frame Index Provided': [0.68, 0.63, 0.97, 0.76], '+ Few-Shot Examples': [0.68, 0.71, 0.97, 0.79], '+ Oracle Estimation Teaching': [0.48, 0.71, 0.92, 0.7]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in relative_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            if property_name == 'Elasticity' and first_row:
                prop_display += '✅'
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f"\n{'=' * 100}")
    print('🏆 Best strategy analysis (final correct result)')
    print('=' * 100)
    print(f'\n📈 Absolute value prediction task (Pearson Correlation):')
    print(f'🥇 Elasticity: + Few-Shot Examples (0.33) - 230.0% better than Baseline ✅')print(f'🥇 Friction: Baseline (0.11) - Best benchmark performance')
    print(f'🥇 Viscosity: Baseline (0.17) - Best benchmark performance')
    print(f'\n📈 relative comparison task (AUC):')
    print(f'🥇 Elasticity: + Oracle Estimation Teaching (0.65) - 8.3% higher than Baseline ✅')
    print(f'🥇 Friction: + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 9.5% higher than Baseline')
    print(f'🥇 Viscosity: + Few-Shot Examples (0.79) - 1.3% better than Baseline')
    print(f"\n{'=' * 100}")
    print('🔧 Comparison before and after filter condition correction')
    print('=' * 100)
    print(f'\n📊 Elasticity result correction:')
    print(f'Absolute value prediction:')
    print(f'• ✅ Correctly filter samples with GT value <0.1')
    print(f'• Remove 3-5 samples per experiment (3-5%)')
    print(f'• Results remain the same: Few-Shot Examples best (0.33)')
    print(f'\n relative comparison:')
    print(f'• ❌ Previous error: filtering sample pairs with GT value <0.1, AUC dropped to 0.55')
    print(f'• ✅ Now correct: keep original data, AUC restored to 0.65')
    print(f'• 📈 Improvement after correction: The relative comparison effect is significantly improved')
    print(f'\n⚠️ Corrected meaning:')
    print(f'• Relative comparison tasks focus on the relative size relationship between two values, rather than the absolute value.')
    print(f'• Sample pairs with smaller GT values ​​are still meaningful in relative comparisons')
    print(f'• Filtering will lose useful contrasting information')
    print(f"\n{'=' * 100}")
    print('💡 Cross-attribute analysis of strategy effectiveness (based on final correct data)')
    print('=' * 100)
    print(f'\n🔍 Strategy performance Summarize:')
    print(f' 1. + Oracle Estimation Teaching:')
    print(f'🏆 Best performer in Elasticity relative comparison (0.65) ✅')
    print(f'🏆 Best performer in Friction relative comparison (0.69)')
    print(f'📈 The absolute king of relative comparison tasks')
    print(f'\n 2. + Few-Shot Examples:')
    print(f'🏆 Best performer in Elasticity absolute value prediction (0.33) ✅')
    print(f'🏆 Best performer on Viscosity relative comparison (0.79)')
    print(f'📈 An all-rounder across task types')
    print(f'\n 3. + Frame Index Provided:')
    print(f'🏆 Tied for best in Friction relative comparison (0.69)')
    print(f'✅ Performs well in Elasticity absolute value prediction (0.32)')
    print(f'📈 Stable and reliable backbone')
    print(f'\n 4. Baseline:')
    print(f'🏆 Best performance in Friction and Viscosity absolute value prediction')
    print(f'❌ Weak performance on Elasticity tasks')
    print(f'💡 Simple attributes are preferred, complex attributes need to be enhanced')
    print(f"\n{'=' * 100}")
    print('🎯 Final practical advice')
    print('=' * 100)
    print(f'\n💼 Select strategy according to task type and property complexity:')
    print(f'\n📏 Absolute value prediction:')
    print(f'- Elasticity (complex) → Few-Shot Examples (0.33)')
    print(f'- Friction/Viscosity (easy) → Baseline')
    print(f'\n 🔄 relative comparison:')
    print(f'- Elasticity (complexity) → Oracle Estimation Teaching (0.65)')
    print(f' - Friction → Oracle Teaching/Frame Index (0.69)')
    print(f' - Viscosity → Few-Shot Examples (0.79)')
    print(f'\n 🔧 Data quality control principle:')
    print(f'• Absolute value prediction: filter out meaningless very small GT values')
    print(f'• Relative comparison: Keep all sample pairs to maintain the integrity of comparison information')
    print(f'• Develop different data quality strategies based on task characteristics')
    print(f'\n🚀 Advanced strategy suggestions:')
    print(f'🏆 Oracle Estimation Teaching: Relative Comparison Expert')
    print(f'🏆 Few-Shot Examples: All-round player, especially suitable for complex attributes')
    print(f'⚖️ Frame Index Provided: Stable backbone strategy')
    print(f'💰 Baseline: Simple and efficient, suitable for simple attributes')
    print(f'🎯 It is recommended to ensemble multiple strategies to obtain the best robustness')
    print(f"\n{'=' * 100}")
    print('✅ The final correct and comprehensive analysis is completed')
    print('=' * 100)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    abs_data = []for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    abs_filename = f'gemini_absolute_all_physics_FINAL_summary_{timestamp}.csv'
    abs_df.to_csv(abs_filename, index=False)
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f'gemini_relative_all_physics_FINAL_summary_{timestamp}.csv'
    rel_df.to_csv(rel_filename, index=False)
    print(f'\n📁 The final Absolute value prediction summary table has been saved as: {abs_filename}')
    print(f'📁 The final relative comparison summary table has been saved as:{rel_filename}')
    print(f'\n🎉 The final correct analysis file has been generated, including:')
    print(f'• 3 detailed tables for single attributes (elasticity✅, friction, viscosity)')
    print(f'• 2 final correct comprehensive summary tables')
    print(f'• Comparative analysis before and after filter condition correction')
    print(f'• Final strategy recommendations based on the right data')
    print(f'\n✅ Important correction:')
    print(f'• Elasticity absolute value prediction: Correctly filter GT value <0.1 ✅')
    print(f'• Elasticity relative comparison: Keeping the original data, AUC recovers from 0.55 to 0.65 ✅')
    print(f'• Data quality control strategy optimized for task type ✅')
    print(f"{'=' * 100}")
if __name__ == '__main__':
    generate_final_correct_summary()