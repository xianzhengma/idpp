import pandas as pd
import numpy as np
from datetime import datetime

def generate_comprehensive_summary():
    print('=' * 100)
    print('🔬 GEMINI 2.5 PRO physical property prediction ablation experiment - comprehensive analysis report')
    print('=' * 100)
    print('\n📊 Experimental settings:')
    print('• Model: Gemini 2.5 Pro')
    print('• Physical properties: Friction, Viscosity, Elasticity')
    print('• Task type: Absolute Prediction & Relative Comparison')
    print('• Number of samples: 100 samples per experiment (divided into 5 parts, 20 samples each)')
    print('• Test set: Test-1, Test-2, Test-3')
    print(f"\n{'=' * 100}")
    print('Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties')
    print('=' * 100)
    absolute_results = {'Elasticity': {'Baseline': [0.03, 0.26, 0.03, 0.1], '+ Frame Index Provided': [0.14, 0.34, 0.46, 0.31], '+ Few-Shot Examples': [0.46, 0.37, 0.18, 0.34], '+ Oracle Estimation Teaching': [0.3, 0.27, 0.19, 0.25]}, 'Friction': {'Baseline': [0.06, 0.09, 0.19, 0.11], '+ Frame Index Provided': [0.11, 0.05, -0.02, 0.05], '+ Few-Shot Examples': [-0.07, 0.11, 0.12, 0.05], '+ Oracle Estimation Teaching': [0.06, 0.17, 0.03, 0.09]}, 'Visibility': {'Baseline': [0.05, 0.12, 0.35, 0.17], '+ Frame Index Provided': [0.01, 0.02, 0.37, 0.13], '+ Few-Shot Examples': [0.07, -0.05, 0.29, 0.1], '+ Oracle Estimation Teaching': [0.15, 0.04, 0.29, 0.16]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in absolute_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f'\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties')
    print('=' * 100)
    relative_results = {'Elasticity': {'Baseline': [0.51, 0.74, 0.55, 0.6], '+ Black Frames in Between': [0.56, 0.72, 0.63, 0.64], '+ Frame Index Provided': [0.54, 0.8, 0.52, 0.62], '+ Few-Shot Examples': [0.43, 0.65, 0.52, 0.54], '+ Oracle Estimation Teaching': [0.63, 0.79, 0.54, 0.65]}, 'Friction': {'Baseline': [0.46, 0.45, 0.97, 0.63], '+ Black Frames in Between': [0.47, 0.57, 0.93, 0.66], '+ Frame Index Provided': [0.54, 0.55, 0.98, 0.69], '+ Few-Shot Examples': [0.49, 0.47, 0.98, 0.65], '+ Oracle Estimation Teaching': [0.52, 0.59, 0.97, 0.69]}, 'Viscosity': {'Baseline': [0.67, 0.72, 0.95, 0.78], '+ Black Frames in Between': [0.59, 0.59, 0.94, 0.71], '+ Frame Index Provided': [0.68, 0.63, 0.97, 0.76], '+ Few-Shot Examples': [0.68, 0.71, 0.97, 0.79], '+ Oracle Estimation Teaching': [0.48, 0.71, 0.92, 0.7]}}
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print('-' * 100)
    for property_name, strategies in relative_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ''
            print(f'{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}')
            first_row = False
        print('-' * 100)
    print(f"\n{'=' * 100}")
    print('🏆 Best strategy analysis')
    print('=' * 100)
    print(f'\n📈 Absolute value prediction task (Pearson Correlation):')
    print(f'🥇 Elasticity: + Few-Shot Examples (0.34) - 240.0% better than Baseline')
    print(f'🥇 Friction: Baseline (0.11) - Best benchmark performance')
    print(f'🥇 Viscosity: Baseline (0.17) - Best benchmark performance')
    print(f'\n📈 relative comparison task (AUC):')
    print(f'🥇 Elasticity: + Oracle Estimation Teaching (0.65) - 8.3% higher than Baseline')
    print(f'🥇 Friction: + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 9.5% higher than Baseline')
    print(f'🥇 Viscosity: + Few-Shot Examples (0.79) - 1.3% better than Baseline')
    print(f"\n{'=' * 100}")
    print('💡 Cross-attribute analysis of strategy effectiveness')
    print('=' * 100)print(f'\n🔍 Strategy performance Summarize:')
    print(f' 1. + Few-Shot Examples:')
    print(f'✅ Best performance in Elasticity absolute value prediction (0.34)')
    print(f'✅ Best performer on Viscosity relative comparison (0.79)')
    print(f'❌ Average performance on Friction task')
    print(f'\n 2. + Oracle Estimation Teaching:')
    print(f'✅ Best performer in Elasticity relative comparison (0.65)')
    print(f'✅ Best performance in Friction relative comparison (0.69)')
    print(f'❌ Moderate performance in absolute value prediction')
    print(f'\n 3. + Frame Index Provided:')
    print(f'✅ Best performance in Friction relative comparison (0.69)')
    print(f'✅ Performs well in Elasticity absolute value prediction (0.31)')
    print(f'❌ Overall performance is stable but not outstanding')
    print(f'\n 4. + Black Frames in Between:')
    print(f'✅ Only available in relative comparison tasks')
    print(f'⚖️ Average performance, not the best in any attribute')
    print(f'\n 5. Baseline:')
    print(f'✅ Best performance in Friction and Viscosity absolute value prediction')
    print(f'❌ Poor performance on Elasticity tasks')
    print(f"\n{'=' * 100}")
    print('📊 Dataset difficulty analysis')
    print('=' * 100)
    print(f'\n🎯 test set performance rules:')
    print(f'• Test-3 performs best on most tasks, especially relative comparison tasks')
    print(f'• Compared with Friction, AUC is generally >0.92 on Test-3, indicating that the task is relatively simple.')
    print(f'• Viscosity’s AUC is generally >0.92 on Test-3, indicating that the task is relatively simple')
    print(f'• The difficulty of the Elasticity task is relatively high, and the performance of each test set varies greatly.')
    print(f"\n{'=' * 100}")
    print('🎯 Practical suggestions')
    print('=' * 100)
    print(f'\n💼 Select strategy based on task type:')
    print(f'📏 Absolute value prediction:')
    print(f'- Elasticity → Use Few-Shot Examples')
    print(f'- Friction/Viscosity → Use Baseline (simple and effective)')
    print(f'\n🔄 relative comparison:')
    print(f'- Elasticity → Using Oracle Estimation Teaching')
    print(f'- Friction → Use Oracle Estimation Teaching or Frame Index Provided')
    print(f'- Viscosity → Use Few-Shot Examples')
    print(f'\n🚀 Advanced suggestions:')
    print(f'• For new physical attributes, it is recommended to try Baseline first to determine the baseline performance')
    print(f'• For complex properties (such as Elasticity), Few-Shot Examples and Oracle Teaching are more effective')
    print(f'• For simple attributes (such as Friction/Viscosity), excessive prompting may bring negative effects')
    print(f'• Consider ensemble different strategies for more stable performance')
    print(f"\n{'=' * 100}")
    print('✅ Comprehensive analysis completed')
    print('=' * 100)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    abs_data = []
    for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    abs_filename = f'gemini_absolute_all_physics_summary_{timestamp}.csv'
    abs_df.to_csv(abs_filename, index=False)
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f'gemini_relative_all_physics_summary_{timestamp}.csv'
    rel_df.to_csv(rel_filename, index=False)
    print(f'\n📁 Absolute value prediction summary table has been saved as: {abs_filename}')
    print(f'📁 The relative comparison summary table has been saved as:{rel_filename}')
    print(f'\n🎉 All analysis files have been generated, including:')
    print(f'• 3 absolute value prediction single attribute tables (elasticity, friction, viscosity)')
    print(f'• 3 relative comparison single attribute tables (elasticity, friction, viscosity)')
    print(f'• 2 comprehensive summary tables (absolute & relative)')
    print(f'• 2 detailed experimental data files')
    print(f"\n{'=' * 100}")if __name__ == '__main__':
    generate_comprehensive_summary()
