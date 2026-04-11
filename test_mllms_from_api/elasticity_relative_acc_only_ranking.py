import pandas as pd

def create_acc_only_ranking():
    data = {'Experiment': ['Baseline', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1_ACC': [0.8421, 0.6875, 0.8667, 0.8235, 0.7857, 0.7895, 0.7692, 0.6471, 0.7692, 1.0], 'Test-2_ACC': [0.8235, 0.75, 0.8571, 0.8333, 0.8235, 0.7895, 0.8125, 0.7059, 0.7143, 0.7333], 'Test-3_ACC': [0.6667, 0.6842, 0.3333, 0.5789, 0.6, 0.6316, 1.0, 0.75, 0.0, 0.5]}
    df = pd.DataFrame(data)
    df['Avg_ACC'] = df[['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df = df.sort_values('Avg_ACC', ascending=False, na_position='last')
    for col in ['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC', 'Avg_ACC']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_acc_ranking(df):
    print('╔' + '═' * 80 + '╗')
    print('║' + ' ' * 30 + 'Elasticity Relative (20 Examples) - ACC Ranking' + ' ' * 30 + '║')
    print('╠' + '═' * 80 + '╣')
    print('║Rank │ Experiment │ Test-1_ACC │ Test-2_ACC │ Test-3_ACC │ Avg_ACC ║')
    print('╟' + '─' * 80 + '╢')
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1_acc = row['Test-1_ACC']
        test2_acc = row['Test-2_ACC']
        test3_acc = row['Test-3_ACC']
        avg_acc = row['Avg_ACC']
        if rank == 1:
            rank_str = '🥇'
        elif rank == 2:
            rank_str = '🥈'
        elif rank == 3:
            rank_str = '🥉'
        else:
            rank_str = f'{rank:2d}'
        print(f'║ {rank_str:<4} │ {experiment:<25} │ {test1_acc:<10} │ {test2_acc:<10} │ {test3_acc:<10} │ {avg_acc:<9} ║')
    print('╚' + '═' * 80 + '╝')

def print_acc_analysis(df):
    print('\n' + '=' * 80)
    print('📊ACC Analysis')
    print('=' * 80)
    top3 = df.head(3)
    print('🏆 Top three performances:')
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        print(f' {i}. {experiment}: ACC={avg_acc}')
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_acc = baseline_row['Avg_ACC']
    print(f'\n📈 Baseline: ACC={baseline_acc}')
    best_acc = float(df.iloc[0]['Avg_ACC'])
    improvement = (best_acc - float(baseline_acc)) / float(baseline_acc) * 100
    print(f'🚀 Best ACC improvements: +{improvement:.1f}% (relative to baseline)')
    print(f'\n📋The best ACC performance of each test set:')
    test_sets = ['Test-1', 'Test-2', 'Test-3']
    for test_set in test_sets:
        best_exp = None
        best_acc=0
        for _, row in df.iterrows():
            acc_val = row[f'{test_set}_ACC']
            if acc_val != 'N/A':
                acc_num = float(acc_val)
                if acc_num > best_acc:
                    best_acc = acc_num
                    best_exp = row['Experiment']
        if best_exp:
            print(f' {test_set}: {best_exp} (ACC={best_acc:.4f})')
    print(f'\n📈 Stability analyze (the standard deviation of each test setACC, the smaller the more stable it is):')
    for _, row in df.iterrows():
        experiment = row['Experiment']
        acc_values = []
        for test_set in test_sets:
            acc_val = row[f'{test_set}_ACC']
            if acc_val != 'N/A':
                acc_values.append(float(acc_val))
        if len(acc_values) >= 2:
            import numpy as np
            std_dev = np.std(acc_values)
            print(f' {experiment}: standard deviation ={std_dev:.4f}')
    print('=' * 80)

def main():
    df = create_acc_only_ranking()
    print_acc_ranking(df)
    print_acc_analysis(df)
    output_file = 'elasticity_relative_acc_only_ranking.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ ACC ranking table has been saved to: {output_file}')
if __name__ == '__main__':
    main()