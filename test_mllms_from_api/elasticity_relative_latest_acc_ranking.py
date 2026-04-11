import pandas as pd

def create_latest_acc_ranking():
    data = {'Experiment': ['Baseline', '+ Concat', '+ ICL', '+ Oracle', '+ FrameIndex', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1_ACC': [0.8421, 0.85, 0.8, 0.8667, 0.7368, 0.6875, 0.8667, 0.8235, 0.7857, 0.7895, 0.7692, 0.6471, 0.7692, 1.0], 'Test-2_ACC': [0.8235, 0.8235, 0.8235, 0.8667, 0.8421, 0.75, 0.8571, 0.8333, 0.8235, 0.7895, 0.8125, 0.7059, 0.7143, 0.7333], 'Test-3_ACC': [0.6667, 0.5556, 0.65, 0.6, 0.6667, 0.6842, 0.3333, 0.5789, 0.6, 0.6316, 1.0, 0.75, 0.0, 0.5]}
    df = pd.DataFrame(data)
    df['Avg_ACC'] = df[['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df = df.sort_values('Avg_ACC', ascending=False, na_position='last')
    for col in ['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC', 'Avg_ACC']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_latest_acc_ranking(df):
    print('╔' + '═' * 80 + '╗')
    print('║' + ' ' * 25 + 'Elasticity Relative (20 Examples) - Latest ACC Ranking' + ' ' * 25 + '║')
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

def print_latest_analysis(df):
    print('\n' + '=' * 80)
    print('📊Latest ACC analysis')
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
    print(f'\n🔍 Single function effect analysis:')
    single_features = ['+ Concat', '+ ICL', '+ Oracle', '+ FrameIndex']
    for feature in single_features:
        if feature in df['Experiment'].values:
            feature_row = df[df['Experiment'] == feature].iloc[0]
            feature_acc = feature_row['Avg_ACC']
            baseline_acc_num = float(baseline_acc)
            feature_acc_num = float(feature_acc)
            improvement = (feature_acc_num - baseline_acc_num) / baseline_acc_num * 100
            print(f' {feature}: ACC={feature_acc} ({improvement:+.1f}% vs Baseline)')
    print('=' * 80)

def main():
    df = create_latest_acc_ranking()
    print_latest_acc_ranking(df)
    print_latest_analysis(df)
    output_file = 'elasticity_relative_latest_acc_ranking.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The latest ACC ranking table has been saved to: {output_file}')
if __name__ == '__main__':
    main()