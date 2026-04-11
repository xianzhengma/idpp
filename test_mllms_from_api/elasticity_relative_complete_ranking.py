import pandas as pd

def create_complete_ranking():
    data = {'Experiment': ['Baseline', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1_AUC': [0.8389, 0.6746, 0.8611, 0.8125, 0.7444, 0.7889, 0.775, 0.6571, 0.7857, 1.0], 'Test-1_ACC': [0.8421, 0.6875, 0.8667, 0.8235, 0.7857, 0.7895, 0.7692, 0.6471, 0.7692, 1.0], 'Test-2_AUC': [0.7583, 0.6333, 0.8, 0.7615, 0.7583, 0.7286, 0.7545, 0.6167, 0.6889, 0.65], 'Test-2_ACC': [0.8235, 0.75, 0.8571, 0.8333, 0.8235, 0.7895, 0.8125, 0.7059, 0.7143, 0.7333], 'Test-3_AUC': [0.8125, 0.5417, 0.5, 0.5441, 0.5625, 0.6458, 1.0, 0.8571, None, 0.2857], 'Test-3_ACC': [0.6667, 0.6842, 0.3333, 0.5789, 0.6, 0.6316, 1.0, 0.75, 0.0, 0.5]}
    df = pd.DataFrame(data)
    df['Avg_AUC'] = df[['Test-1_AUC', 'Test-2_AUC', 'Test-3_AUC']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df['Avg_ACC'] = df[['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df = df.sort_values('Avg_ACC', ascending=False, na_position='last')
    for col in ['Test-1_AUC', 'Test-1_ACC', 'Test-2_AUC', 'Test-2_ACC', 'Test-3_AUC', 'Test-3_ACC', 'Avg_AUC', 'Avg_ACC']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_complete_ranking(df):
    print('╔' + '═' * 120 + '╗')
    print('║' + ' ' * 45 + 'Elasticity Relative (20 Examples) - Complete Ranking' + ' ' * 45 + '║')
    print('╠' + '═' * 120 + '╣')
    print('║Rank │ Experiment │ Test-1_AUC │ Test-1_ACC │ Test-2_AUC │ Test-2_ACC │ Test-3_AUC │ Test-3_ACC │ Avg_AUC │ Avg_ACC ║')
    print('╟' + '─' * 120 + '╢')
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1_auc = row['Test-1_AUC']
        test1_acc = row['Test-1_ACC']
        test2_auc = row['Test-2_AUC']
        test2_acc = row['Test-2_ACC']
        test3_auc = row['Test-3_AUC']
        test3_acc = row['Test-3_ACC']
        avg_auc = row['Avg_AUC']
        avg_acc = row['Avg_ACC']
        if rank == 1:
            rank_str = '🥇'
        elif rank == 2:
            rank_str = '🥈'
        elif rank == 3:
            rank_str = '🥉'
        else:
            rank_str = f'{rank:2d}'
        print(f'║ {rank_str:<4} │ {experiment:<25} │ {test1_auc:<10} │ {test1_acc:<10} │ {test2_auc:<10} │ {test2_acc:<10} │ {test3_auc:<10} │ {test3_acc:<10} │ {avg_auc:<9} │ {avg_acc:<9} ║')
    print('╚' + '═' * 120 + '╝')

def print_detailed_analysis(df):
    print('\n' + '=' * 120)
    print('📊 Detailed analysis')
    print('=' * 120)
    top3 = df.head(3)
    print('🏆Top three performances (sorted by ACC):')
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        avg_auc = row['Avg_AUC']
        print(f' {i}. {experiment}: ACC={avg_acc}, AUC={avg_auc}')
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_acc = baseline_row['Avg_ACC']
    baseline_auc = baseline_row['Avg_AUC']
    print(f'\n📈 Baseline performance (Baseline): ACC={baseline_acc}, AUC={baseline_auc}')
    best_acc = float(df.iloc[0]['Avg_ACC'])
    improvement_acc = (best_acc - float(baseline_acc)) / float(baseline_acc) * 100
    print(f'🚀 Best ACC improvements: +{improvement_acc:.1f}% (relative to baseline)')
    df_auc_sorted = df.copy()
    df_auc_sorted['Avg_AUC_num'] = df_auc_sorted['Avg_AUC'].apply(lambda x: float(x) if x != 'N/A' else 0)
    df_auc_sorted = df_auc_sorted.sort_values('Avg_AUC_num', ascending=False)
    print(f'\n🏆Top three performances (sorted by AUC):')
    for i, (_, row) in enumerate(df_auc_sorted.head(3).iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        avg_auc = row['Avg_AUC']
        print(f' {i}. {experiment}: AUC={avg_auc}, ACC={avg_acc}')
    print(f'\n📋 Best performance of each test set:')
    test_sets = ['Test-1', 'Test-2', 'Test-3']
    for test_set in test_sets:
        best_exp = None
        best_acc=0
        for _, row in df.iterrows():
            acc_val = row[f'{test_set}_ACC']if acc_val != 'N/A':
                acc_num = float(acc_val)
                if acc_num > best_acc:
                    best_acc = acc_num
                    best_exp = row['Experiment']
        if best_exp:
            print(f' {test_set}: {best_exp} (ACC={best_acc:.4f})')
    print('=' * 120)

def main():
    df = create_complete_ranking()
    print_complete_ranking(df)
    print_detailed_analysis(df)
    output_file = 'elasticity_relative_complete_ranking.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The complete ranking table has been saved to: {output_file}')
if __name__ == '__main__':
    main()