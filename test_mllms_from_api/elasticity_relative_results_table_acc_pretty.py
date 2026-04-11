import pandas as pd

def create_pretty_acc_table():
    data = {'Experiment': ['Baseline', '+ ICL', '+ Oracle', '+ FrameIndex', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1': [0.5741, 0.7778, 0.875, 0.7216, 0.7778, 0.875, 0.7216, 0.8167, 0.7889, 0.775, 0.6571, 0.7857, 1.0], 'Test-2': [0.8249, 0.7583, 0.8295, 0.8286, 0.7583, 0.8295, 0.8286, 0.7167, 0.7286, 0.7545, 0.6167, 0.6889, 0.65], 'Test-3': [0.529, 0.5196, 0.5, 0.8125, 0.5196, 0.5, 0.8125, 0.8125, 0.6458, 1.0, 0.8571, None, 0.2857]}
    df = pd.DataFrame(data)
    df['Average'] = df[['Test-1', 'Test-2', 'Test-3']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df = df.sort_values('Average', ascending=False, na_position='last')
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_pretty_table(df):
    print('╔' + '═' * 78 + '╗')
    print('║' + ' ' * 25 + 'Elasticity Relative (20 Examples) - ACC Results' + ' ' * 25 + '║')
    print('║' + ' ' * 30 + '(Sorted by Performance)' + ' ' * 45 + '║')
    print('╠' + '═' * 78 + '╣')
    print('║Rank │ Experiment │ Test-1 │ Test-2 │ Test-3 │ Average ║')
    print('╟' + '─' * 78 + '╢')
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        if rank == 1:
            rank_str = '🥇'
        elif rank == 2:
            rank_str = '🥈'
        elif rank == 3:
            rank_str = '🥉'
        else:
            rank_str = f'{rank:2d}'
        print(f'║ {rank_str:<4} │ {experiment:<25} │ {test1:<8} │ {test2:<8} │ {test3:<8} │ {avg:<8} ║')
    print('╚' + '═' * 78 + '╝')

def print_summary(df):
    print('\n' + '=' * 80)
    print('📊 Summary of results')
    print('=' * 80)
    top3 = df.head(3)
    print('🏆 Top three performances:')
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg = row['Average']
        print(f' {i}. {experiment}: {avg}')
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_avg = baseline_row['Average']
    print(f'\n📈 Baseline: {baseline_avg}')
    best_avg = float(df.iloc[0]['Average'])
    improvement = (best_avg - float(baseline_avg)) / float(baseline_avg) * 100
    print(f'🚀 Best improvements: +{improvement:.1f}% (relative to baseline)')
    print('=' * 80)

def main():
    df = create_pretty_acc_table()
    print_pretty_table(df)
    print_summary(df)
    output_file = 'elasticity_relative_acc_results_pretty.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The table has been saved to: {output_file}')
if __name__ == '__main__':
    main()