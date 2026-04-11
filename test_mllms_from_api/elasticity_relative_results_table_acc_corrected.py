import pandas as pd

def create_corrected_acc_table():
    data = {'Experiment': ['Baseline', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1': [0.8421, 0.6875, 0.8667, 0.8235, 0.7857, 0.7895, 0.7692, 0.6471, 0.7692, 1.0], 'Test-2': [0.8235, 0.75, 0.8571, 0.8333, 0.8235, 0.7895, 0.8125, 0.7059, 0.7143, 0.7333], 'Test-3': [0.6667, 0.6842, 0.3333, 0.5789, 0.6, 0.6316, 1.0, 0.75, 0.0, 0.5]}
    df = pd.DataFrame(data)
    df['Average'] = df[['Test-1', 'Test-2', 'Test-3']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    df = df.sort_values('Average', ascending=False, na_position='last')
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_formatted_table(df):
    print('=' * 80)
    print('Elasticity Relative (20 Examples) - ACC Results (Sorted by Performance)')
    print('=' * 80)
    print(f"{'Rank':<4} {'Experiment':<25} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'Average':<10}")
    print('-' * 80)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        print(f'{rank:<4} {experiment:<25} {test1:<10} {test2:<10} {test3:<10} {avg:<10}')
    print('=' * 80)

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
    df = create_corrected_acc_table()
    print_formatted_table(df)
    print_summary(df)
    output_file = 'elasticity_relative_acc_results_corrected.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The table has been saved to: {output_file}')
if __name__ == '__main__':
    main()