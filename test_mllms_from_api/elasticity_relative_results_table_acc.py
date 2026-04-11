import pandas as pd

def create_acc_results_table():
    data = {'Experiment': ['Baseline', '+ ICL', '+ Oracle', '+ FrameIndex', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1': [0.5741, 0.7778, 0.875, 0.7216, 0.7778, 0.875, 0.7216, 0.8167, 0.7889, 0.775, 0.6571, 0.7857, 1.0], 'Test-2': [0.8249, 0.7583, 0.8295, 0.8286, 0.7583, 0.8295, 0.8286, 0.7167, 0.7286, 0.7545, 0.6167, 0.6889, 0.65], 'Test-3': [0.529, 0.5196, 0.5, 0.8125, 0.5196, 0.5, 0.8125, 0.8125, 0.6458, 1.0, 0.8571, None, 0.2857]}
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

def main():
    df = create_acc_results_table()
    print_formatted_table(df)
    output_file = 'elasticity_relative_acc_results_table.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The table has been saved to: {output_file}')
if __name__ == '__main__':
    main()