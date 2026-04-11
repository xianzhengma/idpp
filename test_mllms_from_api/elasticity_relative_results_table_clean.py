import pandas as pd

def create_clean_results_table():
    data = {'Experiment': ['Baseline', '+ ICL', '+ Oracle', '+ FrameIndex', '+ Concat+ICL', '+ Concat+Oracle', '+ Concat+FrameIndex', '+ ICL+Oracle', '+ ICL+FrameIndex', '+ Oracle+FrameIndex', '+ Concat+ICL+Oracle', '+ Concat+Oracle+FrameIndex', '+ ICL+Oracle+FrameIndex'], 'Test-1': [0.8389, 0.6746, 0.8611, 0.8125, 0.6746, 0.8611, 0.8125, 0.7444, 0.7889, 0.775, 0.6571, 0.7857, 1.0], 'Test-2': [0.7583, 0.6333, 0.8, 0.7615, 0.6333, 0.8, 0.7615, 0.7583, 0.7286, 0.7545, 0.6167, 0.6889, 0.65], 'Test-3': [0.8125, 0.5417, 0.5, 0.5441, 0.5417, 0.5, 0.5441, 0.5625, 0.6458, 1.0, 0.8571, None, 0.2857]}
    df = pd.DataFrame(data)
    df['Average'] = df[['Test-1', 'Test-2', 'Test-3']].apply(lambda x: x.mean() if x.notna().any() else None, axis=1)
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if x is not None else 'N/A')
    return df

def print_formatted_table(df):
    print('=' * 80)
    print('Elasticity Relative (20 Examples) - Ablation Study Results')
    print('=' * 80)
    print(f"{'Experiment':<25} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'Average':<10}")
    print('-' * 80)
    for _, row in df.iterrows():
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        print(f'{experiment:<25} {test1:<10} {test2:<10} {test3:<10} {avg:<10}')
    print('=' * 80)

def main():
    df = create_clean_results_table()
    print_formatted_table(df)
    output_file = 'elasticity_relative_results_table_clean.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ The table has been saved to: {output_file}')
if __name__ == '__main__':
    main()