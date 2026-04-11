#!/usr/bin/env python3
"""
生成简洁的elasticity relative实验结果表格
"""

import pandas as pd

def create_clean_results_table():
    """创建简洁的实验结果表格"""
    
    # 只包含有结果的实验
    data = {
        'Experiment': [
            'Baseline',
            '+ ICL',
            '+ Oracle',
            '+ FrameIndex',
            '+ Concat+ICL',
            '+ Concat+Oracle',
            '+ Concat+FrameIndex',
            '+ ICL+Oracle',
            '+ ICL+FrameIndex',
            '+ Oracle+FrameIndex',
            '+ Concat+ICL+Oracle',
            '+ Concat+Oracle+FrameIndex',
            '+ ICL+Oracle+FrameIndex'
        ],
        'Test-1': [
            0.8389,  # Baseline
            0.6746,  # + ICL
            0.8611,  # + Oracle
            0.8125,  # + FrameIndex
            0.6746,  # + Concat+ICL
            0.8611,  # + Concat+Oracle
            0.8125,  # + Concat+FrameIndex
            0.7444,  # + ICL+Oracle
            0.7889,  # + ICL+FrameIndex
            0.7750,  # + Oracle+FrameIndex
            0.6571,  # + Concat+ICL+Oracle
            0.7857,  # + Concat+Oracle+FrameIndex
            1.0000   # + ICL+Oracle+FrameIndex
        ],
        'Test-2': [
            0.7583,  # Baseline
            0.6333,  # + ICL
            0.8000,  # + Oracle
            0.7615,  # + FrameIndex
            0.6333,  # + Concat+ICL
            0.8000,  # + Concat+Oracle
            0.7615,  # + Concat+FrameIndex
            0.7583,  # + ICL+Oracle
            0.7286,  # + ICL+FrameIndex
            0.7545,  # + Oracle+FrameIndex
            0.6167,  # + Concat+ICL+Oracle
            0.6889,  # + Concat+Oracle+FrameIndex
            0.6500   # + ICL+Oracle+FrameIndex
        ],
        'Test-3': [
            0.8125,  # Baseline
            0.5417,  # + ICL
            0.5000,  # + Oracle
            0.5441,  # + FrameIndex
            0.5417,  # + Concat+ICL
            0.5000,  # + Concat+Oracle
            0.5441,  # + Concat+FrameIndex
            0.5625,  # + ICL+Oracle
            0.6458,  # + ICL+FrameIndex
            1.0000,  # + Oracle+FrameIndex
            0.8571,  # + Concat+ICL+Oracle
            None,    # + Concat+Oracle+FrameIndex (无结果)
            0.2857   # + ICL+Oracle+FrameIndex
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 计算平均值（忽略None值）
    df['Average'] = df[['Test-1', 'Test-2', 'Test-3']].apply(
        lambda x: x.mean() if x.notna().any() else None, axis=1
    )
    
    # 格式化数值为4位小数
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def print_formatted_table(df):
    """打印格式化的表格"""
    
    print("=" * 80)
    print("Elasticity Relative (20 Examples) - Ablation Study Results")
    print("=" * 80)
    
    # 打印表头
    print(f"{'Experiment':<25} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'Average':<10}")
    print("-" * 80)
    
    # 打印数据行
    for _, row in df.iterrows():
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        
        print(f"{experiment:<25} {test1:<10} {test2:<10} {test3:<10} {avg:<10}")
    
    print("=" * 80)

def main():
    """主函数"""
    df = create_clean_results_table()
    print_formatted_table(df)
    
    # 保存到CSV文件
    output_file = "elasticity_relative_results_table_clean.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 表格已保存到: {output_file}")

if __name__ == "__main__":
    main() 