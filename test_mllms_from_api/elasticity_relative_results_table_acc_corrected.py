#!/usr/bin/env python3
"""
生成使用正确ACC分数的elasticity relative实验结果表格
"""

import pandas as pd

def create_corrected_acc_table():
    """创建使用正确ACC分数的实验结果表格"""
    
    # 基于实际实验结果文件的ACC分数数据
    data = {
        'Experiment': [
            'Baseline',
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
            0.8421,  # Baseline
            0.6875,  # + Concat+ICL
            0.8667,  # + Concat+Oracle
            0.8235,  # + Concat+FrameIndex
            0.7857,  # + ICL+Oracle
            0.7895,  # + ICL+FrameIndex
            0.7692,  # + Oracle+FrameIndex
            0.6471,  # + Concat+ICL+Oracle
            0.7692,  # + Concat+Oracle+FrameIndex
            1.0000   # + ICL+Oracle+FrameIndex
        ],
        'Test-2': [
            0.8235,  # Baseline
            0.7500,  # + Concat+ICL
            0.8571,  # + Concat+Oracle
            0.8333,  # + Concat+FrameIndex
            0.8235,  # + ICL+Oracle
            0.7895,  # + ICL+FrameIndex
            0.8125,  # + Oracle+FrameIndex
            0.7059,  # + Concat+ICL+Oracle
            0.7143,  # + Concat+Oracle+FrameIndex
            0.7333   # + ICL+Oracle+FrameIndex
        ],
        'Test-3': [
            0.6667,  # Baseline
            0.6842,  # + Concat+ICL
            0.3333,  # + Concat+Oracle
            0.5789,  # + Concat+FrameIndex
            0.6000,  # + ICL+Oracle
            0.6316,  # + ICL+FrameIndex
            1.0000,  # + Oracle+FrameIndex
            0.7500,  # + Concat+ICL+Oracle
            0.0000,  # + Concat+Oracle+FrameIndex (只有1个有效样本)
            0.5000   # + ICL+Oracle+FrameIndex
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 计算平均值（忽略None值）
    df['Average'] = df[['Test-1', 'Test-2', 'Test-3']].apply(
        lambda x: x.mean() if x.notna().any() else None, axis=1
    )
    
    # 按平均ACC分数排序（降序）
    df = df.sort_values('Average', ascending=False, na_position='last')
    
    # 格式化数值为4位小数
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def print_formatted_table(df):
    """打印格式化的表格"""
    
    print("=" * 80)
    print("Elasticity Relative (20 Examples) - ACC Results (Sorted by Performance)")
    print("=" * 80)
    
    # 打印表头
    print(f"{'Rank':<4} {'Experiment':<25} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'Average':<10}")
    print("-" * 80)
    
    # 打印数据行
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        
        print(f"{rank:<4} {experiment:<25} {test1:<10} {test2:<10} {test3:<10} {avg:<10}")
    
    print("=" * 80)

def print_summary(df):
    """打印结果摘要"""
    print("\n" + "=" * 80)
    print("📊 结果摘要")
    print("=" * 80)
    
    # 获取前三名
    top3 = df.head(3)
    print("🏆 前三名表现:")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg = row['Average']
        print(f"   {i}. {experiment}: {avg}")
    
    # 获取基准表现
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_avg = baseline_row['Average']
    print(f"\n📈 基准表现 (Baseline): {baseline_avg}")
    
    # 计算相对于基准的改进
    best_avg = float(df.iloc[0]['Average'])
    improvement = ((best_avg - float(baseline_avg)) / float(baseline_avg)) * 100
    print(f"🚀 最佳改进: +{improvement:.1f}% (相对于基准)")
    
    print("=" * 80)

def main():
    """主函数"""
    df = create_corrected_acc_table()
    print_formatted_table(df)
    print_summary(df)
    
    # 保存到CSV文件
    output_file = "elasticity_relative_acc_results_corrected.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 表格已保存到: {output_file}")

if __name__ == "__main__":
    main() 