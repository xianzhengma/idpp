#!/usr/bin/env python3
"""
生成美观的ACC分数elasticity relative实验结果表格
"""

import pandas as pd

def create_pretty_acc_table():
    """创建美观的ACC结果表格"""
    
    # ACC分数数据
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
            0.5741,  # Baseline
            0.7778,  # + ICL
            0.8750,  # + Oracle
            0.7216,  # + FrameIndex
            0.7778,  # + Concat+ICL
            0.8750,  # + Concat+Oracle
            0.7216,  # + Concat+FrameIndex
            0.8167,  # + ICL+Oracle
            0.7889,  # + ICL+FrameIndex
            0.7750,  # + Oracle+FrameIndex
            0.6571,  # + Concat+ICL+Oracle
            0.7857,  # + Concat+Oracle+FrameIndex
            1.0000   # + ICL+Oracle+FrameIndex
        ],
        'Test-2': [
            0.8249,  # Baseline
            0.7583,  # + ICL
            0.8295,  # + Oracle
            0.8286,  # + FrameIndex
            0.7583,  # + Concat+ICL
            0.8295,  # + Concat+Oracle
            0.8286,  # + Concat+FrameIndex
            0.7167,  # + ICL+Oracle
            0.7286,  # + ICL+FrameIndex
            0.7545,  # + Oracle+FrameIndex
            0.6167,  # + Concat+ICL+Oracle
            0.6889,  # + Concat+Oracle+FrameIndex
            0.6500   # + ICL+Oracle+FrameIndex
        ],
        'Test-3': [
            0.5290,  # Baseline
            0.5196,  # + ICL
            0.5000,  # + Oracle
            0.8125,  # + FrameIndex
            0.5196,  # + Concat+ICL
            0.5000,  # + Concat+Oracle
            0.8125,  # + Concat+FrameIndex
            0.8125,  # + ICL+Oracle
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
    
    # 按平均ACC分数排序（降序）
    df = df.sort_values('Average', ascending=False, na_position='last')
    
    # 格式化数值为4位小数
    for col in ['Test-1', 'Test-2', 'Test-3', 'Average']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def print_pretty_table(df):
    """打印美观的表格"""
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "Elasticity Relative (20 Examples) - ACC Results" + " " * 25 + "║")
    print("║" + " " * 30 + "(Sorted by Performance)" + " " * 45 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║ Rank │ Experiment                │ Test-1   │ Test-2   │ Test-3   │ Average  ║")
    print("╟" + "─" * 78 + "╢")
    
    # 打印数据行
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1 = row['Test-1']
        test2 = row['Test-2']
        test3 = row['Test-3']
        avg = row['Average']
        
        # 为前三名添加特殊标记
        if rank == 1:
            rank_str = "🥇"
        elif rank == 2:
            rank_str = "🥈"
        elif rank == 3:
            rank_str = "🥉"
        else:
            rank_str = f"{rank:2d}"
        
        print(f"║ {rank_str:<4} │ {experiment:<25} │ {test1:<8} │ {test2:<8} │ {test3:<8} │ {avg:<8} ║")
    
    print("╚" + "═" * 78 + "╝")

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
    df = create_pretty_acc_table()
    print_pretty_table(df)
    print_summary(df)
    
    # 保存到CSV文件
    output_file = "elasticity_relative_acc_results_pretty.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 表格已保存到: {output_file}")

if __name__ == "__main__":
    main() 