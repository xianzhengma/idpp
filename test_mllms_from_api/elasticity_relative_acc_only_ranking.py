#!/usr/bin/env python3
"""
展示Elasticity Relative实验的ACC排名（不参考AUC）
"""

import pandas as pd

def create_acc_only_ranking():
    """创建只按ACC排名的表格"""
    
    # 基于实际实验结果文件的ACC数据
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
        'Test-1_ACC': [
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
        'Test-2_ACC': [
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
        'Test-3_ACC': [
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
    
    # 计算平均ACC
    df['Avg_ACC'] = df[['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC']].apply(
        lambda x: x.mean() if x.notna().any() else None, axis=1
    )
    
    # 按平均ACC分数排序（降序）
    df = df.sort_values('Avg_ACC', ascending=False, na_position='last')
    
    # 格式化数值为4位小数
    for col in ['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC', 'Avg_ACC']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def print_acc_ranking(df):
    """打印ACC排名表格"""
    
    print("╔" + "═" * 80 + "╗")
    print("║" + " " * 30 + "Elasticity Relative (20 Examples) - ACC Ranking" + " " * 30 + "║")
    print("╠" + "═" * 80 + "╣")
    print("║ Rank │ Experiment                │ Test-1_ACC │ Test-2_ACC │ Test-3_ACC │ Avg_ACC   ║")
    print("╟" + "─" * 80 + "╢")
    
    # 打印数据行
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        experiment = row['Experiment']
        test1_acc = row['Test-1_ACC']
        test2_acc = row['Test-2_ACC']
        test3_acc = row['Test-3_ACC']
        avg_acc = row['Avg_ACC']
        
        # 为前三名添加特殊标记
        if rank == 1:
            rank_str = "🥇"
        elif rank == 2:
            rank_str = "🥈"
        elif rank == 3:
            rank_str = "🥉"
        else:
            rank_str = f"{rank:2d}"
        
        print(f"║ {rank_str:<4} │ {experiment:<25} │ {test1_acc:<10} │ {test2_acc:<10} │ {test3_acc:<10} │ {avg_acc:<9} ║")
    
    print("╚" + "═" * 80 + "╝")

def print_acc_analysis(df):
    """打印ACC分析"""
    print("\n" + "=" * 80)
    print("📊 ACC分析")
    print("=" * 80)
    
    # 获取前三名
    top3 = df.head(3)
    print("🏆 前三名表现:")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        print(f"   {i}. {experiment}: ACC={avg_acc}")
    
    # 获取基准表现
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_acc = baseline_row['Avg_ACC']
    print(f"\n📈 基准表现 (Baseline): ACC={baseline_acc}")
    
    # 计算相对于基准的改进
    best_acc = float(df.iloc[0]['Avg_ACC'])
    improvement = ((best_acc - float(baseline_acc)) / float(baseline_acc)) * 100
    print(f"🚀 最佳ACC改进: +{improvement:.1f}% (相对于基准)")
    
    # 分析每个测试集的最佳表现
    print(f"\n📋 各测试集最佳ACC表现:")
    test_sets = ['Test-1', 'Test-2', 'Test-3']
    for test_set in test_sets:
        best_exp = None
        best_acc = 0
        for _, row in df.iterrows():
            acc_val = row[f'{test_set}_ACC']
            if acc_val != "N/A":
                acc_num = float(acc_val)
                if acc_num > best_acc:
                    best_acc = acc_num
                    best_exp = row['Experiment']
        if best_exp:
            print(f"   {test_set}: {best_exp} (ACC={best_acc:.4f})")
    
    # 分析稳定性（标准差）
    print(f"\n📈 稳定性分析 (各测试集ACC的标准差，越小越稳定):")
    for _, row in df.iterrows():
        experiment = row['Experiment']
        acc_values = []
        for test_set in test_sets:
            acc_val = row[f'{test_set}_ACC']
            if acc_val != "N/A":
                acc_values.append(float(acc_val))
        
        if len(acc_values) >= 2:
            import numpy as np
            std_dev = np.std(acc_values)
            print(f"   {experiment}: 标准差={std_dev:.4f}")
    
    print("=" * 80)

def main():
    """主函数"""
    df = create_acc_only_ranking()
    print_acc_ranking(df)
    print_acc_analysis(df)
    
    # 保存到CSV文件
    output_file = "elasticity_relative_acc_only_ranking.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ ACC排名表格已保存到: {output_file}")

if __name__ == "__main__":
    main() 