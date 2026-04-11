#!/usr/bin/env python3
"""
展示Elasticity Relative实验的完整排名
"""

import pandas as pd

def create_complete_ranking():
    """创建完整的排名表格"""
    
    # 基于实际实验结果文件的完整数据
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
        'Test-1_AUC': [
            0.8389,  # Baseline
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
        'Test-2_AUC': [
            0.7583,  # Baseline
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
        'Test-3_AUC': [
            0.8125,  # Baseline
            0.5417,  # + Concat+ICL
            0.5000,  # + Concat+Oracle
            0.5441,  # + Concat+FrameIndex
            0.5625,  # + ICL+Oracle
            0.6458,  # + ICL+FrameIndex
            1.0000,  # + Oracle+FrameIndex
            0.8571,  # + Concat+ICL+Oracle
            None,    # + Concat+Oracle+FrameIndex (无结果)
            0.2857   # + ICL+Oracle+FrameIndex
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
    
    # 计算平均值
    df['Avg_AUC'] = df[['Test-1_AUC', 'Test-2_AUC', 'Test-3_AUC']].apply(
        lambda x: x.mean() if x.notna().any() else None, axis=1
    )
    df['Avg_ACC'] = df[['Test-1_ACC', 'Test-2_ACC', 'Test-3_ACC']].apply(
        lambda x: x.mean() if x.notna().any() else None, axis=1
    )
    
    # 按平均ACC分数排序（降序）
    df = df.sort_values('Avg_ACC', ascending=False, na_position='last')
    
    # 格式化数值为4位小数
    for col in ['Test-1_AUC', 'Test-1_ACC', 'Test-2_AUC', 'Test-2_ACC', 'Test-3_AUC', 'Test-3_ACC', 'Avg_AUC', 'Avg_ACC']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    
    return df

def print_complete_ranking(df):
    """打印完整的排名表格"""
    
    print("╔" + "═" * 120 + "╗")
    print("║" + " " * 45 + "Elasticity Relative (20 Examples) - Complete Ranking" + " " * 45 + "║")
    print("╠" + "═" * 120 + "╣")
    print("║ Rank │ Experiment                │ Test-1_AUC │ Test-1_ACC │ Test-2_AUC │ Test-2_ACC │ Test-3_AUC │ Test-3_ACC │ Avg_AUC   │ Avg_ACC   ║")
    print("╟" + "─" * 120 + "╢")
    
    # 打印数据行
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
        
        # 为前三名添加特殊标记
        if rank == 1:
            rank_str = "🥇"
        elif rank == 2:
            rank_str = "🥈"
        elif rank == 3:
            rank_str = "🥉"
        else:
            rank_str = f"{rank:2d}"
        
        print(f"║ {rank_str:<4} │ {experiment:<25} │ {test1_auc:<10} │ {test1_acc:<10} │ {test2_auc:<10} │ {test2_acc:<10} │ {test3_auc:<10} │ {test3_acc:<10} │ {avg_auc:<9} │ {avg_acc:<9} ║")
    
    print("╚" + "═" * 120 + "╝")

def print_detailed_analysis(df):
    """打印详细分析"""
    print("\n" + "=" * 120)
    print("📊 详细分析")
    print("=" * 120)
    
    # 获取前三名
    top3 = df.head(3)
    print("🏆 前三名表现 (按ACC排序):")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        avg_auc = row['Avg_AUC']
        print(f"   {i}. {experiment}: ACC={avg_acc}, AUC={avg_auc}")
    
    # 获取基准表现
    baseline_row = df[df['Experiment'] == 'Baseline'].iloc[0]
    baseline_acc = baseline_row['Avg_ACC']
    baseline_auc = baseline_row['Avg_AUC']
    print(f"\n📈 基准表现 (Baseline): ACC={baseline_acc}, AUC={baseline_auc}")
    
    # 计算相对于基准的改进
    best_acc = float(df.iloc[0]['Avg_ACC'])
    improvement_acc = ((best_acc - float(baseline_acc)) / float(baseline_acc)) * 100
    print(f"🚀 最佳ACC改进: +{improvement_acc:.1f}% (相对于基准)")
    
    # 按AUC排序
    df_auc_sorted = df.copy()
    df_auc_sorted['Avg_AUC_num'] = df_auc_sorted['Avg_AUC'].apply(lambda x: float(x) if x != "N/A" else 0)
    df_auc_sorted = df_auc_sorted.sort_values('Avg_AUC_num', ascending=False)
    
    print(f"\n🏆 前三名表现 (按AUC排序):")
    for i, (_, row) in enumerate(df_auc_sorted.head(3).iterrows(), 1):
        experiment = row['Experiment']
        avg_acc = row['Avg_ACC']
        avg_auc = row['Avg_AUC']
        print(f"   {i}. {experiment}: AUC={avg_auc}, ACC={avg_acc}")
    
    # 分析每个测试集的最佳表现
    print(f"\n📋 各测试集最佳表现:")
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
    
    print("=" * 120)

def main():
    """主函数"""
    df = create_complete_ranking()
    print_complete_ranking(df)
    print_detailed_analysis(df)
    
    # 保存到CSV文件
    output_file = "elasticity_relative_complete_ranking.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 完整排名表格已保存到: {output_file}")

if __name__ == "__main__":
    main() 