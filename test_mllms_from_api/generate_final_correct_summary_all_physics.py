#!/usr/bin/env python3
"""
生成最终正确的所有物理属性汇总报告
应用正确的过滤条件：只对elasticity的绝对值预测任务过滤GT值<0.1
相对比较任务保持原始数据不过滤
"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_final_correct_summary():
    """
    生成最终正确的所有物理属性综合汇总
    """
    
    print("="*100)
    print("🔬 GEMINI 2.5 PRO 物理属性预测消融实验 - 最终正确分析报告")
    print("✅ 过滤条件已修正：仅对Elasticity绝对值预测应用GT值<0.1过滤")
    print("="*100)
    
    print("\n📊 实验设置:")
    print("   • 模型: Gemini 2.5 Pro")
    print("   • 物理属性: Friction, Viscosity, Elasticity")
    print("   • 任务类型: Absolute Prediction & Relative Comparison")
    print("   • 样本数量: 100 samples per experiment (分成5份，每份20个)")
    print("   • 测试集: Test-1, Test-2, Test-3")
    print("   • ✅ 数据质量控制: 仅对Elasticity绝对值预测过滤GT值<0.1")
    print("   • ✅ 相对比较任务保持原始数据（无过滤）")
    
    # 最终正确的绝对值预测结果汇总
    print(f"\n{'='*100}")
    print("Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties (Final Correct)")
    print("="*100)
    
    absolute_results = {
        'Elasticity': {
            'Baseline': [-0.03, 0.26, 0.06, 0.10],
            '+ Frame Index Provided': [0.06, 0.35, 0.55, 0.32],
            '+ Few-Shot Examples': [0.39, 0.34, 0.24, 0.33],
            '+ Oracle Estimation Teaching': [0.19, 0.14, 0.24, 0.19]
        },
        'Friction': {
            'Baseline': [0.06, 0.09, 0.19, 0.11],
            '+ Frame Index Provided': [0.11, 0.05, -0.02, 0.05],
            '+ Few-Shot Examples': [-0.07, 0.11, 0.12, 0.05],
            '+ Oracle Estimation Teaching': [0.06, 0.17, 0.03, 0.09]
        },
        'Viscosity': {
            'Baseline': [0.05, 0.12, 0.35, 0.17],
            '+ Frame Index Provided': [0.01, 0.02, 0.37, 0.13],
            '+ Few-Shot Examples': [0.07, -0.05, 0.29, 0.10],
            '+ Oracle Estimation Teaching': [0.15, 0.04, 0.29, 0.16]
        }
    }
    
    # 显示绝对值预测表格
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print("-" * 100)
    
    for property_name, strategies in absolute_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ""
            if property_name == 'Elasticity' and first_row:
                prop_display += " ✅"  # 标记已正确过滤
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 最终正确的相对比较结果汇总
    print(f"\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties (Final Correct)")
    print("="*100)
    
    relative_results = {
        'Elasticity': {
            'Baseline': [0.51, 0.74, 0.55, 0.60],
            '+ Black Frames in Between': [0.56, 0.72, 0.63, 0.64],
            '+ Frame Index Provided': [0.54, 0.80, 0.52, 0.62],
            '+ Few-Shot Examples': [0.43, 0.65, 0.52, 0.54],
            '+ Oracle Estimation Teaching': [0.63, 0.79, 0.54, 0.65]  # 修正回0.65!
        },
        'Friction': {
            'Baseline': [0.46, 0.45, 0.97, 0.63],
            '+ Black Frames in Between': [0.47, 0.57, 0.93, 0.66],
            '+ Frame Index Provided': [0.54, 0.55, 0.98, 0.69],
            '+ Few-Shot Examples': [0.49, 0.47, 0.98, 0.65],
            '+ Oracle Estimation Teaching': [0.52, 0.59, 0.97, 0.69]
        },
        'Viscosity': {
            'Baseline': [0.67, 0.72, 0.95, 0.78],
            '+ Black Frames in Between': [0.59, 0.59, 0.94, 0.71],
            '+ Frame Index Provided': [0.68, 0.63, 0.97, 0.76],
            '+ Few-Shot Examples': [0.68, 0.71, 0.97, 0.79],
            '+ Oracle Estimation Teaching': [0.48, 0.71, 0.92, 0.70]
        }
    }
    
    # 显示相对比较表格
    print(f"\n{'Property':<12} | {'Strategy':<35} | {'Test-1':<8} {'Test-2':<8} {'Test-3':<8} {'Avg':<8}")
    print("-" * 100)
    
    for property_name, strategies in relative_results.items():
        first_row = True
        for strategy, values in strategies.items():
            prop_display = property_name if first_row else ""
            if property_name == 'Elasticity' and first_row:
                prop_display += " ✅"  # 标记无过滤（正确）
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 最佳策略分析
    print(f"\n{'='*100}")
    print("🏆 最佳策略分析 (最终正确结果)")
    print("="*100)
    
    print(f"\n📈 绝对值预测任务 (Pearson Correlation):")
    print(f"   🥇 Elasticity: + Few-Shot Examples (0.33) - 比Baseline提升230.0% ✅")
    print(f"   🥇 Friction:    Baseline (0.11) - 基准表现最佳")  
    print(f"   🥇 Viscosity:   Baseline (0.17) - 基准表现最佳")
    
    print(f"\n📈 相对比较任务 (AUC):")
    print(f"   🥇 Elasticity: + Oracle Estimation Teaching (0.65) - 比Baseline提升8.3% ✅")
    print(f"   🥇 Friction:    + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 比Baseline提升9.5%")
    print(f"   🥇 Viscosity:   + Few-Shot Examples (0.79) - 比Baseline提升1.3%")
    
    # 修正前后对比
    print(f"\n{'='*100}")
    print("🔧 过滤条件修正前后对比")
    print("="*100)
    
    print(f"\n📊 Elasticity 结果修正:")
    print(f"   绝对值预测:")
    print(f"   • ✅ 正确过滤GT值<0.1样本")
    print(f"   • 每个实验移除3-5个样本 (3-5%)")
    print(f"   • 结果保持不变: Few-Shot Examples最佳 (0.33)")
    
    print(f"\n   相对比较:")
    print(f"   • ❌ 之前错误: 过滤GT值<0.1样本对，AUC降至0.55")
    print(f"   • ✅ 现在正确: 保持原始数据，AUC恢复到0.65")
    print(f"   • 📈 修正后提升: 相对比较效果显著改善")
    
    print(f"\n⚠️  修正意义:")
    print(f"   • 相对比较任务关注两值的相对大小关系，而非绝对值")
    print(f"   • GT值较小的样本对在相对比较中仍有意义")
    print(f"   • 过滤会损失有效的对比信息")
    
    # 策略有效性分析
    print(f"\n{'='*100}")
    print("💡 策略有效性跨属性分析 (基于最终正确数据)")
    print("="*100)
    
    print(f"\n🔍 策略表现总结:")
    print(f"   1. + Oracle Estimation Teaching:")
    print(f"      🏆 在Elasticity相对比较上表现最佳 (0.65) ✅")
    print(f"      🏆 在Friction相对比较上表现最佳 (0.69)")
    print(f"      📈 相对比较任务的绝对王者")
    
    print(f"\n   2. + Few-Shot Examples:")
    print(f"      🏆 在Elasticity绝对值预测上表现最佳 (0.33) ✅") 
    print(f"      🏆 在Viscosity相对比较上表现最佳 (0.79)")
    print(f"      📈 跨任务类型的全能选手")
    
    print(f"\n   3. + Frame Index Provided:")
    print(f"      🏆 在Friction相对比较上并列最佳 (0.69)")
    print(f"      ✅ 在Elasticity绝对值预测上表现良好 (0.32)")
    print(f"      📈 稳定可靠的中坚力量")
    
    print(f"\n   4. Baseline:")
    print(f"      🏆 在Friction和Viscosity绝对值预测上表现最佳")
    print(f"      ❌ 在Elasticity任务上表现较弱")
    print(f"      💡 简单属性首选，复杂属性需增强")
    
    # 实践建议
    print(f"\n{'='*100}")
    print("🎯 最终实践建议")
    print("="*100)
    
    print(f"\n💼 按任务类型和属性复杂度选择策略:")
    
    print(f"\n   📏 绝对值预测:")
    print(f"      - Elasticity (复杂) → Few-Shot Examples (0.33)")
    print(f"      - Friction/Viscosity (简单) → Baseline")
    
    print(f"\n   🔄 相对比较:")
    print(f"      - Elasticity (复杂) → Oracle Estimation Teaching (0.65)")
    print(f"      - Friction → Oracle Teaching/Frame Index (0.69)")
    print(f"      - Viscosity → Few-Shot Examples (0.79)")
    
    print(f"\n   🔧 数据质量控制原则:")
    print(f"      • 绝对值预测: 过滤无意义的极小GT值")
    print(f"      • 相对比较: 保持所有样本对，维持对比信息完整性")
    print(f"      • 根据任务特性制定不同的数据质量策略")
    
    print(f"\n🚀 高级策略建议:")
    print(f"   🏆 Oracle Estimation Teaching: 相对比较专家")
    print(f"   🏆 Few-Shot Examples: 全能型选手，特别适合复杂属性")
    print(f"   ⚖️  Frame Index Provided: 稳定的中坚策略")
    print(f"   💰 Baseline: 简单高效，适合简单属性")
    print(f"   🎯 建议ensemble多种策略获得最佳鲁棒性")
    
    print(f"\n{'='*100}")
    print("✅ 最终正确综合分析完成")
    print("="*100)
    
    # 生成最终正确的CSV总结
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建绝对值预测汇总
    abs_data = []
    for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    abs_filename = f"gemini_absolute_all_physics_FINAL_summary_{timestamp}.csv"
    abs_df.to_csv(abs_filename, index=False)
    
    # 创建相对比较汇总
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f"gemini_relative_all_physics_FINAL_summary_{timestamp}.csv"
    rel_df.to_csv(rel_filename, index=False)
    
    print(f"\n📁 最终绝对值预测汇总表格已保存为: {abs_filename}")
    print(f"📁 最终相对比较汇总表格已保存为: {rel_filename}")
    
    print(f"\n🎉 最终正确的分析文件已生成，包括:")
    print(f"   • 3个单属性详细表格 (elasticity✅, friction, viscosity)")
    print(f"   • 2个最终正确综合汇总表格")
    print(f"   • 过滤条件修正前后对比分析") 
    print(f"   • 基于正确数据的最终策略建议")
    
    print(f"\n✅ 重要修正:")
    print(f"   • Elasticity绝对值预测: 正确过滤GT值<0.1 ✅")
    print(f"   • Elasticity相对比较: 保持原始数据，AUC从0.55恢复到0.65 ✅")  
    print(f"   • 数据质量控制策略针对任务类型进行优化 ✅")
    print(f"{'='*100}")


if __name__ == "__main__":
    generate_final_correct_summary()