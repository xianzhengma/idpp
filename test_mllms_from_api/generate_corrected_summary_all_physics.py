#!/usr/bin/env python3
"""
生成修正后的所有物理属性汇总报告
整合friction、viscosity、elasticity三种物理属性的实验结果
注意：已对elasticity应用GT值<0.1的过滤条件
"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_corrected_comprehensive_summary():
    """
    生成修正后的所有物理属性综合汇总
    """
    
    print("="*100)
    print("🔬 GEMINI 2.5 PRO 物理属性预测消融实验 - 修正后综合分析报告")
    print("⚠️  注意：已对Elasticity应用GT值<0.1的数据质量过滤")
    print("="*100)
    
    print("\n📊 实验设置:")
    print("   • 模型: Gemini 2.5 Pro")
    print("   • 物理属性: Friction, Viscosity, Elasticity")
    print("   • 任务类型: Absolute Prediction & Relative Comparison")
    print("   • 样本数量: 100 samples per experiment (分成5份，每份20个)")
    print("   • 测试集: Test-1, Test-2, Test-3")
    print("   • 🔧 数据质量控制: Elasticity GT值<0.1的样本已被过滤")
    
    # 修正后的绝对值预测结果汇总
    print(f"\n{'='*100}")
    print("Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties (Corrected)")
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
                prop_display += " 🔧"  # 标记已过滤
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 修正后的相对比较结果汇总
    print(f"\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties (Corrected)")
    print("="*100)
    
    relative_results = {
        'Elasticity': {
            'Baseline': [0.34, 0.49, 0.55, 0.46],
            '+ Black Frames in Between': [0.51, 0.50, 0.63, 0.54],
            '+ Frame Index Provided': [0.47, 0.57, 0.52, 0.52],
            '+ Few-Shot Examples': [0.37, 0.40, 0.52, 0.43],
            '+ Oracle Estimation Teaching': [0.58, 0.54, 0.54, 0.55]
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
                prop_display += " 🔧"  # 标记已过滤
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 最佳策略分析
    print(f"\n{'='*100}")
    print("🏆 最佳策略分析 (修正后结果)")
    print("="*100)
    
    print(f"\n📈 绝对值预测任务 (Pearson Correlation):")
    print(f"   🥇 Elasticity: + Few-Shot Examples (0.33) - 比Baseline提升230.0% 🔧")
    print(f"   🥇 Friction:    Baseline (0.11) - 基准表现最佳")  
    print(f"   🥇 Viscosity:   Baseline (0.17) - 基准表现最佳")
    
    print(f"\n📈 相对比较任务 (AUC):")
    print(f"   🥇 Elasticity: + Oracle Estimation Teaching (0.55) - 比Baseline提升19.6% 🔧")
    print(f"   🥇 Friction:    + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 比Baseline提升9.5%")
    print(f"   🥇 Viscosity:   + Few-Shot Examples (0.79) - 比Baseline提升1.3%")
    
    # 数据质量过滤影响分析
    print(f"\n{'='*100}")
    print("🔧 数据质量过滤影响分析")
    print("="*100)
    
    print(f"\n📊 Elasticity 过滤统计:")
    print(f"   绝对值任务:")
    print(f"   • 每个实验移除 3-5个 GT值<0.1的样本")
    print(f"   • 占总样本的 3-5%，影响相对较小")
    
    print(f"\n   相对比较任务:")
    print(f"   • Test-1: 移除 19个样本对 (约19%)")
    print(f"   • Test-2: 移除 63个样本对 (约63%!) - 显著影响")
    print(f"   • Test-3: 过滤影响较小")
    
    print(f"\n⚠️  过滤效果:")
    print(f"   • Elasticity相对比较的AUC从之前的0.65降至0.55")
    print(f"   • 说明GT值很小的样本往往更容易区分")
    print(f"   • 过滤后的结果更能反映模型在有效数据上的真实表现")
    
    # 策略有效性分析
    print(f"\n{'='*100}")
    print("💡 策略有效性跨属性分析 (基于修正后数据)")
    print("="*100)
    
    print(f"\n🔍 策略表现总结:")
    print(f"   1. + Few-Shot Examples:")
    print(f"      ✅ 在Elasticity绝对值预测上表现最佳 (0.33) 🔧")
    print(f"      ✅ 在Viscosity相对比较上表现最佳 (0.79)")
    print(f"      ❌ 在Friction任务上表现一般")
    print(f"      📈 整体最全能的策略")
    
    print(f"\n   2. + Oracle Estimation Teaching:")
    print(f"      ✅ 在Elasticity相对比较上表现最佳 (0.55) 🔧")
    print(f"      ✅ 在Friction相对比较上表现最佳 (0.69)")
    print(f"      ⚖️  在绝对值预测上表现中等")
    print(f"      📈 相对比较任务的专家")
    
    print(f"\n   3. + Frame Index Provided:")
    print(f"      ✅ 在Friction相对比较上表现最佳 (0.69)")
    print(f"      ✅ 在Elasticity绝对值预测上表现良好 (0.32) 🔧")
    print(f"      ⚖️  整体表现稳定")
    
    print(f"\n   4. Baseline:")
    print(f"      ✅ 在Friction和Viscosity绝对值预测上表现最佳")
    print(f"      ❌ 在Elasticity任务上表现较差 🔧")
    print(f"      💡 对简单属性有效，复杂属性需要增强策略")
    
    # 实践建议
    print(f"\n{'='*100}")
    print("🎯 实践建议 (基于修正后数据)")
    print("="*100)
    
    print(f"\n💼 根据任务类型和数据质量选择策略:")
    
    print(f"\n   📏 绝对值预测:")
    print(f"      - Elasticity (复杂) → 使用 Few-Shot Examples (0.33)")
    print(f"      - Friction/Viscosity (简单) → 使用 Baseline")
    
    print(f"\n   🔄 相对比较:")
    print(f"      - Elasticity (复杂) → 使用 Oracle Estimation Teaching (0.55)")
    print(f"      - Friction → 使用 Oracle/Frame Index (0.69)")
    print(f"      - Viscosity → 使用 Few-Shot Examples (0.79)")
    
    print(f"\n   🔧 数据质量考虑:")
    print(f"      • 对于有大量低质量样本的数据集，先进行数据过滤")
    print(f"      • Elasticity这类物理属性需要特别注意GT值的有效性")
    print(f"      • 过滤后可能需要重新评估策略有效性")
    
    print(f"\n🚀 高级建议:")
    print(f"   • Few-Shot Examples: 最全能，适合复杂属性")
    print(f"   • Oracle Teaching: 相对比较专家")
    print(f"   • 简单属性慎用过度复杂的prompting")
    print(f"   • 建议ensemble Different策略以获得鲁棒性")
    
    print(f"\n{'='*100}")
    print("✅ 修正后综合分析完成")
    print("="*100)
    
    # 生成修正后的CSV总结
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建绝对值预测汇总
    abs_data = []
    for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    abs_filename = f"gemini_absolute_all_physics_corrected_summary_{timestamp}.csv"
    abs_df.to_csv(abs_filename, index=False)
    
    # 创建相对比较汇总
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f"gemini_relative_all_physics_corrected_summary_{timestamp}.csv"
    rel_df.to_csv(rel_filename, index=False)
    
    print(f"\n📁 修正后绝对值预测汇总表格已保存为: {abs_filename}")
    print(f"📁 修正后相对比较汇总表格已保存为: {rel_filename}")
    
    print(f"\n🎉 修正后的分析文件已生成，包括:")
    print(f"   • 3个单属性详细表格 (elasticity🔧, friction, viscosity)")
    print(f"   • 2个修正后综合汇总表格")
    print(f"   • 数据质量过滤统计和影响分析")
    print(f"   • 基于修正数据的策略建议")
    
    print(f"\n🔧 注意：Elasticity结果已应用GT值<0.1过滤，更能反映真实模型性能")
    print(f"{'='*100}")


if __name__ == "__main__":
    generate_corrected_comprehensive_summary()