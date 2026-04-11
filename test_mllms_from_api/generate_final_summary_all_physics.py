#!/usr/bin/env python3
"""
生成所有物理属性的最终汇总报告
整合friction、viscosity、elasticity三种物理属性的实验结果
创建类似论文的综合对比表格
"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_comprehensive_summary():
    """
    生成所有物理属性的综合汇总
    """
    
    print("="*100)
    print("🔬 GEMINI 2.5 PRO 物理属性预测消融实验 - 综合分析报告")
    print("="*100)
    
    print("\n📊 实验设置:")
    print("   • 模型: Gemini 2.5 Pro")
    print("   • 物理属性: Friction, Viscosity, Elasticity")
    print("   • 任务类型: Absolute Prediction & Relative Comparison")
    print("   • 样本数量: 100 samples per experiment (分成5份，每份20个)")
    print("   • 测试集: Test-1, Test-2, Test-3")
    
    # 绝对值预测结果汇总
    print(f"\n{'='*100}")
    print("Table 1: ABSOLUTE PREDICTION RESULTS - All Physics Properties")
    print("="*100)
    
    absolute_results = {
        'Elasticity': {
            'Baseline': [0.03, 0.26, 0.03, 0.10],
            '+ Frame Index Provided': [0.14, 0.34, 0.46, 0.31],
            '+ Few-Shot Examples': [0.46, 0.37, 0.18, 0.34],
            '+ Oracle Estimation Teaching': [0.30, 0.27, 0.19, 0.25]
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
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 相对比较结果汇总
    print(f"\nTable 2: RELATIVE COMPARISON RESULTS - All Physics Properties")
    print("="*100)
    
    relative_results = {
        'Elasticity': {
            'Baseline': [0.51, 0.74, 0.55, 0.60],
            '+ Black Frames in Between': [0.56, 0.72, 0.63, 0.64],
            '+ Frame Index Provided': [0.54, 0.80, 0.52, 0.62],
            '+ Few-Shot Examples': [0.43, 0.65, 0.52, 0.54],
            '+ Oracle Estimation Teaching': [0.63, 0.79, 0.54, 0.65]
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
            print(f"{prop_display:<12} | {strategy:<35} | {values[0]:>7.2f} {values[1]:>8.2f} {values[2]:>8.2f} {values[3]:>8.2f}")
            first_row = False
        print("-" * 100)
    
    # 最佳策略分析
    print(f"\n{'='*100}")
    print("🏆 最佳策略分析")
    print("="*100)
    
    print(f"\n📈 绝对值预测任务 (Pearson Correlation):")
    print(f"   🥇 Elasticity: + Few-Shot Examples (0.34) - 比Baseline提升240.0%")
    print(f"   🥇 Friction:    Baseline (0.11) - 基准表现最佳")  
    print(f"   🥇 Viscosity:   Baseline (0.17) - 基准表现最佳")
    
    print(f"\n📈 相对比较任务 (AUC):")
    print(f"   🥇 Elasticity: + Oracle Estimation Teaching (0.65) - 比Baseline提升8.3%")
    print(f"   🥇 Friction:    + Frame Index Provided & + Oracle Estimation Teaching (0.69) - 比Baseline提升9.5%")
    print(f"   🥇 Viscosity:   + Few-Shot Examples (0.79) - 比Baseline提升1.3%")
    
    # 策略有效性分析
    print(f"\n{'='*100}")
    print("💡 策略有效性跨属性分析")
    print("="*100)
    
    print(f"\n🔍 策略表现总结:")
    print(f"   1. + Few-Shot Examples:")
    print(f"      ✅ 在Elasticity绝对值预测上表现最佳 (0.34)")
    print(f"      ✅ 在Viscosity相对比较上表现最佳 (0.79)")
    print(f"      ❌ 在Friction任务上表现一般")
    
    print(f"\n   2. + Oracle Estimation Teaching:")
    print(f"      ✅ 在Elasticity相对比较上表现最佳 (0.65)")
    print(f"      ✅ 在Friction相对比较上表现最佳 (0.69)")
    print(f"      ❌ 在绝对值预测上表现中等")
    
    print(f"\n   3. + Frame Index Provided:")
    print(f"      ✅ 在Friction相对比较上表现最佳 (0.69)")
    print(f"      ✅ 在Elasticity绝对值预测上表现良好 (0.31)")
    print(f"      ❌ 整体表现稳定但不突出")
    
    print(f"\n   4. + Black Frames in Between:")
    print(f"      ✅ 仅在相对比较任务中可用")
    print(f"      ⚖️  表现中等，未在任何属性上达到最佳")
    
    print(f"\n   5. Baseline:")
    print(f"      ✅ 在Friction和Viscosity绝对值预测上表现最佳")
    print(f"      ❌ 在Elasticity任务上表现较差")
    
    # 数据集难度分析
    print(f"\n{'='*100}")
    print("📊 数据集难度分析")
    print("="*100)
    
    print(f"\n🎯 测试集表现规律:")
    print(f"   • Test-3 在大多数任务上表现最好，特别是相对比较任务")
    print(f"   • Friction相对比较在Test-3上AUC普遍>0.92，显示任务相对简单")
    print(f"   • Viscosity相对比较在Test-3上AUC普遍>0.92，显示任务相对简单")
    print(f"   • Elasticity任务难度相对较高，各测试集表现差异较大")
    
    # 实践建议
    print(f"\n{'='*100}")
    print("🎯 实践建议")
    print("="*100)
    
    print(f"\n💼 根据任务类型选择策略:")
    print(f"   📏 绝对值预测:")
    print(f"      - Elasticity → 使用 Few-Shot Examples")
    print(f"      - Friction/Viscosity → 使用 Baseline (简单有效)")
    
    print(f"\n🔄 相对比较:")
    print(f"      - Elasticity → 使用 Oracle Estimation Teaching")
    print(f"      - Friction → 使用 Oracle Estimation Teaching 或 Frame Index Provided")
    print(f"      - Viscosity → 使用 Few-Shot Examples")
    
    print(f"\n🚀 进阶建议:")
    print(f"   • 对于新的物理属性，建议先尝试Baseline确定基准性能")
    print(f"   • 对于复杂属性（如Elasticity），Few-Shot Examples和Oracle Teaching更有效")
    print(f"   • 对于简单属性（如Friction/Viscosity），过度的prompting可能带来负面效果")
    print(f"   • 考虑ensemble不同策略以获得更稳定的性能")
    
    print(f"\n{'='*100}")
    print("✅ 综合分析完成")
    print("="*100)
    
    # 生成CSV总结
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建绝对值预测汇总
    abs_data = []
    for prop, strategies in absolute_results.items():
        for strategy, values in strategies.items():
            abs_data.append([prop, strategy] + values)
    
    abs_df = pd.DataFrame(abs_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    abs_filename = f"gemini_absolute_all_physics_summary_{timestamp}.csv"
    abs_df.to_csv(abs_filename, index=False)
    
    # 创建相对比较汇总
    rel_data = []
    for prop, strategies in relative_results.items():
        for strategy, values in strategies.items():
            rel_data.append([prop, strategy] + values)
    
    rel_df = pd.DataFrame(rel_data, columns=['Property', 'Strategy', 'Test-1', 'Test-2', 'Test-3', 'Avg'])
    rel_filename = f"gemini_relative_all_physics_summary_{timestamp}.csv"
    rel_df.to_csv(rel_filename, index=False)
    
    print(f"\n📁 绝对值预测汇总表格已保存为: {abs_filename}")
    print(f"📁 相对比较汇总表格已保存为: {rel_filename}")
    
    print(f"\n🎉 所有分析文件已生成，包括:")
    print(f"   • 3个绝对值预测单属性表格 (elasticity, friction, viscosity)")
    print(f"   • 3个相对比较单属性表格 (elasticity, friction, viscosity)")
    print(f"   • 2个综合汇总表格 (absolute & relative)")
    print(f"   • 2个详细实验数据文件")
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    generate_comprehensive_summary()