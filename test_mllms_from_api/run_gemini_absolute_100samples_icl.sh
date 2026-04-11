#!/bin/bash
# Gemini-2.5-Pro 绝对setting测试脚本
# 样本数: 100个绝对样本
# 策略: ICL策略 (In-Context Learning)
# 物理属性: friction, viscosity, elasticity

echo "开始执行 Gemini-2.5-Pro 绝对setting测试..."
echo "模型: gemini-2.5-pro"
echo "设置: 绝对setting (100个样本)"
echo "策略: ICL策略 (In-Context Learning)"
echo "物理属性: friction, viscosity, elasticity"
echo "预计总时间: 约3小时"
echo ""

# =============================================================================
# Friction 绝对测试
# =============================================================================

echo "=== 测试 1/3: Friction 绝对测试 ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties friction \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Friction 绝对测试完成，结果保存在: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
# Viscosity 绝对测试
# =============================================================================

echo "=== 测试 2/3: Viscosity 绝对测试 ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties viscosity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Viscosity 绝对测试完成，结果保存在: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
# Elasticity 绝对测试
# =============================================================================

echo "=== 测试 3/3: Elasticity 绝对测试 ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties elasticity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Elasticity 绝对测试完成，结果保存在: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
# 完成总结
# =============================================================================

echo "🎉 Gemini-2.5-Pro 绝对setting测试完成！"
echo ""
echo "测试总结:"
echo "1. Friction 绝对 (100样本)   -> gemini_physics_analysis_results_absolute_100samples_icl"
echo "2. Viscosity 绝对 (100样本)  -> gemini_physics_analysis_results_absolute_100samples_icl"
echo "3. Elasticity 绝对 (100样本) -> gemini_physics_analysis_results_absolute_100samples_icl"
echo ""
echo "结果目录列表:"
ls -d gemini_physics_analysis_results_absolute_100samples_icl 2>/dev/null | sort
echo ""
echo "所有测试已完成！" 