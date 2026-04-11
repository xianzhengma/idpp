#!/bin/bash
# Qwen-VL-Max 相对setting测试脚本
# 样本数: 100个相对配对样本
# 策略: Oracle策略
# 物理属性: friction, viscosity, elasticity

echo "开始执行 Qwen-VL-Max 相对setting测试..."
echo "模型: qwen-vl-max"
echo "设置: 相对setting (100个样本)"
echo "策略: Oracle策略"
echo "物理属性: friction, viscosity, elasticity"
echo "预计总时间: 约3小时"
echo ""

# =============================================================================
# Friction 相对测试
# =============================================================================

echo "=== 测试 1/3: Friction 相对测试 ==="
python run_gpt_physics_tests.py \
    --model-version qwen-vl-max \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties friction \
    --follow-oracle-test \
    --output-dir qwen_physics_analysis_results_relative_100samples_oracle \
    --auto-confirm
echo "✓ Friction 相对测试完成，结果保存在: qwen_physics_analysis_results_relative_100samples_oracle"
echo ""

# =============================================================================
# Viscosity 相对测试
# =============================================================================

echo "=== 测试 2/3: Viscosity 相对测试 ==="
python run_gpt_physics_tests.py \
    --model-version qwen-vl-max \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties viscosity \
    --follow-oracle-test \
    --output-dir qwen_physics_analysis_results_relative_100samples_oracle \
    --auto-confirm
echo "✓ Viscosity 相对测试完成，结果保存在: qwen_physics_analysis_results_relative_100samples_oracle"
echo ""

# =============================================================================
# Elasticity 相对测试
# =============================================================================

echo "=== 测试 3/3: Elasticity 相对测试 ==="
python run_gpt_physics_tests.py \
    --model-version qwen-vl-max \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties elasticity \
    --follow-oracle-test \
    --output-dir qwen_physics_analysis_results_relative_100samples_oracle \
    --auto-confirm
echo "✓ Elasticity 相对测试完成，结果保存在: qwen_physics_analysis_results_relative_100samples_oracle"
echo ""

# =============================================================================
# 完成总结
# =============================================================================

echo "🎉 Qwen-VL-Max 相对setting测试完成！"
echo ""
echo "测试总结:"
echo "1. Friction 相对 (100样本)   -> qwen_physics_analysis_results_relative_100samples_oracle"
echo "2. Viscosity 相对 (100样本)  -> qwen_physics_analysis_results_relative_100samples_oracle"
echo "3. Elasticity 相对 (100样本) -> qwen_physics_analysis_results_relative_100samples_oracle"
echo ""
echo "结果目录列表:"
ls -d qwen_physics_analysis_results_relative_100samples_oracle 2>/dev/null | sort
echo ""
echo "所有测试已完成！" 