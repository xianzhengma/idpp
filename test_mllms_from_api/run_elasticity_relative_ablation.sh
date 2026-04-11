#!/bin/bash
# Elasticity Relative Ablation Study - Basic Experiments Only
# 基础实验：baseline + 4个单个功能实验
# 每个实验使用20对样本

echo "开始执行 Elasticity Relative Ablation Study - 基础实验..."
echo "执行 baseline + 4个单个功能实验"
echo "总共5个实验，每个实验20对样本"
echo "预计总时间: 2.5 小时"
echo ""

# =============================================================================
# 基础实验 (5个)
# =============================================================================

# echo "=== 基础实验 ==="
# echo ""

# echo "实验 1/5: Baseline"
# echo "描述: 基线实验（无任何额外功能）"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 0 \
#     --relative-pairs 20 \
#     --properties elasticity \
#     --output-dir gemini_physics_analysis_results_elasticity_relative_20samples \
#     --auto-confirm
# echo "✓ 实验1完成，结果保存在: gemini_physics_analysis_results_elasticity_relative_20samples"
# echo ""

# echo "实验 2/5: + Concat"
# echo "描述: 基线 + 拼接模式"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 0 \
#     --relative-pairs 20 \
#     --properties elasticity \
#     --concat-mode \
#     --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_concat \
#     --auto-confirm
# echo "✓ 实验2完成，结果保存在: gemini_physics_analysis_results_elasticity_relative_20samples_concat"
# echo ""

echo "实验 3/5: + ICL"
echo "描述: 基线 + ICL示例"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_icl \
    --auto-confirm
echo "✓ 实验3完成，结果保存在: gemini_physics_analysis_results_elasticity_relative_20samples_icl"
echo ""

echo "实验 4/5: + Oracle"
echo "描述: 基线 + Oracle指导"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_oracle \
    --auto-confirm
echo "✓ 实验4完成，结果保存在: gemini_physics_analysis_results_elasticity_relative_20samples_oracle"
echo ""

echo "实验 5/5: + FrameIndex"
echo "描述: 基线 + 帧索引"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --frame-index \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_frameindex \
    --auto-confirm
echo "✓ 实验5完成，结果保存在: gemini_physics_analysis_results_elasticity_relative_20samples_frameindex"
echo ""

# =============================================================================
# 完成总结
# =============================================================================

echo "🎉 Elasticity Relative Ablation Study - 基础实验 完成！"
echo ""
echo "实验总结:"
echo "=== 基础实验 (5个) ==="
echo "1. Baseline                    -> gemini_physics_analysis_results_elasticity_relative_20samples"
echo "2. + Concat                    -> gemini_physics_analysis_results_elasticity_relative_20samples_concat"
echo "3. + ICL                       -> gemini_physics_analysis_results_elasticity_relative_20samples_icl"
echo "4. + Oracle                    -> gemini_physics_analysis_results_elasticity_relative_20samples_oracle"
echo "5. + FrameIndex                -> gemini_physics_analysis_results_elasticity_relative_20samples_frameindex"
echo ""
echo "结果目录列表:"
ls -d gemini_physics_analysis_results_elasticity_relative_20samples* 2>/dev/null | sort
echo ""
echo "注意: 所有实验都将使用新的置信度输出功能 (0-1之间的分数)"
echo "建议接下来分析结果来比较不同功能的效果" 