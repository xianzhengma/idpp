#!/bin/bash
# Friction Absolute Ablation Study
# 完整的baseline + 单个功能、两两组合和三功能组合实验
# 每个实验使用20个样本

echo "开始执行 Friction Absolute Ablation Study..."
echo "总共8个实验（包含baseline），每个实验20个样本"
echo "预计总时间: 4.0 小时"
echo ""

# =============================================================================
# Baseline实验 (1个)
# =============================================================================

echo "=== Baseline实验 ==="
echo ""

echo "实验 1/8: Baseline"
echo "描述: 标准prompt，作为所有实验的基准"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples \
    --auto-confirm
echo "✓ Baseline实验完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples"
echo ""

# =============================================================================
# 单个功能实验 (3个)
# =============================================================================

echo "=== 单个功能实验 ==="
echo ""

echo "实验 2/8: + FrameIndex"
echo "描述: 为每帧添加显式索引标识"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --frame-index \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_frameindex \
    --auto-confirm
echo "✓ 实验2完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_frameindex"
echo ""

echo "实验 3/8: + ICL"  
echo "描述: 使用in-context learning示例"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --icl-example \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_icl \
    --auto-confirm
echo "✓ 实验3完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_icl"
echo ""

echo "实验 4/8: + Oracle"
echo "描述: 提供具体的分析步骤指导"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_oracle \
    --auto-confirm
echo "✓ 实验4完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_oracle"
echo ""

# =============================================================================
# 两两组合实验 (3个)
# =============================================================================

echo "=== 两两组合实验 ==="
echo ""

echo "实验 5/8: + FrameIndex + ICL"
echo "描述: 帧索引 + ICL示例的组合"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --frame-index \
    --icl-example \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl \
    --auto-confirm
echo "✓ 实验5完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl"
echo ""

echo "实验 6/8: + FrameIndex + Oracle"
echo "描述: 帧索引 + Oracle指导的组合"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --frame-index \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_frameindex_oracle \
    --auto-confirm
echo "✓ 实验6完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_frameindex_oracle"
echo ""

echo "实验 7/8: + ICL + Oracle"
echo "描述: ICL示例 + Oracle指导的组合"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --icl-example \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_icl_oracle \
    --auto-confirm
echo "✓ 实验7完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_icl_oracle"
echo ""

# =============================================================================
# 三功能组合实验 (1个)
# =============================================================================

echo "=== 三功能组合实验 ==="
echo ""

echo "实验 8/8: + FrameIndex + ICL + Oracle"
echo "描述: 所有三个功能的完整组合"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties friction \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl_oracle \
    --auto-confirm
echo "✓ 实验8完成，结果保存在: gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl_oracle"
echo ""

# =============================================================================
# 完成总结
# =============================================================================

echo "🎉 Friction Absolute Ablation Study 完成！"
echo ""
echo "实验总结:"
echo "1. Baseline                       -> gemini_physics_analysis_results_friction_absolute_20samples"
echo "2. + FrameIndex                   -> gemini_physics_analysis_results_friction_absolute_20samples_frameindex"
echo "3. + ICL                          -> gemini_physics_analysis_results_friction_absolute_20samples_icl"  
echo "4. + Oracle                       -> gemini_physics_analysis_results_friction_absolute_20samples_oracle"
echo "5. + FrameIndex + ICL             -> gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl"
echo "6. + FrameIndex + Oracle          -> gemini_physics_analysis_results_friction_absolute_20samples_frameindex_oracle"
echo "7. + ICL + Oracle                 -> gemini_physics_analysis_results_friction_absolute_20samples_icl_oracle"
echo "8. + FrameIndex + ICL + Oracle    -> gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl_oracle"
echo ""
echo "结果目录列表:"
ls -d gemini_physics_analysis_results_friction_absolute_20samples* | sort
echo ""
echo "建议接下来运行 process_individual_model_results_ablation.py 来生成结果表格" 