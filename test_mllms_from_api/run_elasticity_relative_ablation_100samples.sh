#!/bin/bash
# Elasticity Relative Ablation Study - 100 Samples (Gemini Only)
# Gemini-2.5-Pro模型的elasticity relative ablation实验
# 每个实验使用100个样本，消融单个prompting功能（除了oracle）

# 设置专用API key
export OPENAI_API_KEY="sk-Fc2kqPwHQ2OiPGBBNiTE6uDOiIPnb9jQ8nW1yiq7xXMFmvch"

echo "开始执行 Elasticity Relative Ablation Study (100 samples)..."
echo "模型：Gemini-2.5-Pro"
echo "4个实验（除了Oracle），每个实验100个样本"
echo "使用API Key: ${OPENAI_API_KEY:0:10}..."
echo "预计总时间: 2.7 小时"
echo ""

# =============================================================================
# Gemini-2.5-Pro 模型实验
# =============================================================================

echo "=== Gemini-2.5-Pro 模型实验 ==="
echo ""

# echo "实验 1/4: Gemini Baseline"
# echo "描述: 标准prompt，作为所有实验的基准"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 0 \
#     --relative-pairs 100 \
#     --properties elasticity \
#     --output-dir gemini_physics_analysis_results_relative_100samples \
#     --aggressive-retry \
#     --auto-confirm
# echo "✓ Gemini Baseline实验完成"
# echo ""

echo "实验 2/4: Gemini + FrameIndex"
echo "描述: 为每帧添加显式索引标识"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties elasticity \
    --frame-index \
    --output-dir gemini_physics_analysis_results_relative_100samples_frameindex \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini + FrameIndex实验完成"
echo ""

# echo "实验 3/4: Gemini + ICL"
# echo "描述: 提供in-context learning示例"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 0 \
#     --relative-pairs 100 \
#     --properties elasticity \
#     --icl-example \
#     --output-dir gemini_physics_analysis_results_relative_100samples_icl \
#     --aggressive-retry \
#     --auto-confirm
# echo "✓ Gemini + ICL实验完成"
# echo ""

# echo "实验 4/4: Gemini + Concat"
# echo "描述: 使用拼接模式比较视频"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 0 \
#     --relative-pairs 100 \
#     --properties elasticity \
#     --concat-mode \
#     --output-dir gemini_physics_analysis_results_relative_100samples_concat \
#     --aggressive-retry \
#     --auto-confirm
# echo "✓ Gemini + Concat实验完成"
# echo ""

echo "🎉 所有 Elasticity Relative Ablation 实验完成！"
echo "总共完成4个实验，每个100个样本"
echo "结果保存在各自的输出目录中"