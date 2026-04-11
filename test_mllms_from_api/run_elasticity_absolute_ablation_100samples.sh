#!/bin/bash
# Elasticity Absolute Ablation Study - 100 Samples (Gemini Only)
# Gemini-2.5-Pro模型的elasticity absolute ablation实验
# 每个实验使用100个样本，消融单个prompting功能

# 运行前请 export OPENAI_API_KEY=...
: "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY}"

# echo "开始执行 Elasticity Absolute Ablation Study (100 samples)..."
# echo "模型：Gemini-2.5-Pro"
# echo "3个实验（除了ICL），每个实验100个样本"
# echo "使用API Key: ${OPENAI_API_KEY:0:10}..."
# echo "预计总时间: 2.0 小时"
# echo ""

# # =============================================================================
# # Gemini-2.5-Pro 模型实验
# # =============================================================================

# echo "=== Gemini-2.5-Pro 模型实验 ==="
# echo ""

# echo "实验 1/3: Gemini Baseline"
# echo "描述: 标准prompt，作为所有实验的基准"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 100 \
#     --relative-pairs 0 \
#     --properties elasticity \
#     --output-dir gemini_physics_analysis_results_absolute_100samples \
#     --aggressive-retry \
#     --auto-confirm
# echo "✓ Gemini Baseline实验完成"
# echo ""

# echo "实验 2/3: Gemini + FrameIndex"
# echo "描述: 为每帧添加显式索引标识"
# python run_gpt_physics_tests.py \
#     --model-version gemini-2.5-pro \
#     --absolute-samples 100 \
#     --relative-pairs 0 \
#     --properties elasticity \
#     --frame-index \
#     --output-dir gemini_physics_analysis_results_absolute_100samples_frameindex \
#     --aggressive-retry \
#     --auto-confirm
# echo "✓ Gemini + FrameIndex实验完成"
# echo ""

echo "实验 3/3: Gemini + Oracle"
echo "描述: 提供解题步骤指导"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties elasticity \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_absolute_100samples_oracle \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini + Oracle实验完成"
echo ""

echo "🎉 所有 Elasticity Absolute Ablation 实验完成！"
echo "总共完成3个实验，每个100个样本"
echo "结果保存在各自的输出目录中"