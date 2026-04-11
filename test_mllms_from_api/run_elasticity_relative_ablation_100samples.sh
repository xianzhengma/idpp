#!/bin/bash
# Elasticity Relative Ablation Study - 100 Samples (Gemini Only)
# Elasticity relative ablation experiment of Gemini-2.5-Pro model
# Each experiment uses 100 samples and ablates a single prompting function (except oracle)

# Set dedicated API key
: "${OPENAI_API_KEY:? Please export OPENAI_API_KEY first}"

echo "Start executing Elasticity Relative Ablation Study (100 samples)..."
echo "Model: Gemini-2.5-Pro"
echo "4 experiments (except Oracle), 100 samples each"
echo "Use API Key: ${OPENAI_API_KEY:0:10}..."
echo "Estimated total time: 2.7 Hour"
echo ""

# =============================================================================
# Gemini-2.5-Pro model experiment
# =============================================================================

echo "=== Gemini-2.5-Pro model experiment ==="
echo ""

# echo "Experiment 1/4: Gemini Baseline"
# echo "Description: Standard prompt, used as the baseline for all experiments"
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 100 \
# --properties elasticity \
# --output-dir gemini_physics_analysis_results_relative_100samples \
# --aggressive-retry \
# --auto-confirm
# echo "✓ Gemini Baseline experiment completed"
# echo ""

echo "Experiment 2/4: Gemini + FrameIndex"
echo "Description: Add explicit index identifier to each frame"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties elasticity \
    --frame-index \
    --output-dir gemini_physics_analysis_results_relative_100samples_frameindex \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini + FrameIndex experiment completed"
echo ""

# echo "Experiment 3/4: Gemini + ICL"
# echo "Description: Provide in-context learning examples"
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 100 \
# --properties elasticity \
# --icl-example \
# --output-dir gemini_physics_analysis_results_relative_100samples_icl \
# --aggressive-retry \
# --auto-confirm
# echo "✓ Gemini + ICL experiment completed"
# echo ""

# echo "Experiment 4/4: Gemini + Concat"
# echo "Description: Compare videos using stitching mode"
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 100 \
# --properties elasticity \
# --concat-mode \
# --output-dir gemini_physics_analysis_results_relative_100samples_concat \
# --aggressive-retry \
# --auto-confirm
# echo "✓ Gemini + Concat experiment completed"
# echo ""

echo "🎉 All Elasticity Relative Ablation experiments completed!"
echo "Complete a total of 4 experiments, each with 100 samples"
echo "Results are saved in respective output directories"