#!/bin/bash
# Friction Absolute Ablation Study - 100 Samples (Gemini Only)
# Friction absolute ablation experiment of Gemini-2.5-Pro model
# Each experiment uses 100 samples and ablates a single prompting function

# Set dedicated API key
: "${OPENAI_API_KEY:? Please export OPENAI_API_KEY first}"

echo "Start executing Friction Absolute Ablation Study (100 samples)..."
echo "Model: Gemini-2.5-Pro"
echo "3 experiments (except ICL), 100 samples each"
echo "Use API Key: ${OPENAI_API_KEY:0:10}..."
echo "Estimated total time: 2.0 Hour"
echo ""

# =============================================================================
# Gemini-2.5-Pro model experiment
# =============================================================================

echo "=== Gemini-2.5-Pro model experiment ==="
echo ""

echo "Experiment 1/3: Gemini Baseline"
echo "Description: Standard prompt, used as the baseline for all experiments"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties friction \
    --output-dir gemini_physics_analysis_results_absolute_100samples \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini Baseline experiment completed"
echo ""

echo "Experiment 2/3: Gemini + FrameIndex"
echo "Description: Add explicit index identifier to each frame"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties friction \
    --frame-index \
    --output-dir gemini_physics_analysis_results_absolute_100samples_frameindex \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini + FrameIndex experiment completed"
echo ""

echo "Experiment 3/3: Gemini + Oracle"
echo "Description: Provide guidance on problem-solving steps"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties friction \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_absolute_100samples_oracle \
    --aggressive-retry \
    --auto-confirm
echo "✓ Gemini + Oracle experiment completed"
echo ""

echo "🎉 All Friction Absolute Ablation experiments completed!"
echo "Complete a total of 3 experiments, each with 100 samples"
echo "Results are saved in respective output directories"