#!/bin/bash
# Elasticity Absolute Ablation Study
# Complete baseline + single function, two-two combination and three-function combination experiments
# Use 20 samples for each experiment

echo "Start executing Elasticity Absolute Ablation Study..."
echo "A total of 8 experiments (including baseline), each experiment has 20 samples"
echo "Estimated total time: 4.0 Hour"
echo ""

# =============================================================================
# Baseline experiment (1)
# =============================================================================

echo "=== Baseline experiment ==="
echo ""

echo "Experiment 1/8: Baseline"
echo "Description: Standard prompt, used as the baseline for all experiments"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples \
    --auto-confirm
echo "✓ Baseline experiment is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples"
echo ""

# =============================================================================
# Single function experiment (3)
# =============================================================================

echo "=== Single function experiment ==="
echo ""

echo "Experiment 2/8: + FrameIndex"
echo "Description: Add explicit index identifier to each frame"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --frame-index \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex \
    --auto-confirm
echo "✓ Experiment 2 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex"
echo ""

echo "Experiment 3/8: +ICL"
echo "Description: Example of using in-context learning"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_icl \
    --auto-confirm
echo "✓ Experiment 3 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_icl"
echo ""

echo "Experiment 4/8: + Oracle"
echo "Description: Provide specific analysis step guidance"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_oracle \
    --auto-confirm
echo "✓ Experiment 4 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_oracle"
echo ""

# =============================================================================
# Pairwise combination experiments (3)
# =============================================================================

echo "=== Pairwise combination experiment ==="
echo ""

echo "Experiment 5/8: + FrameIndex + ICL"
echo "Description: combination of frame index + ICL example"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --frame-index \
    --icl-example \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl \
    --auto-confirm
echo "✓ Experiment 5 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl"
echo ""

echo "Experiment 6/8: + FrameIndex + Oracle"
echo "Description: combination of frame index + Oracle guidance"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --frame-index \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_oracle \
    --auto-confirm
echo "✓ Experiment 6 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_oracle"
echo ""

echo "Experiment 7/8: + ICL + Oracle"
echo "Description: Combination of ICL examples + Oracle guidance"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --icl-example \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_icl_oracle \
    --auto-confirm
echo "✓ Experiment 7 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_icl_oracle"
echo ""

# =============================================================================
# Three-function combination experiment (1)
# =============================================================================echo "=== Three-function combination experiment ==="
echo ""

echo "Experiment 8/8: + FrameIndex + ICL + Oracle"
echo "Description: Complete combination of all three functions"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 20 \
    --relative-pairs 0 \
    --properties elasticity \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl_oracle \
    --auto-confirm
echo "✓ Experiment 8 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl_oracle"
echo ""

# =============================================================================
# Complete summary
# =============================================================================

echo "🎉 Elasticity Absolute Ablation Study completed!"
echo ""
echo "Experiment Summarize:"
echo "1. Baseline -> gemini_physics_analysis_results_elasticity_absolute_20samples"
echo "2. + FrameIndex -> gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex"
echo "3. + ICL -> gemini_physics_analysis_results_elasticity_absolute_20samples_icl"
echo "4. + Oracle -> gemini_physics_analysis_results_elasticity_absolute_20samples_oracle"
echo "5. + FrameIndex + ICL -> gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl"
echo "6. + FrameIndex + Oracle -> gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_oracle"
echo "7. + ICL + Oracle -> gemini_physics_analysis_results_elasticity_absolute_20samples_icl_oracle"
echo "8. + FrameIndex + ICL + Oracle -> gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl_oracle"
echo ""
echo "Result directory list:"
ls -d gemini_physics_analysis_results_elasticity_absolute_20samples* | sort
echo ""
echo "It is recommended to run process_elasticity_absolute_ablation.py next to generate the result table"