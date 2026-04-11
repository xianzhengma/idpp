#!/bin/bash
# Elasticity Relative Ablation Study - Basic Experiments Only
#Basic experiment: baseline + 4 individual function experiments
# Use 20 pairs of samples for each experiment

echo "Start executing the Elasticity Relative Ablation Study - Basic Experiment..."
echo "Execute baseline + 4 single function experiments"
echo "A total of 5 experiments, each experiment has 20 pairs of samples"
echo "Estimated total time: 2.5 Hour"
echo ""

# =============================================================================
#Basic experiments (5)
# =============================================================================

# echo "=== Basic experiment ==="
# echo ""

# echo "Experiment 1/5: Baseline"
# echo "Description: Baseline experiment (without any additional features)"
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 20 \
# --properties elasticity \
# --output-dir gemini_physics_analysis_results_elasticity_relative_20samples \
# --auto-confirm
# echo "✓ Experiment 1 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_relative_20samples"
# echo ""

# echo "Experiment 2/5: + Concat"
# echo "Description: baseline + stitching mode"
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 20 \
# --properties elasticity \
# --concat-mode \
# --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_concat \
# --auto-confirm
# echo "✓ Experiment 2 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_relative_20samples_concat"
# echo ""

echo "Experiment 3/5: +ICL"
echo "Description: Baseline + ICL example"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_icl \
    --auto-confirm
echo "✓ Experiment 3 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_relative_20samples_icl"
echo ""

echo "Experiment 4/5: + Oracle"
echo "Description: Baseline + Oracle Guidance"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_oracle \
    --auto-confirm
echo "✓ Experiment 4 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_relative_20samples_oracle"
echo ""

echo "Experiment 5/5: + FrameIndex"
echo "Description: baseline +frame index"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 20 \
    --properties elasticity \
    --frame-index \
    --output-dir gemini_physics_analysis_results_elasticity_relative_20samples_frameindex \
    --auto-confirm
echo "✓ Experiment 5 is completed, the results are saved in: gemini_physics_analysis_results_elasticity_relative_20samples_frameindex"
echo ""

# =============================================================================
# Complete summary
# =============================================================================

echo "🎉 Elasticity Relative Ablation Study - Basic experiment completed!"
echo ""
echo "Experiment Summarize:"
echo "=== Basic experiments (5) ==="
echo "1. Baseline -> gemini_physics_analysis_results_elasticity_relative_20samples"
echo "2. + Concat -> gemini_physics_analysis_results_elasticity_relative_20samples_concat"
echo "3. + ICL -> gemini_physics_analysis_results_elasticity_relative_20samples_icl"
echo "4. + Oracle -> gemini_physics_analysis_results_elasticity_relative_20samples_oracle"
echo "5. + FrameIndex -> gemini_physics_analysis_results_elasticity_relative_20samples_frameindex"
echo ""
echo "Result directory list:"
ls -d gemini_physics_analysis_results_elasticity_relative_20samples* 2>/dev/null | sort
echo ""
echo "Note: All experiments will use the new confidence output function (a score between 0-1)"
echo "It is recommended to analyze the results next to compare the effects of different functions"