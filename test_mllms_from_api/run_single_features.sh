#!/bin/bash
# Run the elasticityrelative comparison experiment of three functions separately
# Retest after fixing ICL bug

echo "Start running a single function experiment..."
echo "Model: Gemini-2.5-pro"
echo "property:elasticity"
echo "type:relative comparison"
echo "Number of samples: 50 relative comparison pairs"
echo ""

echo "Experiment 1: + FrameIndex"
echo "Description: Test the frame index function alone"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --frame-index \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_frameindex \
    --auto-confirm
echo "Experiment 1 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_frameindex"
echo "=================================="
echo ""

echo "Experiment 2: +ICL (after fix, using 1 example)"
echo "Description: Test the ICL function alone (relative comparisonbug has been fixed, automatically using 1 example to avoid API restrictions)"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --icl-example \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_icl \
    --auto-confirm
echo "Experiment 2 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_icl"
echo "=================================="
echo ""

echo "Experiment 3: + Oracle"
echo "Description: Test Oracle guidance function alone"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --follow-oracle-test \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_oracle \
    --auto-confirm
echo "Experiment 3 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_oracle"
echo "=================================="
echo ""

echo "All individual function experiments completed!"
echo "Result directory:"
echo "-gemini_physics_analysis_results_elasticity_relative_50samples_frameindex"
echo "-gemini_physics_analysis_results_elasticity_relative_50samples_icl"
echo "-gemini_physics_analysis_results_elasticity_relative_50samples_oracle"