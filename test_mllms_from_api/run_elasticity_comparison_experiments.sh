#!/bin/bash
# Elasticityrelative comparison comparison experiment
# Model: Gemini-2.5-pro
# property:elasticity
# type:relative only
# Number of samples: 50 relative comparison pairs

echo "Start executing the Elasticityrelative comparison experiment..."
echo "Model: Gemini-2.5-pro"
echo "property:elasticity"
echo "Number of samples: 50 relative comparison pairs"
echo "Estimated total time: 2.5Hour"
echo ""

echo "Experiment 1: Baseline"
echo "Description: standard prompt, used as baseline"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples \
    --auto-confirm
echo "Experiment 1 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples"
echo "=" * 50
echo ""

echo "Experiment 2: Baseline + Concat"
echo "Description: Basic mode + splicing mode"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat \
    --auto-confirm
echo "Experiment 2 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_concat"
echo "=" * 50
echo ""

echo "Experiment 3: Baseline + Concat + FrameIndex"
echo "Description: Basic mode + splicing mode + frame index"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex \
    --auto-confirm
echo "Experiment 3 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex"
echo "=" * 50
echo ""

echo "Experiment 4: Baseline + Concat + FrameIndex + ICL"
echo "Description: Basic mode + splicing mode + frame index + ICL example"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl \
    --auto-confirm
echo "Experiment 4 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl"
echo "=" * 50
echo ""

echo "Experiment 5: Baseline + Concat + FrameIndex + ICL + Oracle"
echo "Description: All function combinations"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl_oracle \
    --auto-confirm
echo "Experiment 5 completed, results saved in gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl_oracle"
echo "=" * 50
echo ""

python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_frameindex_icl_oracle \
    --auto-confirm

echo "All Elasticityrelative comparison experiments completed!"
echo "The results are saved in the following directory:"
ls -d gemini_physics_analysis_results_elasticity_relative_50samples*

echo ""
echo "You can use the following command to analyze the results:"
echo "python process_individual_model_results_ablation.py"