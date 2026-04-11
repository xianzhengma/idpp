#!/bin/bash
# Execute recommended comparative experiments in batches
# Sort by priority, starting with the most important

echo "Start executing the recommended comparison experiment..."
echo "Estimated total time: 5.0 Hour"
echo ""

# echo "Experiment 1: Basic Mode [High Priority]"
# echo "Description: Standard prompt, used as baseline"
# echo "Reason: as a baseline for all experiments"
# python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --properties viscosity elasticity --auto-confirm
# echo "Experiment 1 completed, results saved in gemini_physics_analysis_results"
# echo "=" * 50
# echo ""

echo "Experiment 2: ICL example mode [high priority]"
echo "Description: Example of using in-context learning"
echo "Reason: ICL often significantly improves performance"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --icl-example --properties viscosity elasticity --auto-confirm
echo "Experiment 2 completed, results saved in gemini_physics_analysis_results_icl-example"
echo "=" * 50
echo ""

echo "Experiment 3: Oracle Guidance Mode [High Priority]"
echo "Description: Provide specific analysis step guidance"
echo "Reason: Problem-solving step-by-step guidance may be effective"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --follow-oracle-test --properties viscosity elasticity --auto-confirm
echo "Experiment 3 is completed, the results are saved in gemini_physics_analysis_results_follow-oracle-test"
echo "=" * 50
echo ""

echo "Experiment 4: ICL + Oracle combination [high priority]"
echo "Description: ICL example + problem solving step guide"
echo "Reason: combination of two important functions"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --icl-example --follow-oracle-test --properties viscosity elasticity --auto-confirm
echo "Experiment 4 completed, results saved in gemini_physics_analysis_results_icl-example_follow-oracle-test"
echo "=" * 50
echo ""

echo "Experiment 5: Splice Mode [Medium Priority]"
echo "Description: Use splicing mode when relative comparison"
echo "Reason: It may improve the effect of relative comparison"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --concat-mode --properties viscosity elasticity --auto-confirm
echo "Experiment 5 completed, results saved in gemini_physics_analysis_results_concat-mode"
echo "=" * 50
echo ""

echo "Experiment 6: frame index mode [medium priority]"
echo "Description: Add explicit index to each frame"
echo "Reason: May help the model better understand timing"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --frame-index --properties viscosity elasticity --auto-confirm
echo "Experiment 6 completed, results saved in gemini_physics_analysis_results_frame-index"
echo "=" * 50
echo ""

echo "Experiment 7: ICL + splicing combination [medium priority]"
echo "Description: ICL example + splicing mode"
echo "Reason: combination of ICL and splicing"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --concat-mode --properties viscosity elasticity --auto-confirm
echo "Experiment 7 completed, results saved in gemini_physics_analysis_results_icl-example_concat-mode"
echo "=" * 50
echo ""

echo "Experiment 8: Oracle + splicing combination [medium priority]"
echo "Description: Oracle guidance + splicing mode"
echo "Reason: combination of Oracle and splicing"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --follow-oracle-test --concat-mode --properties viscosity elasticity --auto-confirm
echo "Experiment 8 completed, results saved in gemini_physics_analysis_results_follow-oracle-test_concat-mode"
echo "=" * 50
echo ""

echo "Experiment 9: ICL + Oracle +frame index [medium priority]"
echo "Description: A combination of three main functions"
echo "Reason: Combination of three important functions"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --follow-oracle-test --frame-index --properties viscosity elasticity --auto-confirm
echo "Experiment 9 completed, results saved in gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index"
echo "=" * 50
echo ""

echo "Experiment 10: Fully functional mode [low priority]"
echo "Description: Combination of all functions"
echo "Reason: Test the effect of all function combinations"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --follow-oracle-test --concat-mode --frame-index --properties viscosity elasticity --auto-confirm
echo "Experiment 10 completed, results saved in gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index"
echo "=" * 50
echo ""

echo "All recommended experiments completed!"
echo "The results are saved in the following directory:"
ls -d gemini_physics_analysis_results*