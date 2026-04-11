#!/bin/bash
# Gemini-2.5-Pro relative setting test script
# Number of samples: 100 relatively paired samples
# Strategy: Oracle strategy
# Physical properties:friction, viscosity, elasticity

echo "Start executing Gemini-2.5-Pro relative setting test..."
echo "Model: gemini-2.5-pro"
echo "setting: relative setting (100 samples)"
echo "Strategy: Oracle Policy"
echo "Physical properties:friction, viscosity, elasticity"
echo "Estimated total time: about 3Hour"
echo ""

# =============================================================================
# Friction relative testing
# =============================================================================

echo "=== Test 1/3: Friction relative test ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 100 \
    --properties friction \
    --follow-oracle-test \
    --output-dir gemini_physics_analysis_results_relative_100samples_oracle \
    --auto-confirm
echo "✓ Friction relative test is completed, the results are saved in: gemini_physics_analysis_results_relative_100samples_oracle"
echo ""

# =============================================================================
# Viscosity relative testing
# =============================================================================

# echo "=== Test 2/3: Viscosity Relative Test ==="
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 100 \
# --properties viscosity \
# --follow-oracle-test \
# --output-dir gemini_physics_analysis_results_relative_100samples_oracle \
# --auto-confirm
# echo "✓ Viscosity relative test is completed, the results are saved in: gemini_physics_analysis_results_relative_100samples_oracle"
# echo ""

# # =============================================================================
# # Elasticity relative testing
# # =============================================================================

# echo "=== Test 3/3: Elasticity relative test ==="
# python run_gpt_physics_tests.py \
# --model-version gemini-2.5-pro \
# --absolute-samples 0 \
# --relative-pairs 100 \
# --properties elasticity \
# --follow-oracle-test \
# --output-dir gemini_physics_analysis_results_relative_100samples_oracle \
# --auto-confirm
# echo "✓ Elasticity relative testing is completed and the results are saved in: gemini_physics_analysis_results_relative_100samples_oracle"
echo ""

# =============================================================================
# Complete summary
# =============================================================================

echo "🎉 Gemini-2.5-Pro relative setting test completed!"
echo ""
echo "Test summary:"
echo "1. Friction relative (100 samples) -> gemini_physics_analysis_results_relative_100samples_oracle"
echo "2. Viscosity relative (100 samples) -> gemini_physics_analysis_results_relative_100samples_oracle"
echo "3. Elasticity relative (100 samples) -> gemini_physics_analysis_results_relative_100samples_oracle"
echo ""
echo "Result directory list:"
ls -d gemini_physics_analysis_results_relative_100samples_oracle 2>/dev/null | sort
echo ""
echo "All tests completed!"