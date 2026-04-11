#!/bin/bash
# Gemini-2.5-Pro absolute setting test script
# Number of samples: 100 absolute samples
# Strategy: ICL strategy (In-Context Learning)
# Physical properties:friction, viscosity, elasticity

echo "Start executing Gemini-2.5-Pro absolute setting test..."
echo "Model: gemini-2.5-pro"
echo "setting: absolute setting (100 samples)"
echo "Strategy: ICL Strategy (In-Context Learning)"
echo "Physical properties:friction, viscosity, elasticity"
echo "Estimated total time: about 3Hour"
echo ""

# =============================================================================
#Friction Absolute Test
# =============================================================================

echo "=== Test 1/3: Friction absolute test ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties friction \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Friction absolute test is completed, the results are saved in: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
#Viscosity Absolute Test
# =============================================================================

echo "=== Test 2/3: Viscosity Absolute Test ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties viscosity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Viscosity absolute test is completed, the results are saved in: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
#Elasticity Absolute Testing
# =============================================================================

echo "=== Test 3/3: Elasticity Absolute Test ==="
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 100 \
    --relative-pairs 0 \
    --properties elasticity \
    --icl-example \
    --output-dir gemini_physics_analysis_results_absolute_100samples_icl \
    --auto-confirm
echo "✓ Elasticity absolute test is completed, the results are saved in: gemini_physics_analysis_results_absolute_100samples_icl"
echo ""

# =============================================================================
# Complete summary
# =============================================================================

echo "🎉 Gemini-2.5-Pro absolute setting test completed!"
echo ""
echo "Test summary:"
echo "1. Friction absolute (100 samples) -> gemini_physics_analysis_results_absolute_100samples_icl"
echo "2. Viscosity Absolute (100 samples) -> gemini_physics_analysis_results_absolute_100samples_icl"
echo "3. Elasticity absolute (100 samples) -> gemini_physics_analysis_results_absolute_100samples_icl"
echo ""
echo "Result directory list:"
ls -d gemini_physics_analysis_results_absolute_100samples_icl 2>/dev/null | sort
echo ""
echo "All tests completed!"