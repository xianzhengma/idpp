#!/bin/bash
# 单独运行三个功能的elasticity相对比较实验
# 修复ICL bug后重新测试

echo "开始运行单独功能实验..."
echo "模型: Gemini-2.5-pro"
echo "属性: elasticity"
echo "类型: relative comparison"
echo "样本数: 50个相对比较对"
echo ""

echo "实验 1: + FrameIndex"
echo "描述: 单独测试帧索引功能"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --frame-index \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_frameindex \
    --auto-confirm
echo "实验 1 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_frameindex"
echo "=================================="
echo ""

echo "实验 2: + ICL (修复后，使用1个示例)"
echo "描述: 单独测试ICL功能 (已修复相对比较bug，自动使用1个示例避免API限制)"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --icl-example \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_icl \
    --auto-confirm
echo "实验 2 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_icl"
echo "=================================="
echo ""

echo "实验 3: + Oracle"
echo "描述: 单独测试Oracle指导功能"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --follow-oracle-test \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_oracle \
    --auto-confirm
echo "实验 3 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_oracle"
echo "=================================="
echo ""

echo "所有单独功能实验完成！"
echo "结果目录："
echo "  - gemini_physics_analysis_results_elasticity_relative_50samples_frameindex"
echo "  - gemini_physics_analysis_results_elasticity_relative_50samples_icl"
echo "  - gemini_physics_analysis_results_elasticity_relative_50samples_oracle" 