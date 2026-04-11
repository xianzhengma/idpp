#!/bin/bash
# Elasticity相对比较对比实验
# 模型: Gemini-2.5-pro
# 属性: elasticity
# 类型: relative only
# 样本数: 50个相对比较对

echo "开始执行Elasticity相对比较对比实验..."
echo "模型: Gemini-2.5-pro"
echo "属性: elasticity"
echo "样本数: 50个相对比较对"
echo "预计总时间: 2.5小时"
echo ""

echo "实验 1: Baseline"
echo "描述: 标准prompt，作为baseline"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples \
    --auto-confirm
echo "实验 1 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples"
echo "=" * 50
echo ""

echo "实验 2: Baseline + Concat"
echo "描述: 基础模式 + 拼接模式"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat \
    --auto-confirm
echo "实验 2 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_concat"
echo "=" * 50
echo ""

echo "实验 3: Baseline + Concat + FrameIndex"
echo "描述: 基础模式 + 拼接模式 + 帧索引"
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex \
    --auto-confirm
echo "实验 3 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex"
echo "=" * 50
echo ""

echo "实验 4: Baseline + Concat + FrameIndex + ICL"
echo "描述: 基础模式 + 拼接模式 + 帧索引 + ICL示例"
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
echo "实验 4 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl"
echo "=" * 50
echo ""

echo "实验 5: Baseline + Concat + FrameIndex + ICL + Oracle"
echo "描述: 所有功能组合"
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
echo "实验 5 完成，结果保存在 gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl_oracle"
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

echo "所有Elasticity相对比较实验完成！"
echo "结果保存在以下目录："
ls -d gemini_physics_analysis_results_elasticity_relative_50samples*

echo ""
echo "可以使用以下命令分析结果:"
echo "python process_individual_model_results_ablation.py" 