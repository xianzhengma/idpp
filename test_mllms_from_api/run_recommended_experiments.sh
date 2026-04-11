#!/bin/bash
# 批量执行推荐的对比实验
# 按优先级排序，从最重要的开始

echo "开始执行推荐的对比实验..."
echo "预计总时间: 5.0 小时"
echo ""

# echo "实验 1: 基础模式 [高优先级]"
# echo "描述: 标准prompt，作为baseline"
# echo "原因: 作为所有实验的baseline"
# python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --properties viscosity elasticity --auto-confirm
# echo "实验 1 完成，结果保存在 gemini_physics_analysis_results"
# echo "=" * 50
# echo ""

echo "实验 2: ICL示例模式 [高优先级]"
echo "描述: 使用in-context learning示例"
echo "原因: ICL通常能显著提升性能"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --icl-example --properties viscosity elasticity --auto-confirm
echo "实验 2 完成，结果保存在 gemini_physics_analysis_results_icl-example"
echo "=" * 50
echo ""

echo "实验 3: Oracle指导模式 [高优先级]"
echo "描述: 提供具体的分析步骤指导"
echo "原因: 解题步骤指导可能很有效"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --follow-oracle-test --properties viscosity elasticity --auto-confirm
echo "实验 3 完成，结果保存在 gemini_physics_analysis_results_follow-oracle-test"
echo "=" * 50
echo ""

echo "实验 4: ICL + Oracle组合 [高优先级]"
echo "描述: ICL示例 + 解题步骤指导"
echo "原因: 两个重要功能的组合"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --icl-example --follow-oracle-test --properties viscosity elasticity --auto-confirm
echo "实验 4 完成，结果保存在 gemini_physics_analysis_results_icl-example_follow-oracle-test"
echo "=" * 50
echo ""

echo "实验 5: 拼接模式 [中优先级]"
echo "描述: 相对比较时使用拼接模式"
echo "原因: 可能改善相对比较的效果"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --concat-mode --properties viscosity elasticity --auto-confirm
echo "实验 5 完成，结果保存在 gemini_physics_analysis_results_concat-mode"
echo "=" * 50
echo ""

echo "实验 6: 帧索引模式 [中优先级]"
echo "描述: 为每帧添加显式索引"
echo "原因: 可能帮助模型更好地理解时序"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --frame-index --properties viscosity elasticity --auto-confirm
echo "实验 6 完成，结果保存在 gemini_physics_analysis_results_frame-index"
echo "=" * 50
echo ""

echo "实验 7: ICL + 拼接组合 [中优先级]"
echo "描述: ICL示例 + 拼接模式"
echo "原因: ICL与拼接的组合"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --concat-mode --properties viscosity elasticity --auto-confirm
echo "实验 7 完成，结果保存在 gemini_physics_analysis_results_icl-example_concat-mode"
echo "=" * 50
echo ""

echo "实验 8: Oracle + 拼接组合 [中优先级]"
echo "描述: Oracle指导 + 拼接模式"
echo "原因: Oracle与拼接的组合"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --follow-oracle-test --concat-mode --properties viscosity elasticity --auto-confirm
echo "实验 8 完成，结果保存在 gemini_physics_analysis_results_follow-oracle-test_concat-mode"
echo "=" * 50
echo ""

echo "实验 9: ICL + Oracle + 帧索引 [中优先级]"
echo "描述: 三个主要功能的组合"
echo "原因: 三个重要功能的组合"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --follow-oracle-test --frame-index --properties viscosity elasticity --auto-confirm
echo "实验 9 完成，结果保存在 gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index"
echo "=" * 50
echo ""

echo "实验 10: 全功能模式 [低优先级]"
echo "描述: 所有功能的组合"
echo "原因: 测试所有功能组合的效果"
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 0 --relative-pairs 20 --icl-example --follow-oracle-test --concat-mode --frame-index --properties viscosity elasticity --auto-confirm
echo "实验 10 完成，结果保存在 gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index"
echo "=" * 50
echo ""

echo "所有推荐实验完成！"
echo "结果保存在以下目录："
ls -d gemini_physics_analysis_results*
