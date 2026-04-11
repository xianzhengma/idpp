#!/usr/bin/env python3
"""
进度检查脚本
快速检查elasticity absolute ablation实验的完成情况
"""

import os
import glob
import re
from typing import Dict, List


def check_experiment_progress() -> None:
    """检查实验进度"""
    print("检查 Elasticity Absolute Ablation 实验进度...")
    
    # 实验目录列表
    experiment_dirs = [
        "gemini_physics_analysis_results_elasticity_absolute_20samples",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_icl",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_oracle",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_oracle",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_icl_oracle",
        "gemini_physics_analysis_results_elasticity_absolute_20samples_frameindex_icl_oracle",
    ]
    
    total_complete = 0
    total_samples = 0
    
    print("\n" + "="*100)
    print(f"{'实验名称':<60} {'Test-1':<10} {'Test-2':<10} {'Test-3':<10} {'总计':<10}")
    print("="*100)
    
    for exp_dir in experiment_dirs:
        if not os.path.exists(exp_dir):
            print(f"{exp_dir:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        
        # 查找结果文件
        result_files = glob.glob(os.path.join(exp_dir, "*absolute_results.txt"))
        
        if not result_files:
            print(f"{exp_dir:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue
        
        test_results = {}
        
        for result_file in result_files:
            filename = os.path.basename(result_file)
            
            # 确定测试名称
            if 'test1' in filename:
                test_name = 'Test-1'
            elif 'test2' in filename:
                test_name = 'Test-2'
            elif 'test3' in filename:
                test_name = 'Test-3'
            else:
                continue
            
            # 统计有效预测
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找所有预测值
            pattern = r'预测值: (.*?)\n'
            predictions = re.findall(pattern, content)
            
            valid_count = sum(1 for pred in predictions if pred != 'None')
            total_count = len(predictions)
            
            test_results[test_name] = f"{valid_count}/{total_count}"
            
            total_complete += valid_count
            total_samples += total_count
        
        # 显示结果
        test1_result = test_results.get('Test-1', 'N/A')
        test2_result = test_results.get('Test-2', 'N/A') 
        test3_result = test_results.get('Test-3', 'N/A')
        
        # 计算总计
        total_valid = 0
        total_all = 0
        for test_result in test_results.values():
            if test_result != 'N/A':
                valid, all_count = map(int, test_result.split('/'))
                total_valid += valid
                total_all += all_count
        
        total_result = f"{total_valid}/{total_all}" if total_all > 0 else 'N/A'
        
        print(f"{exp_dir:<60} {test1_result:<10} {test2_result:<10} {test3_result:<10} {total_result:<10}")
    
    print("="*100)
    print(f"{'总体进度':<60} {'':<10} {'':<10} {'':<10} {total_complete}/{total_samples}")
    print(f"完成率: {total_complete/total_samples*100:.1f}%" if total_samples > 0 else "完成率: N/A")
    
    # 显示需要重测的样本数
    failed_samples = total_samples - total_complete
    if failed_samples > 0:
        print(f"仍需重测: {failed_samples} 个样本")
        print(f"预估时间: {failed_samples * 15 / 60:.1f} 分钟")
    else:
        print("🎉 所有样本都已完成！")


if __name__ == "__main__":
    check_experiment_progress() 