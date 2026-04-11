#!/usr/bin/env python3
"""
测试批量处理实验结果的功能
"""

import os
import glob
import pandas as pd
import numpy as np

def get_experiment_name_from_dir(dir_path):
    """
    从目录名中提取实验名称
    """
    dir_name = os.path.basename(dir_path)
    
    # 定义实验名称映射
    experiment_mapping = {
        'gemini_physics_analysis_results': 'Baseline',
        'gemini_physics_analysis_results_icl-example': 'ICL',
        'gemini_physics_analysis_results_follow-oracle-test': 'Oracle',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test': 'ICL+Oracle',
        'gemini_physics_analysis_results_concat-mode': 'Concat',
        'gemini_physics_analysis_results_frame-index': 'FrameIndex',
        'gemini_physics_analysis_results_icl-example_concat-mode': 'ICL+Concat',
        'gemini_physics_analysis_results_follow-oracle-test_concat-mode': 'Oracle+Concat',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index': 'ICL+Oracle+FrameIndex',
        'gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index': 'AllFeatures'
    }
    
    return experiment_mapping.get(dir_name, dir_name)

def main():
    print("测试批量处理实验结果功能...")
    
    # 自动发现所有gemini_physics_analysis_results*目录
    pattern = "gemini_physics_analysis_results*"
    base_dirs = glob.glob(pattern)
    
    print(f"搜索模式: {pattern}")
    print(f"当前工作目录: {os.getcwd()}")
    
    if not base_dirs:
        print(f"未找到匹配的目录: {pattern}")
        
        # 显示当前目录下的所有目录
        all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        print(f"当前目录下的所有目录: {all_dirs}")
        
        # 检查是否有相关的目录
        related_dirs = [d for d in all_dirs if 'gemini' in d.lower() or 'physics' in d.lower() or 'analysis' in d.lower()]
        if related_dirs:
            print(f"找到相关目录: {related_dirs}")
        
        return
    
    base_dirs.sort()  # 按字母顺序排序
    
    print(f"\n发现 {len(base_dirs)} 个实验目录:")
    for dir_path in base_dirs:
        experiment_name = get_experiment_name_from_dir(dir_path)
        print(f"  - {dir_path} -> {experiment_name}")
        
        # 检查目录内容
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            result_files = [f for f in files if f.endswith('_results.txt')]
            print(f"    包含 {len(result_files)} 个结果文件")
        else:
            print(f"    警告: 不是目录")
    
    # 创建模拟的对比表格
    print(f"\n生成模拟的实验对比表格...")
    
    # 模拟数据
    experiments = [get_experiment_name_from_dir(d) for d in base_dirs]
    properties = ['viscosity', 'elasticity']
    test_sets = ['test1', 'test2', 'test3']
    
    # 生成随机模拟数据用于演示
    np.random.seed(42)
    
    print(f"\n模拟相对比较结果 (AUC):")
    print("-" * 60)
    
    for prop in properties:
        print(f"\n{prop.upper()} - 相对比较 (AUC):")
        
        # 创建模拟数据
        data = {}
        for exp in experiments:
            data[exp] = [np.random.uniform(0.5, 0.9) for _ in test_sets]
        
        df = pd.DataFrame(data, index=['Test-1', 'Test-2', 'Test-3'])
        df = df.T  # 转置，让实验成为行
        
        # 计算平均值
        df['Average'] = df.mean(axis=1)
        df = df.round(4)
        
        # 按平均值排序
        df = df.sort_values('Average', ascending=False)
        
        print(df.to_string())
    
    print(f"\n脚本功能验证完成！")
    print(f"实际使用时，请运行: python process_individual_model_results_ablation.py")

if __name__ == "__main__":
    main() 