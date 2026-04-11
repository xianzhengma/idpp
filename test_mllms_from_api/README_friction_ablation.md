# Friction Absolute Ablation Study

本文档介绍如何运行完整的friction absolute ablation study实验并分析结果。

## 📋 实验概述

本ablation study包含8个实验（包含baseline），测试三个主要功能及其组合对friction absolute预测性能的影响：

### Baseline（1个）
1. **Baseline**: 标准prompt，作为所有实验的基准

### 单个功能（3个）
2. **+ FrameIndex**: 为每帧添加显式索引标识（frame1:, frame2:, ...）
3. **+ ICL**: 使用In-Context Learning，从训练集采样示例
4. **+ Oracle**: 提供具体的物理分析步骤指导

### 两两组合（3个）
5. **+ FrameIndex + ICL**: 帧索引 + ICL示例
6. **+ FrameIndex + Oracle**: 帧索引 + Oracle指导  
7. **+ ICL + Oracle**: ICL示例 + Oracle指导

### 三功能组合（1个）
8. **+ FrameIndex + ICL + Oracle**: 所有功能的完整组合

每个实验使用**20个样本**，目标属性为**friction**，测试类型为**absolute**。

## 🚀 快速开始

### 1. 运行所有实验

```bash
cd test_mllms_from_api
./run_friction_absolute_ablation.sh
```

**预计时间**: 约4.0小时  
**模型**: gemini-2.5-pro  
**样本数**: 每个实验20个样本

### 2. 处理结果

实验完成后运行：

```bash
cd test_mllms_from_api
python process_friction_absolute_ablation.py
```

这将生成：
- 详细的对比表格（控制台显示）
- CSV结果文件：`friction_absolute_ablation_results_YYYYMMDD_HHMMSS.csv`
- 相对于Baseline的性能分析

## 📊 结果目录结构

实验完成后将产生以下目录结构：

```
gemini_physics_analysis_results_friction_absolute_20samples/                    # Baseline
gemini_physics_analysis_results_friction_absolute_20samples_frameindex/         # + FrameIndex
gemini_physics_analysis_results_friction_absolute_20samples_icl/                # + ICL
gemini_physics_analysis_results_friction_absolute_20samples_oracle/             # + Oracle
gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl/     # + FrameIndex + ICL
gemini_physics_analysis_results_friction_absolute_20samples_frameindex_oracle/  # + FrameIndex + Oracle
gemini_physics_analysis_results_friction_absolute_20samples_icl_oracle/         # + ICL + Oracle
gemini_physics_analysis_results_friction_absolute_20samples_frameindex_icl_oracle/  # + FrameIndex + ICL + Oracle
```

每个目录包含：
- `v120_vjepa_test1_absolute_results.txt`
- `v120_vjepa_test2_absolute_results.txt` 
- `v120_vjepa_test3_absolute_results.txt`

## 📈 评估指标

结果表格包含以下指标：

- **MSE** (Mean Squared Error): 均方误差，越小越好
- **MAE** (Mean Absolute Error): 平均绝对误差，越小越好  
- **Pearson**: 皮尔逊相关系数，越接近1越好
- **Valid**: 有效预测数/总样本数

每个实验都会计算Test-1、Test-2、Test-3的指标以及平均值。

## 🔧 自定义参数

如需修改参数，可以编辑 `run_friction_absolute_ablation.sh`：

```bash
# 修改样本数
--absolute-samples 20    # 改为其他数值

# 修改模型版本  
--model-version gemini-2.5-pro    # 改为gpt-4o等

# 修改目标属性
--properties friction    # 改为viscosity或elasticity
```

## 🎯 预期结果

典型的实验结果表格示例：

```
Experiment                 | Test-1_MSE | Test-2_MSE | Test-3_MSE | Avg_MSE    | Avg_Pearson
---------------------------|------------|------------|------------|------------|------------
Baseline                   | 0.123456   | 0.134567   | 0.145678   | 0.134567   | 0.6789
+ FrameIndex              | 0.112345   | 0.123456   | 0.134567   | 0.123456   | 0.7123
+ ICL                     | 0.109876   | 0.120987   | 0.132098   | 0.120987   | 0.7456
+ Oracle                  | 0.098765   | 0.109876   | 0.120987   | 0.109876   | 0.7789
...
```

## 💡 使用提示

1. **断点续传**: 如果实验中断，脚本会自动跳过已完成的实验
2. **资源监控**: 实验过程中会有详细的进度输出
3. **错误处理**: 单个实验失败不会影响其他实验
4. **结果备份**: 建议及时保存生成的CSV文件

## 🔍 故障排查

### 常见问题

1. **找不到结果目录**: 确保实验已完成且目录名称正确
2. **CSV生成失败**: 检查pandas和sklearn是否安装
3. **API限制**: 如遇到API限制，脚本会自动重试

### 日志查看

每个实验的详细日志保存在各自的结果目录中，可以查看具体的错误信息。

## 📝 注意事项

- 确保有足够的API配额（约140次调用，每次约$0.01-0.02）
- 网络连接稳定，避免长时间中断
- 实验期间避免修改相关文件和目录
- 定期备份重要结果 