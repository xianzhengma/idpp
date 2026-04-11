# Elasticity Relative Ablation Study

本文档介绍如何运行elasticity relative ablation study实验并分析结果。

## 📋 实验概述

本ablation study包含11个组合实验，测试四个主要功能及其组合对elasticity relative预测性能的影响：

**注意**: 本脚本跳过单个功能实验（假设已完成），只执行组合实验。

### 已跳过的单个功能实验（假设已完成）
- **Baseline**: 标准prompt
- **+ Concat**: 拼接模式（视频间加黑帧分隔）
- **+ ICL**: In-Context Learning示例
- **+ Oracle**: 物理分析步骤指导
- **+ FrameIndex**: 帧索引标识

### 本次执行的组合实验（11个）

#### 两两组合（6个）
1. **+ Concat + ICL**: 拼接模式 + ICL示例
2. **+ Concat + Oracle**: 拼接模式 + Oracle指导
3. **+ Concat + FrameIndex**: 拼接模式 + 帧索引
4. **+ ICL + Oracle**: ICL示例 + Oracle指导
5. **+ ICL + FrameIndex**: ICL示例 + 帧索引
6. **+ Oracle + FrameIndex**: Oracle指导 + 帧索引

#### 三功能组合（4个）
7. **+ Concat + ICL + Oracle**: 拼接模式 + ICL示例 + Oracle指导
8. **+ Concat + ICL + FrameIndex**: 拼接模式 + ICL示例 + 帧索引
9. **+ Concat + Oracle + FrameIndex**: 拼接模式 + Oracle指导 + 帧索引
10. **+ ICL + Oracle + FrameIndex**: ICL示例 + Oracle指导 + 帧索引

#### 四功能组合（1个）
11. **+ Concat + ICL + Oracle + FrameIndex**: 所有功能的完整组合

每个实验使用**20对样本**，目标属性为**elasticity**，测试类型为**relative**。

## 🚀 快速开始

### 1. 运行组合实验

```bash
cd test_mllms_from_api
./run_elasticity_relative_ablation.sh
```

**预计时间**: 约5.5小时  
**模型**: gemini-2.5-pro  
**样本数**: 每个实验20对样本

### 2. 处理结果

实验完成后运行：

```bash
cd test_mllms_from_api
python process_elasticity_relative_ablation.py
```

这将生成：
- 详细的对比表格（控制台显示）
- CSV结果文件：`elasticity_relative_20samples_ablation_results_YYYYMMDD_HHMMSS.csv`
- 相对于Baseline的性能分析

## 📊 结果目录结构

实验完成后将产生以下目录结构：

```
# 本次执行的实验
gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl/              # + Concat + ICL
gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle/           # + Concat + Oracle
gemini_physics_analysis_results_elasticity_relative_20samples_concat_frameindex/       # + Concat + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle/              # + ICL + Oracle
gemini_physics_analysis_results_elasticity_relative_20samples_icl_frameindex/          # + ICL + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_oracle_frameindex/       # + Oracle + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle/       # + Concat + ICL + Oracle
gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_frameindex/   # + Concat + ICL + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_concat_oracle_frameindex/# + Concat + Oracle + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_icl_oracle_frameindex/   # + ICL + Oracle + FrameIndex
gemini_physics_analysis_results_elasticity_relative_20samples_concat_icl_oracle_frameindex/ # + All Features

# 假设已存在的单个功能实验
gemini_physics_analysis_results_elasticity_relative_20samples/                      # Baseline
gemini_physics_analysis_results_elasticity_relative_20samples_concat/               # + Concat
gemini_physics_analysis_results_elasticity_relative_20samples_icl/                  # + ICL
gemini_physics_analysis_results_elasticity_relative_20samples_oracle/               # + Oracle
gemini_physics_analysis_results_elasticity_relative_20samples_frameindex/           # + FrameIndex
```

每个目录包含：
- `v125_vjepa_test1_relative_results.txt`
- `v125_vjepa_test2_relative_results.txt` 
- `v125_vjepa_test3_relative_results.txt`

## 📈 评估指标

结果表格包含以下指标：

- **AUC** (Area Under ROC Curve): ROC曲线下面积，越接近1越好
- **Acc** (Accuracy): 准确率，越接近1越好
- **Valid**: 有效预测数/总样本数

每个实验都会计算Test-1、Test-2、Test-3的指标以及平均值。

## 🎯 Elasticity Relative特殊说明

### 相对比较任务
- 给定两个弹性视频，判断哪个具有更高的弹性系数
- 输出格式：`Comparison result: 1`（视频1弹性更高）或`Comparison result: 0`（视频2弹性更高）

### Concat模式
- 将两个视频用黑帧分隔后拼接成一个序列
- 提示词会说明黑帧前后分别是视频1和视频2

### ICL示例
- 从训练集中采样1对视频作为示例（相对比较测试自动限制为1个示例以避免API限制）
- 展示完整的输入-输出格式

### Oracle指导
- 提供详细的弹性分析步骤（球的轨迹、撞击、弹跳分析）
- 指导模型如何计算恢复系数

### 数据集信息
- **训练集**: v125_vjepa使用`5.24_elasticity_synthetic_train`
- **测试集**: test1/test2使用`5.24_elasticity_synthetic_test1/test2`，test3使用`2.16_elasticity_test3`
- **GT文件**: 使用`gt_retitution1.json`

## 🔧 自定义参数

如需修改参数，可以编辑 `run_elasticity_relative_ablation.sh`：

```bash
# 修改样本数
--relative-pairs 20      # 改为其他数值

# 修改模型版本  
--model-version gemini-2.5-pro    # 改为gpt-4o等

# 修改目标属性
--properties elasticity    # 改为friction或viscosity
```

## 🎯 预期结果

典型的实验结果表格示例：

```
Experiment                 | Test-1_AUC | Test-2_AUC | Test-3_AUC | Avg_AUC    | Avg_Acc
---------------------------|------------|------------|------------|------------|----------
+ Concat + ICL            | 0.7234     | 0.7456     | 0.7123     | 0.7271     | 0.6850
+ Concat + Oracle         | 0.7456     | 0.7678     | 0.7345     | 0.7493     | 0.7100
+ ICL + Oracle            | 0.7678     | 0.7890     | 0.7567     | 0.7712     | 0.7250
...
```

## 💡 使用提示

1. **断点续传**: 如果实验中断，脚本会自动跳过已完成的实验
2. **资源监控**: 实验过程中会有详细的进度输出
3. **API限制**: ICL相对比较测试自动使用1个示例以避免API大小限制
4. **结果备份**: 建议及时保存生成的CSV文件

## 🔍 故障排查

### 常见问题

1. **找不到结果目录**: 确保实验已完成且目录名称正确
2. **CSV生成失败**: 检查pandas和sklearn是否安装
3. **API限制**: 如遇到API限制，脚本会自动重试

### 日志查看

每个实验的详细日志保存在各自的结果目录中，可以查看具体的错误信息。

## 📝 注意事项

- 确保有足够的API配额（约220次调用，每次约$0.01-0.02）
- 网络连接稳定，避免长时间中断
- 实验期间避免修改相关文件和目录
- 如果单个功能实验未完成，需要先运行这些实验 