# Gemini-2.5-Pro Ablation实验脚本总结

## 🎯 实验概述

已为您创建6个shell脚本，专门用于运行**Gemini-2.5-Pro模型**在三个物理属性任务上的ablation实验：

- **任务**: Friction, Viscosity, Elasticity
- **实验类型**: Absolute (绝对值预测) 和 Relative (相对比较)
- **样本数**: 每个实验100个样本
- **消融策略**: 单个prompting功能消融

## 📋 脚本列表

### Absolute实验（3个脚本，每个3个实验）

1. **`run_friction_absolute_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + Oracle
   - 3个实验，300个样本，预计2.0小时

2. **`run_viscosity_absolute_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + Oracle
   - 3个实验，300个样本，预计2.0小时

3. **`run_elasticity_absolute_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + Oracle
   - 3个实验，300个样本，预计2.0小时

### Relative实验（3个脚本，每个4个实验）

4. **`run_friction_relative_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + ICL, + Concat
   - 4个实验，400个样本，预计2.7小时

5. **`run_viscosity_relative_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + ICL, + Concat
   - 4个实验，400个样本，预计2.7小时

6. **`run_elasticity_relative_ablation_100samples.sh`**
   - Baseline, + FrameIndex, + ICL, + Concat
   - 4个实验，400个样本，预计2.7小时

## 📊 实验统计

- **总脚本数**: 6个
- **总实验数**: 21个（9个absolute + 12个relative）
- **总样本数**: 2,100个
- **预计总时间**: 14小时（串行）/ 4.7小时（并行）

## 🚀 快速开始

### 运行所有实验（并行推荐）

```bash
# 同时运行所有absolute实验（6小时）
./run_friction_absolute_ablation_100samples.sh &
./run_viscosity_absolute_ablation_100samples.sh &
./run_elasticity_absolute_ablation_100samples.sh &
wait

# 同时运行所有relative实验（2.7小时）
./run_friction_relative_ablation_100samples.sh &
./run_viscosity_relative_ablation_100samples.sh &
./run_elasticity_relative_ablation_100samples.sh &
wait
```

### 运行单个实验

```bash
# 运行单个任务
./run_friction_absolute_ablation_100samples.sh

# 查看进度
tail -f nohup.out  # 如果使用nohup运行
```

## 🎯 输出目录格式

所有结果将保存在以下格式的目录中：
```
gemini_physics_analysis_results_{type}_100samples[_{feature}]
```

示例：
- `gemini_physics_analysis_results_absolute_100samples`
- `gemini_physics_analysis_results_relative_100samples_frameindex`
- `gemini_physics_analysis_results_absolute_100samples_oracle`

## ⚡ 增强特性

- **激进重试模式**: 所有脚本使用 `--aggressive-retry` 确保95%+成功率
- **智能错误处理**: 自动处理API错误和网络问题
- **断点续传**: 支持实验中断后继续运行
- **详细日志**: 清晰的进度提示和错误信息

## 🔧 消融配置说明

### Absolute实验（除了ICL）
- **Baseline**: 标准prompt
- **+ FrameIndex**: 添加帧索引标识
- **+ Oracle**: 提供解题步骤指导

### Relative实验（除了Oracle）
- **Baseline**: 标准prompt
- **+ FrameIndex**: 添加帧索引标识
- **+ ICL**: 提供in-context learning示例
- **+ Concat**: 使用视频拼接模式

## ✅ 准备就绪

所有脚本已设置执行权限并准备运行。您可以立即开始实验！

建议使用并行运行方式以节省时间，总共约4.7小时即可完成所有21个实验。