# 100样本Ablation实验脚本说明 (Gemini Only)

## 概述

本文档描述了6个shell脚本，用于在三个任务（friction, viscosity, elasticity）上运行Gemini-2.5-Pro模型的ablation实验。每个实验使用100个样本，只消融单个prompting功能。

## 脚本列表

### Absolute实验（3个脚本）

每个脚本包含3个实验（Gemini模型的3种配置）：

1. **`run_friction_absolute_ablation_100samples.sh`**
   - 任务：Friction Absolute
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + Oracle（除了ICL）
   - 总实验数：3个
   - 预计时间：2.0小时

2. **`run_viscosity_absolute_ablation_100samples.sh`**
   - 任务：Viscosity Absolute
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + Oracle（除了ICL）
   - 总实验数：3个
   - 预计时间：2.0小时

3. **`run_elasticity_absolute_ablation_100samples.sh`**
   - 任务：Elasticity Absolute
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + Oracle（除了ICL）
   - 总实验数：3个
   - 预计时间：2.0小时

### Relative实验（3个脚本）

每个脚本包含4个实验（Gemini模型的4种配置）：

4. **`run_friction_relative_ablation_100samples.sh`**
   - 任务：Friction Relative
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + ICL, + Concat（除了Oracle）
   - 总实验数：4个
   - 预计时间：2.7小时

5. **`run_viscosity_relative_ablation_100samples.sh`**
   - 任务：Viscosity Relative
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + ICL, + Concat（除了Oracle）
   - 总实验数：4个
   - 预计时间：2.7小时

6. **`run_elasticity_relative_ablation_100samples.sh`**
   - 任务：Elasticity Relative
   - 模型：Gemini-2.5-Pro
   - 配置：Baseline, + FrameIndex, + ICL, + Concat（除了Oracle）
   - 总实验数：4个
   - 预计时间：2.7小时

## 实验配置说明

### Absolute实验配置

- **Baseline**: 标准prompt，不使用任何增强功能
- **+ FrameIndex**: 为每帧添加显式索引标识（frame1:, frame2:等）
- **+ Oracle**: 提供解题步骤指导，告诉模型具体的分析方法

### Relative实验配置

- **Baseline**: 标准prompt，不使用任何增强功能
- **+ FrameIndex**: 为每帧添加显式索引标识
- **+ ICL**: 提供in-context learning示例
- **+ Concat**: 使用拼接模式比较视频（两视频间用黑帧分隔）

## 输出目录命名规则

所有输出目录都遵循以下命名格式：
```
gemini_physics_analysis_results_{type}_100samples[_{feature}]
```

示例：
- `gemini_physics_analysis_results_absolute_100samples`
- `gemini_physics_analysis_results_relative_100samples_frameindex`
- `gemini_physics_analysis_results_absolute_100samples_oracle`

## 使用方法

### 单独运行脚本

```bash
# 运行friction absolute实验
./run_friction_absolute_ablation_100samples.sh

# 运行elasticity relative实验
./run_elasticity_relative_ablation_100samples.sh
```

### 批量运行所有脚本

```bash
# 按顺序运行所有absolute实验
./run_friction_absolute_ablation_100samples.sh
./run_viscosity_absolute_ablation_100samples.sh
./run_elasticity_absolute_ablation_100samples.sh

# 按顺序运行所有relative实验
./run_friction_relative_ablation_100samples.sh
./run_viscosity_relative_ablation_100samples.sh
./run_elasticity_relative_ablation_100samples.sh
```

### 并行运行（推荐）

可以同时运行多个脚本以节省时间：

```bash
# 方案1：同时运行所有absolute实验
./run_friction_absolute_ablation_100samples.sh &
./run_viscosity_absolute_ablation_100samples.sh &
./run_elasticity_absolute_ablation_100samples.sh &
wait

# 方案2：同时运行所有relative实验
./run_friction_relative_ablation_100samples.sh &
./run_viscosity_relative_ablation_100samples.sh &
./run_elasticity_relative_ablation_100samples.sh &
wait
```

## 实验统计

### 总体统计
- **总脚本数**: 6个
- **总实验数**: 21个（9个absolute + 12个relative）
- **总样本数**: 2,100个
- **预计总时间**: 14小时（串行）/ 4.7小时（并行）

### 按模型统计
Gemini-2.5-Pro模型将运行：
- Absolute实验：9个（3个任务 × 3种配置）
- Relative实验：12个（3个任务 × 4种配置）
- 总计：21个实验，2,100个样本

### 按任务统计
每个任务将运行：
- Absolute实验：3个（1个模型 × 3种配置）
- Relative实验：4个（1个模型 × 4种配置）
- 总计：7个实验，700个样本

## 增强重试系统

所有脚本都使用了 `--aggressive-retry` 参数，确保：
- API重试次数：8次
- 验证重试次数：5次
- 重试延迟：20秒
- 第一次运行成功率：95%+

## 监控和管理

### 检查进度

```bash
# 检查正在运行的实验
ps aux | grep run_gpt_physics_tests

# 检查输出目录
ls -la *_physics_analysis_results_*

# 检查特定实验的进度
tail -f nohup.out  # 如果使用nohup运行
```

### 结果处理

每个实验完成后，可以使用相应的处理脚本分析结果：

```bash
# 处理absolute实验结果
python process_friction_absolute_ablation.py
python process_viscosity_absolute_ablation.py  
python process_elasticity_absolute_ablation.py

# 处理relative实验结果
python process_friction_relative_ablation.py
python process_viscosity_relative_ablation.py
python process_elasticity_relative_ablation.py
```

## 注意事项

1. **API配额**: 确保有足够的API调用配额
2. **存储空间**: 每个实验约需要1GB存储空间
3. **网络稳定性**: 建议在稳定的网络环境下运行
4. **并发限制**: 避免同时运行过多实验以防止API限制
5. **日志记录**: 建议使用nohup或screen来保持长时间运行

## 故障排除

如果实验中断或失败：

1. **检查日志**: 查看错误信息
2. **重新运行**: 脚本支持断点续传
3. **使用重试脚本**: 对失败的样本进行重测
4. **检查API状态**: 确认API服务正常

```bash
# 重测失败的样本
python retry_failed_predictions.py --experiments [experiment_dir]
```