# Elasticity Absolute Ablation 失败预测重测系统

## 概述

本系统用于重新测试elasticity absolute ablation实验中的失败预测（预测值为None的样本），并将重测成功的结果合并回原始结果文件，确保每个实验的60个样本全部有效。

## 核心脚本

### 1. `retry_failed_predictions.py`
主要的重测脚本，功能包括：
- 自动识别失败的预测样本
- 根据实验配置重新调用GPT进行预测
- 将成功的重测结果合并回原始文件
- 支持试运行模式统计失败样本数量

### 2. `check_progress.py`
进度检查脚本，快速显示所有实验的完成情况

### 3. `process_elasticity_absolute_ablation.py`（已修复）
原始的结果处理脚本，修复了文件查找模式问题

## 使用方法

### 试运行模式（统计失败样本）
```bash
python retry_failed_predictions.py --dry-run
```

### 重测所有实验
```bash
python retry_failed_predictions.py
```

### 重测指定实验
```bash
python retry_failed_predictions.py --experiments gemini_physics_analysis_results_elasticity_absolute_20samples
```

### 检查进度
```bash
python check_progress.py
```

### 生成最终结果表格
```bash
python process_elasticity_absolute_ablation.py
```

## 实验配置

系统支持以下8个实验配置：

| 实验名称 | Frame Index | ICL | Oracle | 描述 |
|---------|-------------|-----|--------|------|
| Baseline | ❌ | ❌ | ❌ | 基础配置 |
| + FrameIndex | ✅ | ❌ | ❌ | 添加帧索引标识 |
| + ICL | ❌ | ✅ | ❌ | 添加上下文学习示例 |
| + Oracle | ❌ | ❌ | ✅ | 添加解题步骤指导 |
| + FrameIndex+ICL | ✅ | ✅ | ❌ | 帧索引 + ICL |
| + FrameIndex+Oracle | ✅ | ❌ | ✅ | 帧索引 + Oracle |
| + ICL+Oracle | ❌ | ✅ | ✅ | ICL + Oracle |
| + FrameIndex+ICL+Oracle | ✅ | ✅ | ✅ | 全部功能 |

## 重测结果统计

### 初始状态（重测前）
- **总样本数**: 480个
- **失败样本数**: 99个
- **整体成功率**: 79.4%

### 当前状态（重测后）
- **总样本数**: 480个
- **失败样本数**: 73个
- **整体成功率**: 84.8%
- **改善情况**: 成功重测了26个样本

### 各实验完成情况

| 实验 | 完成度 | 状态 |
|------|--------|------|
| Baseline | 60/60 | ✅ 完成 |
| + FrameIndex | 60/60 | ✅ 完成 |
| + ICL | 60/60 | ✅ 完成 |
| + FrameIndex+ICL | 60/60 | ✅ 完成 |
| + Oracle | 55/60 | 🔄 进行中 |
| + ICL+Oracle | 50/60 | 🔄 进行中 |
| + FrameIndex+Oracle | 37/60 | 🔄 进行中 |
| + FrameIndex+ICL+Oracle | 25/60 | 🔄 进行中 |

## 技术特性

### 自动备份
- 每次更新结果文件前自动创建时间戳备份
- 备份文件格式: `原文件名.backup_YYYYMMDD_HHMMSS`

### 智能错误处理
- 自动跳过不存在的视频文件
- 支持ICL示例生成失败时的降级处理
- API调用失败时的重试机制

### 配置自适应
- 根据实验目录名自动识别实验配置
- 动态生成ICL示例（如需要）
- 自动匹配数据目录路径

### 进度追踪
- 实时显示重测进度
- 统计成功率和预估剩余时间
- 详细的日志输出

## 文件结构

```
test_mllms_from_api/
├── retry_failed_predictions.py          # 主重测脚本
├── check_progress.py                     # 进度检查脚本
├── process_elasticity_absolute_ablation.py  # 结果处理脚本（已修复）
├── README_retry_failed_predictions.md   # 本说明文档
└── gemini_physics_analysis_results_elasticity_absolute_20samples*/
    ├── v124_vjepa_test1_absolute_results.txt
    ├── v124_vjepa_test2_absolute_results.txt
    ├── v124_vjepa_test3_absolute_results.txt
    └── *.backup_*                        # 自动生成的备份文件
```

## 性能分析

基于重测后的完整数据，主要发现：

1. **ICL是最有效的改进方法**
   - MSE改善65.3%
   - MAE改善33.7%
   - Pearson相关系数改善65.5%

2. **Oracle功能反而降低了性能**
   - 可能由于过度指导导致模型过拟合

3. **FrameIndex有适度改善**
   - 在Pearson相关系数上有32.4%的改善

4. **组合多个特征通常不如单独使用ICL**
   - 表明特征间可能存在负面交互

## 注意事项

1. **API限制**: 每个样本重测间隔15秒，避免API速率限制
2. **数据完整性**: 某些Test-3的视频文件可能不存在，系统会自动跳过
3. **时间成本**: 完整重测所有失败样本预计需要18.2分钟
4. **备份管理**: 定期清理备份文件以节省存储空间

## 后续改进建议

1. **并行处理**: 实现多线程重测以提高效率
2. **智能重试**: 根据失败原因调整重试策略
3. **结果验证**: 添加重测结果的质量检查机制
4. **配置优化**: 基于成功率动态调整实验配置 