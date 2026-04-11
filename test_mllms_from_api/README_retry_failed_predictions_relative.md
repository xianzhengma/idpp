# Elasticity Relative Ablation 失败预测重测系统

## 概述

本系统用于重新测试elasticity relative ablation实验中的失败预测（预测结果为None的比较），并将重测成功的结果合并回原始结果文件，确保每个实验的比较数据尽可能完整。

## 核心脚本

### 1. `retry_failed_predictions_relative.py`
主要的重测脚本，功能包括：
- 自动识别失败的比较预测
- 根据实验配置重新调用GPT进行视频比较
- 将成功的重测结果合并回原始文件
- 支持试运行模式统计失败比较数量

### 2. `check_progress_relative.py`
进度检查脚本，快速显示所有relative实验的完成情况

### 3. `process_elasticity_relative_ablation.py`（现有）
原始的结果处理脚本，用于生成最终的对比表格和性能分析

## 使用方法

### 试运行模式（统计失败比较）
```bash
python retry_failed_predictions_relative.py --dry-run
```

### 重测所有实验
```bash
python retry_failed_predictions_relative.py
```

### 重测指定实验
```bash
python retry_failed_predictions_relative.py --experiments gemini_physics_analysis_results_elasticity_relative_20samples_oracle
```

### 检查进度
```bash
python check_progress_relative.py
```

### 生成最终结果表格
```bash
python process_elasticity_relative_ablation.py
```

## 实验配置

系统支持以下5个实验配置：

| 实验名称 | Frame Index | ICL | Oracle | Concat Mode | 描述 |
|---------|-------------|-----|--------|-------------|------|
| Baseline | ❌ | ❌ | ❌ | ❌ | 基础配置 |
| + FrameIndex | ✅ | ❌ | ❌ | ❌ | 添加帧索引标识 |
| + ICL | ❌ | ✅ | ❌ | ❌ | 添加上下文学习示例 |
| + Oracle | ❌ | ❌ | ✅ | ❌ | 添加解题步骤指导 |
| + Concat | ❌ | ❌ | ❌ | ✅ | 使用视频拼接模式 |

## 重测结果统计

### 初始状态（重测前）
- **总比较数**: 240个
- **失败比较数**: 23个
- **整体成功率**: 90.4%

### 当前状态（重测后）
- **总比较数**: 240个
- **失败比较数**: 17个
- **整体成功率**: 92.9%
- **改善情况**: 成功重测了6个比较

### 各实验完成情况

| 实验 | 完成度 | 状态 |
|------|--------|------|
| Baseline | 59/60 | 🔄 几乎完成 |
| + FrameIndex | 58/60 | 🔄 几乎完成 |
| + ICL | N/A | ❌ 缺少结果文件 |
| + Oracle | 46/60 | 🔄 需要改进 |
| + Concat | 60/60 | ✅ 完成 |

## 技术特性

### 相对比较处理
- 专门处理两个视频之间的弹性比较任务
- 支持不同的比较模式（标准模式 vs 拼接模式）
- 自动解析视频对信息和比较结果

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
├── retry_failed_predictions_relative.py     # 主重测脚本
├── check_progress_relative.py               # 进度检查脚本
├── process_elasticity_relative_ablation.py  # 结果处理脚本
├── README_retry_failed_predictions_relative.md  # 本说明文档
└── gemini_physics_analysis_results_elasticity_relative_20samples*/
    ├── v125_vjepa_test1_relative_results.txt
    ├── v125_vjepa_test2_relative_results.txt
    ├── v125_vjepa_test3_relative_results.txt
    └── *.backup_*                           # 自动生成的备份文件
```

## 性能分析

基于重测后的数据，主要发现：

1. **Oracle在AUC上表现最佳**
   - AUC改善8.6%，达到0.8044
   - 但准确率下降17.7%，可能过度依赖oracle指导

2. **Concat模式相对均衡**
   - AUC略有改善（+2.4%）
   - 数据完整性最好（60/60）

3. **FrameIndex表现较差**
   - 所有指标都有明显下降
   - 可能帧索引对相对比较任务不够有效

4. **Baseline仍然是最均衡的选择**
   - 在准确率、精确度、F1等指标上表现最好
   - 数据完整性良好（59/60）

## 数据完整性问题

### Test-3数据目录缺失
某些实验的Test-3数据目录不存在：
- `/scratch/shared/beegfs/guanqi/genesis_data/5.24_elasticity_synthetic_test3`

这导致Test-3的失败比较无法重测，影响了整体完成率。

### ICL实验缺失
`gemini_physics_analysis_results_elasticity_relative_20samples_icl`目录存在但缺少结果文件，可能需要重新运行该实验。

## 注意事项

1. **API限制**: 每个比较重测间隔15秒，避免API速率限制
2. **数据完整性**: Test-3的某些视频文件可能不存在，系统会自动跳过
3. **时间成本**: 完整重测所有失败比较预计需要4.2分钟
4. **备份管理**: 定期清理备份文件以节省存储空间

## 相对于Absolute实验的差异

1. **更高的基础成功率**: Relative实验的初始成功率（90.4%）明显高于Absolute实验（79.4%）
2. **更少的失败样本**: 只需重测23个比较 vs Absolute的99个样本
3. **不同的评估指标**: 使用AUC、准确率等分类指标，而不是MSE、MAE等回归指标
4. **比较任务的复杂性**: 需要处理两个视频的相对比较，而不是单个视频的绝对值预测

## 后续改进建议

1. **解决数据缺失问题**: 确保所有测试集的数据目录完整可用
2. **补充ICL实验**: 重新运行缺失的ICL实验
3. **优化Oracle配置**: 调整Oracle指导以平衡AUC和准确率
4. **并行处理**: 实现多线程重测以提高效率
5. **结果验证**: 添加重测结果的质量检查机制 