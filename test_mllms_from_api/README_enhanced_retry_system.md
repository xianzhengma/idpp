# 增强重试系统使用指南

## 概述

为了在第一次运行时就尽量减少无效数据和失败预测，我们实现了一个多层次的增强重试系统。该系统在API调用和结果验证两个层面提供了强大的重试机制。

## 核心改进

### 1. 多层重试架构

```
验证重试循环 (validation_retries)
├── API调用重试循环 (max_retries)
│   ├── 第三方API内部重试 (gpt_api_third_party.py)
│   ├── 错误检测和分类
│   └── 智能等待策略
└── 结果验证和提取
    ├── 预测值提取验证
    ├── 比较结果提取验证
    └── 最佳结果保留
```

### 2. 智能错误处理

- **API错误检测**: 自动识别API调用失败、服务器错误等
- **空结果处理**: 检测并重试空响应
- **格式验证**: 确保能够提取到有效的预测值或比较结果
- **指数退避**: 根据错误类型采用不同的等待策略

### 3. 激进重试模式

启用 `--aggressive-retry` 时，系统会自动调整参数以最大化成功率：
- API重试次数: 至少8次
- 验证重试次数: 至少5次  
- 重试延迟: 至少20秒

## 命令行参数

### 基础重试参数

```bash
# API调用重试次数（默认：5）
--max-retries 8

# 重试间隔时间，单位秒（默认：15）
--retry-delay 20

# 结果验证重试次数（默认：3）
--validation-retries 5
```

### 激进重试模式

```bash
# 启用激进重试模式，自动优化所有重试参数
--aggressive-retry
```

## 使用示例

### 1. 标准模式（推荐用于常规测试）

```bash
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 50 \
    --model-version gpt-4o \
    --max-retries 5 \
    --retry-delay 15 \
    --validation-retries 3
```

### 2. 高可靠性模式（推荐用于重要实验）

```bash
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 50 \
    --model-version gpt-4o \
    --max-retries 8 \
    --retry-delay 20 \
    --validation-retries 5 \
    --icl-example \
    --follow-oracle-test
```

### 3. 激进模式（推荐用于最大化数据完整性）

```bash
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 50 \
    --model-version gpt-4o \
    --aggressive-retry \
    --icl-example \
    --follow-oracle-test
```

## 重试机制详解

### 验证重试循环

对于每个样本，系统会进行多轮验证重试：

1. **第1轮验证重试**: 尝试获取有效结果
2. **第2轮验证重试**: 如果第1轮失败，等待后重新尝试
3. **第N轮验证重试**: 继续重试直到达到 `validation_retries` 限制

每轮验证重试内部包含完整的API调用重试循环。

### API调用重试循环

在每轮验证重试中，系统会进行多次API调用重试：

1. **API调用**: 调用GPT API获取响应
2. **错误检测**: 检查是否为API错误、空响应等
3. **结果验证**: 尝试提取预测值或比较结果
4. **重试决策**: 根据错误类型决定是否重试

### 第三方API内部重试

`gpt_api_third_party.py` 中的重试机制：

- **HTTP错误处理**: 429(速率限制)、5xx(服务器错误)等
- **连接错误处理**: 超时、连接中断等
- **指数退避策略**: 根据错误类型和重试次数调整等待时间
- **智能等待**: 解析 `Retry-After` 头部信息

## 性能优化建议

### 1. 参数调优

根据API稳定性和网络条件调整参数：

```bash
# 网络不稳定环境
--max-retries 10 --retry-delay 30 --validation-retries 3

# API限制较严格环境  
--max-retries 5 --retry-delay 60 --validation-retries 2

# 高速稳定环境
--max-retries 3 --retry-delay 10 --validation-retries 2
```

### 2. 批量处理策略

```bash
# 分批处理大量样本
--absolute-samples 20 --relative-pairs 20 --aggressive-retry

# 然后使用retry脚本处理失败的样本
python retry_failed_predictions.py
```

### 3. 模型选择影响

不同模型的稳定性和响应格式可能影响重试效果：

```bash
# GPT-4o (推荐，稳定性好)
--model-version gpt-4o --max-retries 5

# 其他模型可能需要更多重试
--model-version qwen-vl-max --max-retries 8 --validation-retries 5
```

## 监控和调试

### 1. 日志输出

系统会输出详细的重试日志：

```
🔄 验证重试 1/3
  📡 API调用 1/5
  ✅ 成功获得有效预测值: 0.75
```

### 2. 错误分类

常见错误类型和处理方式：

- `⚠️ API返回空结果`: API响应为空，将重试
- `❌ 检测到API错误响应`: API返回错误信息，将重试
- `⚠️ 无法从响应中提取预测值`: 响应格式问题，将重试
- `✅ 成功获得有效预测值`: 成功获取结果

### 3. 失败处理

即使所有重试都失败，系统也会：

1. 保留最佳响应作为备用
2. 记录详细错误信息
3. 继续处理下一个样本

## 预期效果

使用增强重试系统后，预期可以达到：

- **第一次运行成功率**: 95%+ (相比之前的85%)
- **减少后续重试需求**: 大幅减少需要使用 `retry_failed_predictions.py` 的情况
- **提高数据质量**: 更多有效的预测值和比较结果
- **降低人工干预**: 自动处理大部分临时性错误

## 注意事项

1. **API费用**: 更多重试意味着更多API调用，请注意费用控制
2. **运行时间**: 重试会增加总运行时间，特别是在激进模式下
3. **网络稳定性**: 在网络不稳定环境下，适当增加延迟时间
4. **API限制**: 注意API的速率限制，避免过于频繁的重试

## 故障排除

### 常见问题

1. **重试次数过多但仍失败**
   - 检查API密钥是否有效
   - 检查网络连接
   - 尝试减少并发请求

2. **提取不到预测值**
   - 检查模型响应格式
   - 调整提示词
   - 增加验证重试次数

3. **运行时间过长**
   - 减少重试次数
   - 减少延迟时间
   - 分批处理样本

### 调试命令

```bash
# 测试单个属性的重试效果
python run_gpt_physics_tests.py \
    --properties elasticity \
    --absolute-samples 5 \
    --aggressive-retry \
    --auto-confirm

# 查看详细日志
python run_gpt_physics_tests.py [...] 2>&1 | tee retry_test.log
```

通过这个增强的重试系统，您应该能够在第一次运行时就获得更高质量和更完整的实验数据。 