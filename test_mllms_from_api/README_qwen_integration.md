# Qwen API 集成使用说明

## 概述

本框架已经集成了Qwen Vision API支持，现在可以同时使用GPT和Qwen模型进行物理属性分析。

## 环境配置

### 1. 安装依赖
```bash
pip install openai
```

### 2. 设置API密钥

需要设置环境变量 `DASHSCOPE_API_KEY`：

```bash
export DASHSCOPE_API_KEY="your_qwen_api_key"
```

或者在代码中直接传入API密钥。

## 支持的模型

### GPT模型
- `gpt-4o` (默认)
- `gpt-4o-mini`
- `gpt-4-turbo`

### Qwen模型
- `qwen-vl-max` (推荐)
- `qwen-vl-plus`
- `qwen-vl-turbo`

## 使用方法

### 1. 运行完整测试

```bash
# 使用GPT模型（默认）
python run_gpt_physics_tests.py

# 使用Qwen模型
python run_gpt_physics_tests.py --model-version qwen-vl-max

# 自定义参数
python run_gpt_physics_tests.py \
    --model-version qwen-vl-max \
    --absolute-samples 30 \
    --relative-pairs 30 \
    --output-dir my_qwen_results
```

### 2. 直接使用API

```python
from qwen_api_wrapper import qwen_response

# 准备消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请分析这些视频帧的物理属性"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }
]

# 调用API
response = qwen_response(messages, model_version="qwen-vl-max")
print(response)
```

### 3. 测试API连接

```python
from qwen_api_wrapper import QwenVisionAPI

# 创建API实例
api = QwenVisionAPI()

# 测试连接
if api.test_connection():
    print("API连接成功")
else:
    print("API连接失败")
```

## 输出目录结构

不同模型的结果会保存在不同的目录中：

```
├── gpt_physics_analysis_results/          # GPT模型结果
├── qwen_max_physics_analysis_results/     # Qwen-VL-Max结果
├── qwen_plus_physics_analysis_results/    # Qwen-VL-Plus结果
└── qwen_turbo_physics_analysis_results/   # Qwen-VL-Turbo结果
```

## 特点

1. **无缝集成**：与现有GPT框架完全兼容
2. **自动格式转换**：自动将GPT格式的提示转换为Qwen格式
3. **重试机制**：内置重试和错误处理
4. **统一接口**：与GPT API使用相同的接口
5. **灵活配置**：支持不同的Qwen模型版本

## 注意事项

1. 确保API密钥正确设置
2. Qwen模型对图像序列的处理方式与GPT略有不同
3. 建议使用`qwen-vl-max`获得最佳性能
4. 注意API调用频率限制

## 错误处理

框架会自动处理以下错误：
- API连接失败
- 模型不支持
- 响应格式错误
- 网络超时

所有错误都会记录在日志中，并返回相应的错误信息。

## 示例输出

```
正在初始化多模态模型物理属性分析器...
已加载配置: 6 个测试版本
✓ 分析器初始化成功

================================================================================
多模态模型物理属性分析 - 测试计划
================================================================================
模型版本: qwen-vl-max
绝对值测试样本数: 50
相对比较测试成对数: 50
视频帧尺寸: 640x480
输出目录: qwen_max_physics_analysis_results
```

## 性能对比

可以通过运行相同的测试来比较GPT和Qwen模型的性能：

```bash
# 运行GPT测试
python run_gpt_physics_tests.py --model-version gpt-4o

# 运行Qwen测试
python run_gpt_physics_tests.py --model-version qwen-vl-max
```

结果会保存在不同的目录中，便于比较分析。 