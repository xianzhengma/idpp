# In-Context Learning (ICL) 功能使用说明

## 概述

In-Context Learning (ICL) 功能从训练集中采样3个示例作为输入输出对，为模型提供参考，帮助模型更好地理解任务格式和预期输出。

## 功能特点

### 传统方式 vs ICL方式

**传统方式**：
- 只提供任务描述和提示词
- 模型需要自行理解输出格式
- 依赖模型的先验知识

**ICL方式**：
- 提供3个具体的输入输出示例
- 明确展示期望的输出格式
- 通过示例引导模型行为

## ICL示例采样策略

### 采样规则
1. **数量**：每个测试固定采样3个示例
2. **来源**：从对应的训练集中采样
3. **方式**：确定性均匀采样（按文件名排序后均匀分布选择）
4. **帧数**：ICL示例使用默认16帧（与主测试保持一致），确保示例和测试的一致性

### 示例格式
- **绝对值测试**：`输入视频帧` → `输出数值`
- **相对比较测试**：`输入视频帧` → `输出比较结果`

## Content组织结构

### 绝对值测试ICL格式
```
[
    {"type": "text", "text": "任务提示词（包含ICL引导）"},
    
    // ICL示例1
    {"type": "text", "text": "Example 1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,example1_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,example1_frame2"}},
    ...
    {"type": "text", "text": "Output: Estimated friction coefficient value: 0.456"},
    {"type": "text", "text": ""},  // 空行分隔
    
    // ICL示例2
    {"type": "text", "text": "Example 2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,example2_frame1"}},
    ...
    {"type": "text", "text": "Output: Estimated friction coefficient value: 0.789"},
    {"type": "text", "text": ""},
    
    // ICL示例3
    {"type": "text", "text": "Example 3:"},
    ...
    {"type": "text", "text": "Output: Estimated friction coefficient value: 0.234"},
    {"type": "text", "text": ""},
    
    // 当前任务
    {"type": "text", "text": "Now analyze the following video:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test_frame2"}},
    ...
]
```

### ICL + 帧索引格式
```
[
    {"type": "text", "text": "任务提示词"},
    
    // ICL示例1（带帧索引）
    {"type": "text", "text": "Example 1:"},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,example1_frame1"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,example1_frame2"}},
    ...
    {"type": "text", "text": "Output: Estimated friction coefficient value: 0.456"},
    {"type": "text", "text": ""},
    
    // 当前任务（带帧索引）
    {"type": "text", "text": "Now analyze the following video:"},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test_frame1"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test_frame2"}},
    ...
]
```

## 输出格式示例

### 不同属性的ICL输出格式

#### 摩擦力 (Friction)
```
Output: Estimated friction coefficient value: 0.456
```

#### 粘度 (Viscosity)
```
Output: Estimated viscosity value: 125.0 cP
```

#### 弹性 (Elasticity)
```
Output: Estimated restitution coefficient value: 0.789
```

#### 相对比较 (Relative)
```
Output: Comparison result: 1
```

## 使用方法

### 命令行参数
```bash
# 基本用法
python run_gpt_physics_tests.py --icl-example

# 结合其他功能
python run_gpt_physics_tests.py --icl-example --properties friction
python run_gpt_physics_tests.py --icl-example --concat-mode
python run_gpt_physics_tests.py --icl-example --frame-index
python run_gpt_physics_tests.py --icl-example --concat-mode --frame-index --properties viscosity
```

### 程序化调用
```python
from gpt_physics_analyzer import GPTPhysicsAnalyzer

analyzer = GPTPhysicsAnalyzer()

# 启用ICL的绝对值测试
result = analyzer.analyze_single_video_with_gpt(
    video_path="video.mp4",
    property_type="friction",
    icl_examples=icl_examples  # ICL示例列表
)

# 启用ICL的相对比较测试
result = analyzer.compare_two_videos_with_gpt(
    video_path1="video1.mp4",
    video_path2="video2.mp4",
    property_type="friction",
    icl_examples=icl_examples
)

# 启用ICL的批量测试
results = analyzer.run_all_tests(
    output_dir="results",
    icl_example=True  # 启用ICL
)
```

## ICL示例采样实现

### 核心方法
```python
def sample_icl_examples(self, train_dir: str, property_type: str, analysis_type: str, 
                       dataset_class: str, num_examples: int = 3, 
                       frame_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """
    从训练集中采样ICL示例
    
    返回格式:
    [
        {
            'video_file': 'train_video1.mp4',
            'frames': [base64_frame1, base64_frame2, ...],
            'gt_value': 0.456,
            'output': 'Estimated friction coefficient value: 0.456'
        },
        ...
    ]
    """
```

## 提示词适配

### ICL引导语
ICL模式下，提示词会自动添加引导语：
- 传统模式：`"Analyze the object's deceleration..."`
- ICL模式：`"Based on the examples above, analyze the object's deceleration..."`

这种适配帮助模型更好地利用提供的示例。

## 优势和应用场景

### 优势
1. **格式引导**：明确展示期望的输出格式
2. **任务理解**：通过示例帮助模型理解任务
3. **一致性提升**：减少输出格式的变异性
4. **性能改进**：Few-shot learning提升预测准确性

### 适用场景
1. **新任务**：模型对特定物理属性分析不熟悉时
2. **格式要求严格**：需要精确的输出格式时
3. **性能优化**：希望提升模型表现时
4. **一致性要求**：需要稳定的输出格式时

## 注意事项

1. **训练集依赖**：需要对应的训练集目录存在
2. **示例质量**：ICL示例的质量影响模型表现
3. **Token增加**：ICL示例会增加输入token数量
4. **采样策略**：确定性采样保证结果可复现

## 对比测试建议

建议对比ICL和传统模式的效果：

```bash
# 传统模式
python run_gpt_physics_tests.py \
    --properties friction \
    --output-dir traditional_results

# ICL模式
python run_gpt_physics_tests.py \
    --properties friction \
    --icl-example \
    --output-dir icl_results

# 全功能ICL模式
python run_gpt_physics_tests.py \
    --properties friction \
    --icl-example \
    --concat-mode \
    --frame-index \
    --output-dir full_icl_results
```

## 实际使用示例

### 开发测试
```bash
# 快速验证ICL功能
python run_gpt_physics_tests.py \
    --properties friction \
    --absolute-samples 3 \
    --relative-pairs 2 \
    --icl-example \
    --dry-run
```

### 生产测试
```bash
# 大规模ICL测试
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 30 \
    --icl-example \
    --auto-confirm
```

ICL功能通过提供具体示例，帮助模型更好地理解任务要求和输出格式，是提升模型性能的重要工具。 