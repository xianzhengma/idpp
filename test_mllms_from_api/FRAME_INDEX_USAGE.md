# 帧索引功能使用说明

## 概述

帧索引功能为每个视频帧添加显式的索引标识，格式为：`frame1:`, `frame2:`, `frame3:`, ...，帮助模型更好地理解视频的时序信息。

## 功能特点

### 传统方式 vs 帧索引方式

**传统方式**：
- 直接发送图像，没有明确的帧序号
- 依赖模型自行推断帧的顺序关系
- content格式：`[提示词, 图像1, 图像2, 图像3, ...]`

**帧索引方式**：
- 每个图像前添加明确的帧索引标识
- 清晰指示每帧的时序位置
- content格式：`[提示词, "frame1:", 图像1, "frame2:", 图像2, "frame3:", 图像3, ...]`

## Content格式详解

### 绝对值测试

#### 传统格式：
```
[
    {"type": "text", "text": "分析提示词..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame1_data"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame2_data"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame3_data"}},
    ...
]
```

#### 帧索引格式：
```
[
    {"type": "text", "text": "分析提示词..."},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame1_data"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame2_data"}},
    {"type": "text", "text": "frame3:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame3_data"}},
    ...
]
```

### 相对比较测试（传统模式）

#### 传统格式：
```
[
    {"type": "text", "text": "比较提示词..."},
    {"type": "text", "text": "VIDEO 1 frames:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame2"}},
    {"type": "text", "text": "VIDEO 2 frames:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame2"}},
    ...
]
```

#### 帧索引格式：
```
[
    {"type": "text", "text": "比较提示词..."},
    {"type": "text", "text": "VIDEO 1 frames:"},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame1"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame2"}},
    {"type": "text", "text": "VIDEO 2 frames:"},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame1"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame2"}},
    ...
]
```

### 相对比较测试（拼接模式）

#### 传统拼接格式：
```
[
    {"type": "text", "text": "拼接比较提示词..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame2"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame2"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame3"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame2"}},
    ...
]
```

#### 拼接+帧索引格式：
```
[
    {"type": "text", "text": "拼接比较提示词..."},
    {"type": "text", "text": "frame1:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame1"}},
    {"type": "text", "text": "frame2:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v1_frame2"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame1"}},  // 黑帧不加索引
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame2"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame3"}},
    {"type": "text", "text": "frame3:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame1"}},
    {"type": "text", "text": "frame4:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,v2_frame2"}},
    ...
]
```

## 使用方法

### 命令行参数

```bash
# 基本用法
python run_gpt_physics_tests.py --frame-index

# 结合其他功能
python run_gpt_physics_tests.py --frame-index --concat-mode
python run_gpt_physics_tests.py --frame-index --properties friction
python run_gpt_physics_tests.py --frame-index --concat-mode --properties viscosity
```

### 程序化调用

```python
from gpt_physics_analyzer import GPTPhysicsAnalyzer

analyzer = GPTPhysicsAnalyzer()

# 绝对值测试（启用帧索引）
result = analyzer.analyze_single_video_with_gpt(
    video_path="video.mp4",
    property_type="friction",
    frame_index=True  # 启用帧索引
)

# 相对比较测试（启用帧索引）
result = analyzer.compare_two_videos_with_gpt(
    video_path1="video1.mp4",
    video_path2="video2.mp4",
    property_type="friction",
    frame_index=True  # 启用帧索引
)

# 批量测试（启用帧索引）
results = analyzer.run_all_tests(
    output_dir="results",
    frame_index=True  # 启用帧索引
)
```

## 帧索引规则

### 绝对值测试
- 每个视频的帧从 `frame1:` 开始编号
- 按时序递增：`frame1:`, `frame2:`, `frame3:`, ...

### 相对比较测试（传统模式）
- 每个视频内部独立编号
- VIDEO 1: `frame1:`, `frame2:`, `frame3:`, ...
- VIDEO 2: `frame1:`, `frame2:`, `frame3:`, ...

### 相对比较测试（拼接模式）
- 全局连续编号（跨越两个视频）
- VIDEO 1: `frame1:`, `frame2:`, `frame3:`
- 黑帧分隔符：无索引标识
- VIDEO 2: `frame4:`, `frame5:`, `frame6:`

## 优势和应用场景

### 优势
1. **时序清晰**：明确指示每帧的时间顺序
2. **便于引用**：模型可以明确引用特定帧（如"在frame3中..."）
3. **减少歧义**：避免模型对帧顺序的误解
4. **提升理解**：帮助模型更好地理解视频的时间维度

### 适用场景
1. **运动分析**：需要精确跟踪物体运动轨迹
2. **时序比较**：比较不同时刻的物理状态
3. **过程分析**：分析物理过程的阶段性变化
4. **调试诊断**：便于定位模型关注的特定帧

## 注意事项

1. **帧数限制**：帧索引不会改变最大帧数限制
2. **内容增加**：每帧增加一个文本元素，总content元素数量增加
3. **兼容性**：与拼接模式、属性过滤等功能完全兼容
4. **性能影响**：轻微增加token数量，但提升理解准确性

## 对比测试建议

建议同时运行传统模式和帧索引模式，比较效果：

```bash
# 传统模式
python run_gpt_physics_tests.py \
    --properties friction \
    --output-dir traditional_results

# 帧索引模式
python run_gpt_physics_tests.py \
    --properties friction \
    --frame-index \
    --output-dir frame_index_results

# 全功能模式
python run_gpt_physics_tests.py \
    --properties friction \
    --concat-mode \
    --frame-index \
    --output-dir full_feature_results
```

## 实际使用示例

### 开发测试
```bash
# 快速验证帧索引功能
python run_gpt_physics_tests.py \
    --properties friction \
    --absolute-samples 3 \
    --relative-pairs 2 \
    --frame-index \
    --dry-run
```

### 生产测试
```bash
# 大规模帧索引测试
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 30 \
    --frame-index \
    --auto-confirm
```

帧索引功能为视频分析提供了更精确的时序信息，有助于提升模型对物理过程的理解能力。 