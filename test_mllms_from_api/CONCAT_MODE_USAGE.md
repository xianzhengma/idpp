# 拼接模式使用说明

## 概述

拼接模式是一种新的相对比较方式，它将两个视频的帧拼接在一起，并使用3个黑帧作为分隔符，然后通过语言告诉模型黑帧前面是video1，黑帧后面是video2。

## 功能特点

### 传统模式 vs 拼接模式

**传统模式**：
- 使用文本标识："VIDEO 1 frames:" 和 "VIDEO 2 frames:"
- 两个视频的帧分别标识和展示
- 依赖文本理解来区分两个视频

**拼接模式**：
- 将两个视频的帧连续排列：video1帧 → 3个黑帧 → video2帧
- 通过视觉分隔符（黑帧）来区分两个视频
- 更直观的时序展示方式

### Content格式对比

#### 传统模式content格式：
```
[
    {"type": "text", "text": "提示词..."},
    {"type": "text", "text": "VIDEO 1 frames:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame2"}},
    ...
    {"type": "text", "text": "VIDEO 2 frames:"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,frame2"}},
    ...
]
```

#### 拼接模式content格式：
```
[
    {"type": "text", "text": "拼接模式提示词..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,video1_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,video1_frame2"}},
    ...
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame2"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,black_frame3"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,video2_frame1"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,video2_frame2"}},
    ...
]
```

## 使用方法

### 1. 命令行参数

在运行`run_gpt_physics_tests.py`时添加`--concat-mode`参数：

```bash
# 启用拼接模式
python run_gpt_physics_tests.py --concat-mode

# 只测试特定属性
python run_gpt_physics_tests.py --properties friction viscosity

# 完整示例
python run_gpt_physics_tests.py \
    --model-version gpt-4o \
    --absolute-samples 20 \
    --relative-pairs 20 \
    --concat-mode \
    --properties friction \
    --output-dir gpt_concat_mode_results
```

### 2. 程序化调用

```python
from gpt_physics_analyzer import GPTPhysicsAnalyzer

# 初始化分析器
analyzer = GPTPhysicsAnalyzer()

# 使用拼接模式比较两个视频
result = analyzer.compare_two_videos_with_gpt(
    video_path1="path/to/video1.mp4",
    video_path2="path/to/video2.mp4", 
    property_type="friction",
    max_frames_per_video=16,
    frame_size=(640, 480),
    model_version="gpt-4o",
    concat_mode=True  # 启用拼接模式
)

# 运行所有测试（仅相对比较测试会使用拼接模式）
results = analyzer.run_all_tests(
    output_dir="results",
    frame_size=(640, 480),
    max_samples=50,
    model_version="gpt-4o",
    concat_mode=True
)
```

## 提示词差异

### 拼接模式专用提示词

拼接模式使用专门的提示词来指导模型理解帧序列：

```
You are comparing two {property_type} videos to determine which has a higher {property_type} value.

IMPORTANT: You CAN analyze these images. You MUST provide a comparison result.

The frames are arranged in sequence: VIDEO 1 frames first, followed by 3 BLACK SEPARATOR frames, then VIDEO 2 frames.

Compare the {property_type} characteristics between the two videos:
- The frames BEFORE the black separator frames belong to VIDEO 1
- The frames AFTER the black separator frames belong to VIDEO 2  
- The black frames are just separators and should be ignored for analysis

Analyze the motion, physics, and {property_type} characteristics in both video segments.

YOUR RESPONSE MUST END WITH A SPECIFIC COMPARISON RESULT IN THIS FORMAT:
"Comparison result: 1" (if VIDEO 1 has higher {property_type})
"Comparison result: 0" (if VIDEO 2 has higher {property_type})

Do not say you cannot analyze the images. You must provide your best comparison based on the visual information.
```

## 技术实现

### 黑帧生成

```python
def generate_black_frames(self, num_frames: int = 3, frame_size: Tuple[int, int] = (640, 480)) -> List[str]:
    """
    生成指定数量的黑帧（base64编码）
    """
    black_frames = []
    
    # 创建黑色图像
    black_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    for _ in range(num_frames):
        # 将黑色图像编码为JPEG格式
        _, buffer = cv2.imencode(".jpg", black_image)
        black_frame_b64 = base64.b64encode(buffer).decode("utf-8")
        black_frames.append(black_frame_b64)
    
    return black_frames
```

### 帧序列组织

拼接模式按以下顺序组织帧：
1. VIDEO 1的所有帧（最多16帧）
2. 3个黑色分隔帧
3. VIDEO 2的所有帧（最多16帧）

## 优势和局限

### 优势
- 更直观的时序展示
- 减少对文本理解的依赖
- 提供清晰的视觉分隔
- 可能提高模型对视频序列的理解

### 局限
- 增加了总帧数（+3帧）
- 黑帧可能被误解为视频内容
- 需要模型理解分隔符概念
- 仅适用于相对比较测试

## 注意事项

1. **仅用于相对比较**：拼接模式只影响相对比较测试，绝对值测试不受影响
2. **帧数限制**：总帧数 = video1帧数 + 3个黑帧 + video2帧数
3. **模型兼容性**：确保使用的模型支持处理较长的图像序列
4. **结果格式**：输出格式与传统模式保持一致

## 测试建议

建议同时运行传统模式和拼接模式，比较两种方式的效果：

```bash
# 传统模式
python run_gpt_physics_tests.py --output-dir traditional_results

# 拼接模式  
python run_gpt_physics_tests.py --concat-mode --output-dir concat_results

# 只测试特定属性
python run_gpt_physics_tests.py --properties friction --output-dir friction_only_results

# 综合使用
python run_gpt_physics_tests.py \
    --concat-mode \
    --properties friction viscosity \
    --relative-pairs 20 \
    --output-dir comprehensive_results
```

然后对比两种模式的准确率和一致性。

## 属性过滤功能

### 新增参数

- `--properties`: 指定要测试的物理属性，可以是以下一个或多个：
  - `friction`: 摩擦力测试
  - `viscosity`: 粘度测试  
  - `elasticity`: 弹性测试

### 使用示例

```bash
# 只测试摩擦力
python run_gpt_physics_tests.py --properties friction

# 测试摩擦力和弹性
python run_gpt_physics_tests.py --properties friction elasticity

# 测试所有属性（默认行为）
python run_gpt_physics_tests.py

# 结合其他参数使用
python run_gpt_physics_tests.py \
    --properties viscosity \
    --concat-mode \
    --relative-pairs 10 \
    --model-version gpt-4o
``` 