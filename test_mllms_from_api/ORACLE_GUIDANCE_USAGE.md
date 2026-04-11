# 解题步骤指导 (Oracle Guidance) 功能使用说明

## 概述

解题步骤指导功能通过在提示词中加入具体的分析方法和视觉线索，告诉模型如何分析每种物理属性，提供专业的解题思路和步骤。

## 功能特点

### 传统方式 vs Oracle指导方式

**传统方式**：
- 只提供任务描述和输出格式要求
- 模型需要自行判断分析方法
- 依赖模型的物理知识储备

**Oracle指导方式**：
- 提供具体的分析方法和步骤
- 明确指出关键的视觉线索
- 引导模型使用正确的分析思路

## 各物理属性的解题步骤

### 摩擦力 (Friction)

#### 视觉线索
滑动物体的加速度，即摩擦力导致物体速度减少的快慢程度，可以通过物体随时间的位置变化推断。

#### 分析步骤
1. **识别物体角点**：识别物体顶面的4个角点
2. **单应性变换**：估计单应性变换将物体上平面投影到鸟瞰图
3. **位置跟踪**：在变换后的坐标系中跟踪物体随时间的位置
4. **拟合抛物线**：对位置-时间数据拟合抛物函数 ax² + bx + c
5. **提取加速度**：提取加速度系数'a'，它反映摩擦力水平
6. **判断摩擦力**：更高的摩擦力 = 更大的减速度（更负的'a'）

#### 关键词
- homography transformation
- acceleration
- parabola fitting
- ax² + bx + c

### 粘度 (Viscosity)

#### 视觉线索
液体在地面扩散的速度。可以计算相机平面中的面积大小，然后用液体接触地面时的第一个面积大小对面积大小进行归一化；归一化面积大小序列的增长速度反映粘度。

#### 分析步骤
1. **识别接触点**：识别液体首次接触地面的时刻（初始接触帧）
2. **测量面积**：测量后续每帧中液体的面积大小
3. **归一化处理**：用初始接触面积对所有面积大小进行归一化
4. **计算增长率**：计算归一化面积序列随时间的增长率
5. **判断粘度**：高粘度 = 面积扩展较慢，低粘度 = 面积扩展较快

#### 关键词
- liquid expansion
- area calculation
- normalization
- growth speed

### 弹性 (Elasticity)

#### 视觉线索
球的下落高度和弹跳高度。只需要计算相机坐标中的高度差比值。从轨迹中需要识别关键点：初始位置、首次地面接触和弹跳峰值。

#### 分析步骤
1. **识别初始位置**：识别球的初始下落位置（撞击前的最高点）
2. **定位接触点**：定位首次地面接触点（撞击期间的最低点）
3. **找到弹跳峰值**：找到弹跳峰值（首次弹跳后的最高点）
4. **计算高度比**：计算高度比值：(弹跳高度 - 地面高度) / (初始高度 - 地面高度)
5. **确定弹性系数**：该比值代表恢复系数（弹性）

#### 关键词
- dropping height
- bouncing height  
- trajectory analysis
- initial position

## 提示词结构

### 基本结构
```
任务描述
↓
ANALYSIS METHOD:
[具体的分析方法说明]

Follow these steps:
1. [步骤1]
2. [步骤2]
3. [步骤3]
...

↓
参考值范围
↓
分析指导
↓
输出格式要求
```

### 示例：摩擦力绝对值测试
```
You are analyzing frames from a video showing object motion with friction. 
Your task is to estimate the coefficient of friction.

IMPORTANT: You CAN analyze these images. You MUST provide a specific friction coefficient estimate.

ANALYSIS METHOD:
The intrinsic visual cue to reflect the friction property is the acceleration of the sliding object, i.e., how fast is the velocity of the object reduced because of friction and can be inferred by the position of the object over time. You can estimate a homography transformation from the 4 corner points of the top surface of the object. You then use the homography transformation to project the object upper plane to bird eye view, and estimate the acceleration a via fitting a parabola ax² + bx + c to the position of the object over time.

Follow these steps:
1. Identify the 4 corner points of the object's top surface in each frame
2. Estimate homography transformation to convert to bird's eye view
3. Track the object's position over time in the transformed coordinate system
4. Fit a parabolic function (ax² + bx + c) to the position-time data
5. Extract the acceleration coefficient 'a' which reflects the friction level
6. Higher friction = larger deceleration (more negative 'a'), lower friction = smaller deceleration

The coefficient of friction is typically between 0 and 1:
- Very smooth surfaces (ice on ice): 0.02-0.03
- Steel on steel: 0.4-0.6  
- Rubber on concrete: 0.6-0.8
- Wood on wood: 0.3-0.5

analyze the object's deceleration, sliding behavior, and surface interaction.

YOUR RESPONSE MUST END WITH A SPECIFIC FRICTION COEFFICIENT VALUE IN THIS FORMAT:
"Estimated friction coefficient value: X"

Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
```

## 使用方法

### 命令行参数
```bash
# 基本用法
python run_gpt_physics_tests.py --follow-oracle-test

# 结合属性选择
python run_gpt_physics_tests.py --follow-oracle-test --properties friction

# 结合其他功能
python run_gpt_physics_tests.py --follow-oracle-test --concat-mode
python run_gpt_physics_tests.py --follow-oracle-test --frame-index
python run_gpt_physics_tests.py --follow-oracle-test --icl-example

# 全功能组合
python run_gpt_physics_tests.py \
    --follow-oracle-test \
    --icl-example \
    --concat-mode \
    --frame-index \
    --properties elasticity
```

### 程序化调用
```python
from gpt_physics_analyzer import GPTPhysicsAnalyzer

analyzer = GPTPhysicsAnalyzer()

# 启用Oracle指导的绝对值测试
result = analyzer.analyze_single_video_with_gpt(
    video_path="video.mp4",
    property_type="friction",
    follow_oracle_test=True  # 启用解题步骤指导
)

# 启用Oracle指导的相对比较测试
result = analyzer.compare_two_videos_with_gpt(
    video_path1="video1.mp4",
    video_path2="video2.mp4",
    property_type="friction",
    follow_oracle_test=True
)

# 启用Oracle指导的批量测试
results = analyzer.run_all_tests(
    output_dir="results",
    follow_oracle_test=True  # 启用解题步骤指导
)
```

## 与其他功能的结合

### Oracle + ICL（强力组合）
```bash
python run_gpt_physics_tests.py --follow-oracle-test --icl-example
```

**效果**：
- Oracle提供分析方法和步骤
- ICL提供具体的输入输出示例
- 双重引导确保模型理解任务和格式

### Oracle + 拼接模式
```bash
python run_gpt_physics_tests.py --follow-oracle-test --concat-mode
```

**效果**：
- 在拼接的视频序列中应用Oracle指导
- 帮助模型更好地分析连续的视频内容

### Oracle + 帧索引
```bash
python run_gpt_physics_tests.py --follow-oracle-test --frame-index
```

**效果**：
- 结合明确的帧标识和分析步骤
- 便于模型按步骤分析特定帧

## 优势和应用场景

### 优势
1. **专业指导**：提供物理学专业的分析方法
2. **步骤明确**：详细的分析步骤减少模型困惑
3. **视觉线索**：明确指出关键的视觉特征
4. **准确性提升**：引导模型使用正确的分析思路

### 适用场景
1. **复杂物理分析**：需要专业分析方法的任务
2. **新模型测试**：对物理知识不足的模型
3. **精确度要求高**：需要准确分析的场景
4. **教学演示**：展示正确的分析方法

## 对比测试建议

建议对比Oracle指导和传统模式的效果：

```bash
# 传统模式
python run_gpt_physics_tests.py \
    --properties friction \
    --output-dir traditional_results

# Oracle指导模式
python run_gpt_physics_tests.py \
    --properties friction \
    --follow-oracle-test \
    --output-dir oracle_results

# Oracle + ICL组合模式
python run_gpt_physics_tests.py \
    --properties friction \
    --follow-oracle-test \
    --icl-example \
    --output-dir oracle_icl_results
```

## 实际使用示例

### 开发测试
```bash
# 快速验证Oracle功能
python run_gpt_physics_tests.py \
    --properties elasticity \
    --absolute-samples 3 \
    --relative-pairs 2 \
    --follow-oracle-test \
    --dry-run
```

### 生产测试
```bash
# 大规模Oracle测试
python run_gpt_physics_tests.py \
    --absolute-samples 50 \
    --relative-pairs 30 \
    --follow-oracle-test \
    --icl-example \
    --auto-confirm
```

### 性能对比
```bash
# 对比不同模式的性能
python run_gpt_physics_tests.py --properties friction --output-dir baseline_results
python run_gpt_physics_tests.py --properties friction --icl-example --output-dir icl_results  
python run_gpt_physics_tests.py --properties friction --follow-oracle-test --output-dir oracle_results
python run_gpt_physics_tests.py --properties friction --follow-oracle-test --icl-example --output-dir oracle_icl_results
```

## 注意事项

1. **方法适用性**：Oracle指导基于特定的物理分析方法，适用于相应的视频内容
2. **模型理解能力**：不同模型对复杂指导的理解能力可能不同
3. **Token增加**：Oracle指导会显著增加提示词长度
4. **领域专业性**：指导内容基于物理学原理，需要模型有一定的理解能力

Oracle指导功能通过提供专业的分析方法和详细的步骤，帮助模型更准确地分析物理属性，是提升分析质量的重要工具。 