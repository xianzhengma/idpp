# 功能改进说明

## 概述

本文档介绍了两个重要的功能改进：
1. **自动输出目录后缀**：根据启用的功能自动为输出目录添加描述性后缀
2. **完整GPT回复保存**：保存模型的完整回复内容，不再截断

## 1. 自动输出目录后缀功能

### 功能描述

系统会根据启用的功能自动为输出目录添加相应的后缀，使目录名称更具描述性，便于区分不同配置的测试结果。

### 后缀规则

| 功能参数 | 后缀 | 说明 |
|---------|------|------|
| `--concat-mode` | `concat` | 拼接模式 |
| `--frame-index` | `frameindex` | 帧索引模式 |
| `--icl-example` | `icl` | In-Context Learning |
| `--follow-oracle-test` | `oracle` | 解题步骤指导 |

### 后缀组合规则

- 多个功能启用时，后缀按照固定顺序组合：`concat_frameindex_icl_oracle`
- 只启用部分功能时，只添加对应的后缀
- 后缀之间用下划线 `_` 连接

### 示例

#### 基础目录名（无功能启用）
```bash
python run_gpt_physics_tests.py
# 输出目录: gpt_physics_analysis_results

python run_gpt_physics_tests.py --model-version gemini-1.5-pro
# 输出目录: gemini_physics_analysis_results
```

#### 单个功能启用
```bash
python run_gpt_physics_tests.py --concat-mode
# 输出目录: gpt_physics_analysis_results_concat

python run_gpt_physics_tests.py --icl-example
# 输出目录: gpt_physics_analysis_results_icl

python run_gpt_physics_tests.py --follow-oracle-test
# 输出目录: gpt_physics_analysis_results_oracle
```

#### 多个功能组合
```bash
python run_gpt_physics_tests.py --concat-mode --frame-index
# 输出目录: gpt_physics_analysis_results_concat_frameindex

python run_gpt_physics_tests.py --icl-example --follow-oracle-test
# 输出目录: gpt_physics_analysis_results_icl_oracle

python run_gpt_physics_tests.py --concat-mode --frame-index --icl-example --follow-oracle-test
# 输出目录: gpt_physics_analysis_results_concat_frameindex_icl_oracle
```

#### 不同模型 + 功能组合
```bash
python run_gpt_physics_tests.py --model-version gemini-1.5-pro --icl-example --oracle-test
# 输出目录: gemini_physics_analysis_results_icl_oracle

python run_gpt_physics_tests.py --model-version qwen-vl-max --concat-mode --frame-index
# 输出目录: qwen_max_physics_analysis_results_concat_frameindex
```

### 用户指定目录的处理

当用户明确指定输出目录时，系统不会添加自动后缀，完全保持用户的选择：

```bash
python run_gpt_physics_tests.py --concat-mode --icl-example --output-dir my_test_results
# 输出目录: my_test_results（保持用户指定的名称）
```

### 目录结构示例

使用自动后缀功能后，目录结构变得更加清晰：

```
project/
├── gpt_physics_analysis_results/                    # 基础测试
├── gpt_physics_analysis_results_concat/             # 拼接模式测试
├── gpt_physics_analysis_results_icl/                # ICL测试
├── gpt_physics_analysis_results_oracle/             # Oracle指导测试
├── gpt_physics_analysis_results_concat_icl/         # 拼接+ICL测试
├── gpt_physics_analysis_results_icl_oracle/         # ICL+Oracle测试
├── gpt_physics_analysis_results_concat_frameindex_icl_oracle/  # 全功能测试
├── gemini_physics_analysis_results_icl/             # Gemini ICL测试
└── my_custom_results/                               # 用户指定目录
```

## 2. 完整GPT回复保存功能

### 功能描述

之前的版本为了节省空间，将GPT回复截断为200字符并添加"..."。现在保存完整的GPT回复内容，便于详细分析模型的推理过程。

### 格式改进

#### 之前的格式（截断）
```
GPT回复: Based on the analysis of the provided frames, here is the estimation of the friction coefficient.

**Analysis:**
1.  **Object State:** The yellow cube remains completely stationary throughout all the ...
```

#### 现在的格式（完整）
```
GPT完整回复:
----------------------------------------
Based on the analysis of the provided frames, here is the estimation of the friction coefficient.

**Analysis:**
1.  **Object State:** The yellow cube remains completely stationary throughout all the frames, showing no signs of movement or sliding.
2.  **Surface Interaction:** There appears to be no relative motion between the cube and the surface.
3.  **Physics Interpretation:** When an object remains stationary on a surface, it indicates that the applied forces are insufficient to overcome static friction.

**Detailed Analysis:**
The object shows no signs of sliding or movement throughout the video sequence. This suggests either:
- Very high friction coefficient preventing any movement
- No external force being applied to initiate motion
- The object is in a stable equilibrium state

Given the visual evidence of complete stationarity, I estimate a moderate to high friction coefficient.

Estimated friction coefficient value: 0.4
----------------------------------------
```

### 保存格式特点

1. **明确标识**：使用`GPT完整回复:`标识，区别于其他信息
2. **清晰分隔**：使用40个短横线`----------------------------------------`作为分隔线
3. **完整内容**：保存模型的完整回复，不进行任何截断
4. **格式保持**：保持原有的换行、缩进等格式

### 对比示例

#### 绝对值测试结果
```
--- 样本 1 ---
视频: friction_sim_box_objFriction_0.01_tableFriction_0.01_size_0.10_dir_-0.00_0.00_id_958_rgb.mp4
预测值: 0.4
GT值: 0.0
GPT完整回复:
----------------------------------------
Based on the analysis of the provided frames, here is the estimation of the friction coefficient.

ANALYSIS METHOD:
The intrinsic visual cue to reflect the friction property is the acceleration of the sliding object, i.e., how fast is the velocity of the object reduced because of friction and can be inferred by the position of the object over time. You can estimate a homography transformation from the 4 corner points of the top surface of the object. You then use the homography transformation to project the object upper plane to bird eye view, and estimate the acceleration a via fitting a parabola ax² + bx + c to the position of the object over time.

Follow these steps:
1. Identify the 4 corner points of the object's top surface in each frame
2. Estimate homography transformation to convert to bird's eye view
3. Track the object's position over time in the transformed coordinate system
4. Fit a parabolic function (ax² + bx + c) to the position-time data
5. Extract the acceleration coefficient 'a' which reflects the friction level
6. Higher friction = larger deceleration (more negative 'a'), lower friction = smaller deceleration

**Analysis:**
1. **Object State:** The yellow cube remains completely stationary throughout all the frames, showing no signs of movement or sliding.
2. **Surface Interaction:** There appears to be no relative motion between the cube and the surface.
3. **Physics Interpretation:** When an object remains stationary on a surface, it indicates that the applied forces are insufficient to overcome static friction.

**Detailed Analysis:**
The object shows no signs of sliding or movement throughout the video sequence. This suggests either:
- Very high friction coefficient preventing any movement
- No external force being applied to initiate motion
- The object is in a stable equilibrium state

Given the visual evidence of complete stationarity, I estimate a moderate friction coefficient.

Estimated friction coefficient value: 0.4
----------------------------------------
```

#### 相对比较测试结果
```
--- 比较 1 ---
视频对: video1.mp4 vs video2.mp4
预测结果: 1
GT结果: 1
GT值: 100.0 vs 50.0
GPT完整回复:
----------------------------------------
比较两个液体的粘度特性，我进行了详细的分析：

ANALYSIS METHOD:
The intrinsic visual cue to reflect the viscosity property is the speed that the liquid expands on the ground. You can calculate the area size in the camera plane and then normalise the area sizes by the first area size when the liquid touches ground; the growth speed of the normalised area size sequence reflects the viscosity.

Follow these steps:
1. Identify when the liquid first touches the ground (initial contact frame)
2. Measure the liquid area size in each subsequent frame
3. Normalize all area sizes by dividing by the initial contact area
4. Calculate the growth rate of the normalized area sequence over time
5. Higher viscosity = slower area expansion, lower viscosity = faster area expansion

视频1分析：
- 液体流动速度较慢
- 扩散面积增长缓慢
- 表面波动较小
- 估计粘度较高

视频2分析：
- 液体流动速度较快
- 扩散面积增长迅速
- 表面波动明显
- 估计粘度较低

综合比较：
视频1中的液体明显比视频2中的液体粘度更高，流动性更差。根据观察到的扩散速度差异，我判断视频1具有更高的粘度值。

Comparison result: 1
----------------------------------------
```

### 优势

1. **完整信息**：保留模型的完整推理过程，便于分析
2. **错误诊断**：可以查看模型的完整回复，诊断预测错误的原因
3. **质量评估**：可以评估模型回复的质量和详细程度
4. **研究价值**：完整的回复对研究模型行为更有价值
5. **格式清晰**：使用分隔线和标识，便于阅读和解析

### 存储考虑

- 完整回复会增加结果文件的大小
- 对于大规模测试，建议定期清理旧的结果文件
- 如果存储空间有限，可以考虑压缩存储结果文件

## 使用建议

### 目录管理
1. **功能测试**：利用自动后缀功能，为不同配置的测试创建清晰的目录结构
2. **对比分析**：通过目录名称快速识别测试配置，便于结果对比
3. **版本管理**：结合时间戳和功能后缀，创建有序的测试版本

### 结果分析
1. **完整回复**：利用完整的GPT回复分析模型的推理过程
2. **错误诊断**：查看完整回复找出预测错误的原因
3. **质量评估**：评估不同配置下模型回复的质量差异

### 最佳实践
1. **命名规范**：对于重要的测试，使用`--output-dir`指定有意义的目录名
2. **结果备份**：定期备份重要的测试结果
3. **空间管理**：监控磁盘空间，及时清理不需要的结果文件

这些改进使得工具更加智能和实用，提供了更好的用户体验和分析能力。 