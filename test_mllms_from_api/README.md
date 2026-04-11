# GPT Physics Analysis Tool

这是一个综合的GPT物理属性分析工具，用于测试GPT在三个物理任务上的性能：friction（摩擦），viscosity（粘度），elasticity（弹性）。

## 功能特点

- 支持两种分析模式：
  - **Absolute（绝对值预测）**: 分析单个视频，预测具体的物理属性值
  - **Relative（相对比较）**: 比较两个视频，判断哪个的物理属性值更大

- 支持三种物理属性：
  - **Friction（摩擦系数）**: 0-1之间的值
  - **Viscosity（粘度）**: 以厘泊(cP)为单位
  - **Elasticity（弹性系数）**: 恢复系数，0-1之间的值

## 文件结构

```
test_mllms_from_api/
├── gpt_physics_analyzer.py      # 主分析器代码
├── gpt_api_third_party.py       # 第三方GPT API调用接口
├── vjepa_data_dirs.json         # 测试数据路径配置
├── run_gpt_physics_tests.py     # 运行脚本
└── README.md                    # 使用说明
```

## 配置要求

### 1. 环境依赖
```bash
pip install opencv-python numpy
```

### 2. API 配置
- 复制仓库根目录 `secrets/gpt_key.json.example` 为 `secrets/gpt_key.json`，或设置环境变量 **`GPT_KEY_JSON`** 指向密钥 JSON。
- `gpt_api_third_party.py` 默认读取 **`$SVD_ROOT/secrets/gpt_key.json`**（若未设置 `GPT_KEY_JSON`）。
- 遗留脚本 `gpt_zhan.py` 需设置 **`GPT_PROXY_KEY`** / **`GPT_PROXY_URL`**，请勿在代码中硬编码密钥。

### 3. 数据路径
`vjepa_data_dirs.json` 为占位模板，请替换为本地或 HF 解压后的数据根路径：
- **v119_vjepa**: viscosity absolute
- **v110_vjepa**: viscosity relative  
- **v120_vjepa**: friction absolute
- **v121_vjepa**: friction relative
- **v124_vjepa**: elasticity absolute
- **v125_vjepa**: elasticity relative

每个版本都有3个测试集：test1, test2, test3

## 使用方法

### 快速开始
```bash
cd test_mllms_from_api
python run_gpt_physics_tests.py
```

### 高级用法
```python
from gpt_physics_analyzer import GPTPhysicsAnalyzer

# 初始化分析器
analyzer = GPTPhysicsAnalyzer("vjepa_data_dirs.json")

# 运行所有测试
results = analyzer.run_all_tests(
    output_dir="results", 
    frame_size=(640, 480)
)
```

### 运行单个测试
```python
# 分析单个视频的物理属性
result = analyzer.analyze_single_video_with_gpt(
    video_path="/path/to/video.mp4",
    property_type="friction"  # 或 "viscosity", "elasticity"
)

# 比较两个视频的物理属性
result = analyzer.compare_two_videos_with_gpt(
    video_path1="/path/to/video1.mp4",
    video_path2="/path/to/video2.mp4", 
    property_type="viscosity"
)
```

## 测试配置

工具会自动从每个数据目录中随机采样最多50个视频进行测试：

- **Absolute测试**: 对每个视频预测物理属性值
- **Relative测试**: 随机配对视频进行比较（最多25对）

## 结果输出

测试结果会保存在 `results.txt` 格式的文件中，包含：

- 测试版本信息
- 物理属性类型 
- 测试类型（absolute/relative）
- 总样本数和成功/失败分析数
- 详细的个体结果：
  - Absolute: `predicted_value` vs `gt_value`
  - Relative: `predicted_comparison` vs `gt_comparison`

## 数据集映射

不同版本使用不同的数据集类和GT文件：

| 版本 | 属性 | 类型 | 数据集类 | GT文件 |
|------|------|------|----------|--------|
| v119 | viscosity | absolute | WebVidv2Viscosity | bounce_analysis_results.json |
| v110 | viscosity | relative | WebVidv3ViscosityTest | 从文件名解析 |
| v120 | friction | absolute | WebVidv2Viscosity | bounce_analysis_results.json |
| v121 | friction | relative | WebVidv3FrictionTest | 从文件名解析 |
| v124 | elasticity | absolute | WebVidv2Viscosity | bounce_analysis_results.json |
| v125 | elasticity | relative | WebVidv3ElasticityTest | gt_restitution1.json |

## 故障排除

### 常见问题

1. **API调用失败**
   - 检查网络连接
   - 确认API密钥文件路径和格式
   - 查看是否有API限制

2. **视频读取失败**
   - 确认视频文件路径正确
   - 检查视频文件是否损坏
   - 确认OpenCV能正确处理视频格式

3. **数据目录不存在**
   - 检查 `vjepa_data_dirs.json` 中的路径
   - 确认数据已正确挂载/下载

### 调试模式

在代码中设置调试标志：
```python
analyzer = GPTPhysicsAnalyzer()
# 手动运行单个测试进行调试
result = analyzer.run_absolute_test(...)
```

## 性能优化

- 使用较小的帧尺寸 (640x480) 以节省带宽
- 设置合适的API调用间隔以避免速率限制
- 可以调整采样的视频数量和帧数

## 扩展开发

要添加新的物理属性或分析类型：

1. 在 `properties` 列表中添加新属性
2. 在 `get_analysis_prompt()` 中添加相应的提示词
3. 在 `extract_predicted_value()` 中添加结果解析逻辑
4. 更新数据集映射关系

## 许可证

本工具仅用于研究目的。 