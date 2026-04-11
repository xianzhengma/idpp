# 多模态模型物理属性分析 - 使用示例

## 基本使用

### 1. 测试所有属性（默认）
```bash
python run_gpt_physics_tests.py
```

### 2. 干运行模式（查看测试计划）
```bash
python run_gpt_physics_tests.py --dry-run
```

### 3. 自动确认运行
```bash
python run_gpt_physics_tests.py --auto-confirm
```

## 属性选择

### 4. 测试单个属性
```bash
# 只测试摩擦力
python run_gpt_physics_tests.py --properties friction

# 只测试粘度
python run_gpt_physics_tests.py --properties viscosity

# 只测试弹性
python run_gpt_physics_tests.py --properties elasticity
```

### 5. 测试多个属性
```bash
# 测试摩擦力和弹性
python run_gpt_physics_tests.py --properties friction elasticity

# 测试粘度和摩擦力
python run_gpt_physics_tests.py --properties viscosity friction
```

## 拼接模式

### 6. 启用拼接模式
```bash
# 使用拼接模式进行相对比较
python run_gpt_physics_tests.py --concat-mode
```

### 7. 结合属性选择和拼接模式
```bash
# 只测试摩擦力，使用拼接模式
python run_gpt_physics_tests.py --properties friction --concat-mode
```

## 帧索引模式

### 8. 启用帧索引
```bash
# 为每个视频帧添加索引标识
python run_gpt_physics_tests.py --frame-index
```

### 9. 结合帧索引和其他功能
```bash
# 帧索引 + 拼接模式
python run_gpt_physics_tests.py --frame-index --concat-mode

# 帧索引 + 属性选择
python run_gpt_physics_tests.py --frame-index --properties friction

# 帧索引 + 拼接模式 + 属性选择
python run_gpt_physics_tests.py --frame-index --concat-mode --properties viscosity
```

## In-Context Learning (ICL)

### 10. 启用ICL示例
```bash
# 从训练集采样3个示例作为输入输出对
python run_gpt_physics_tests.py --icl-example
```

### 11. 结合ICL和其他功能
```bash
# ICL + 属性选择
python run_gpt_physics_tests.py --icl-example --properties friction

# ICL + 拼接模式
python run_gpt_physics_tests.py --icl-example --concat-mode

# ICL + 帧索引
python run_gpt_physics_tests.py --icl-example --frame-index

# ICL + 所有功能
python run_gpt_physics_tests.py --icl-example --concat-mode --frame-index --properties viscosity
```

## 解题步骤指导 (Oracle Guidance)

### 12. 启用解题步骤指导
```bash
# 告诉模型具体的分析方法和视觉线索
python run_gpt_physics_tests.py --follow-oracle-test
```

### 13. 结合解题步骤指导和其他功能
```bash
# Oracle + 属性选择
python run_gpt_physics_tests.py --follow-oracle-test --properties elasticity

# Oracle + 拼接模式
python run_gpt_physics_tests.py --follow-oracle-test --concat-mode

# Oracle + ICL（强力组合）
python run_gpt_physics_tests.py --follow-oracle-test --icl-example

# 全功能组合
python run_gpt_physics_tests.py --follow-oracle-test --icl-example --concat-mode --frame-index --properties friction
```

## 样本数量控制

### 14. 调整样本数量
```bash
# 绝对值测试每个测试集10个样本，相对比较测试每个测试集5对视频
python run_gpt_physics_tests.py --absolute-samples 10 --relative-pairs 5
```

### 15. 结合样本数量和属性选择
```bash
# 只测试粘度，减少样本数量
python run_gpt_physics_tests.py \
    --properties viscosity \
    --absolute-samples 20 \
    --relative-pairs 10
```

## 模型选择

### 16. 使用不同模型
```bash
# 使用GPT-4o模型
python run_gpt_physics_tests.py --model-version gpt-4o

# 使用Qwen模型
python run_gpt_physics_tests.py --model-version qwen-vl-max
```

## 输出控制

### 17. 自动输出目录后缀
```bash
# 系统会根据启用的功能自动添加后缀到输出目录
python run_gpt_physics_tests.py --concat-mode --frame-index
# 输出目录: gpt_physics_analysis_results_concat_frameindex

python run_gpt_physics_tests.py --icl-example --follow-oracle-test --properties friction
# 输出目录: gpt_physics_analysis_results_icl_oracle

python run_gpt_physics_tests.py --concat-mode --frame-index --icl-example --follow-oracle-test
# 输出目录: gpt_physics_analysis_results_concat_frameindex_icl_oracle

# 不同模型也会自动生成对应的目录名
python run_gpt_physics_tests.py --model-version gemini-1.5-pro --icl-example
# 输出目录: gemini_physics_analysis_results_icl
```

### 18. 指定输出目录
```bash
# 用户指定的输出目录不会被自动后缀覆盖
python run_gpt_physics_tests.py \
    --properties friction \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --output-dir my_custom_results
# 输出目录: my_custom_results（保持用户指定的名称）
```

## 视频处理参数

### 19. 调整视频帧参数
```bash
# 调整视频帧尺寸
python run_gpt_physics_tests.py --frame-width 320 --frame-height 240

# 高分辨率测试
python run_gpt_physics_tests.py --frame-width 1280 --frame-height 720
```

## 综合示例

### 20. 快速测试（少量样本）
```bash
# 快速测试单个属性，少量样本，启用所有功能
python run_gpt_physics_tests.py \
    --properties friction \
    --absolute-samples 5 \
    --relative-pairs 3 \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --dry-run
```

### 21. 完整测试配置
```bash
# 完整的测试配置（包含所有新功能）
python run_gpt_physics_tests.py \
    --model-version gpt-4o \
    --properties friction viscosity \
    --absolute-samples 30 \
    --relative-pairs 20 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --frame-width 640 \
    --frame-height 480 \
    --output-dir comprehensive_test_results \
    --auto-confirm
```

### 22. 对比测试
```bash
# 传统模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --output-dir traditional_friction_results

# 拼接模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --concat-mode \
    --output-dir concat_friction_results

# 帧索引模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --frame-index \
    --output-dir frame_index_friction_results

# ICL模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --icl-example \
    --output-dir icl_friction_results

# Oracle模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --follow-oracle-test \
    --output-dir oracle_friction_results

# 全功能模式测试
python run_gpt_physics_tests.py \
    --properties friction \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --output-dir full_feature_friction_results
```

### 23. 大规模测试
```bash
# 大规模测试（高样本数）
python run_gpt_physics_tests.py \
    --absolute-samples 100 \
    --relative-pairs 50 \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --output-dir large_scale_results \
    --auto-confirm
```

### 24. 开发测试（极少样本）
```bash
# 开发阶段的快速验证
python run_gpt_physics_tests.py \
    --properties elasticity viscosity \
    --absolute-samples 1 \
    --relative-pairs 1 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --model-version gemini-2.5-pro \
    --auto-confirm
```

## 错误处理示例

### 25. 查看帮助信息
```bash
python run_gpt_physics_tests.py --help
```

### 26. 无效参数示例（会报错）
```bash
# 无效的属性名称（会被argparse拦截）
python run_gpt_physics_tests.py --properties invalid_property

# 无效的模型版本（程序会处理）
python run_gpt_physics_tests.py --model-version invalid_model
```

## 实际使用建议

### 开发阶段
```bash
# 快速验证功能
python run_gpt_physics_tests.py --properties friction --absolute-samples 3 --relative-pairs 2 --dry-run
```

### 测试阶段
```bash
# 中等规模测试
python run_gpt_physics_tests.py --properties friction viscosity --absolute-samples 20 --relative-pairs 10
```

### 生产阶段
```bash
# 完整测试
python run_gpt_physics_tests.py --absolute-samples 50 --relative-pairs 30 --auto-confirm
``` 