# Elasticity相对比较实验 - 单独命令

## 实验配置
- **模型**: Gemini-2.5-pro
- **属性**: elasticity
- **测试类型**: 仅相对比较 (relative)
- **样本数**: 50个相对比较对
- **测试集**: test1, test2, test3

## 单独运行命令

### 实验 1: Baseline
```bash
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples \
    --auto-confirm
```

### 实验 2: Baseline + Concat
```bash
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat \
    --auto-confirm
```

### 实验 3: Baseline + Concat + FrameIndex
```bash
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex \
    --auto-confirm
```

### 实验 4: Baseline + Concat + FrameIndex + ICL
```bash
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl \
    --auto-confirm
```

### 实验 5: Baseline + Concat + FrameIndex + ICL + Oracle
```bash
python run_gpt_physics_tests.py \
    --model-version gemini-2.5-pro \
    --absolute-samples 0 \
    --relative-pairs 50 \
    --concat-mode \
    --frame-index \
    --icl-example \
    --follow-oracle-test \
    --properties elasticity \
    --output-dir gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl_oracle \
    --auto-confirm
```

## 批量运行
```bash
bash run_elasticity_comparison_experiments.sh
```

## 结果分析
实验完成后，运行以下命令分析结果：
```bash
python process_individual_model_results_ablation.py
```

## 输出目录
实验将生成以下目录：
- `gemini_physics_analysis_results_elasticity_relative_50samples`
- `gemini_physics_analysis_results_elasticity_relative_50samples_concat`
- `gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex`
- `gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl`
- `gemini_physics_analysis_results_elasticity_relative_50samples_concat_frameindex_icl_oracle`

## 预期结果
- 每个实验将测试3个测试集（test1, test2, test3）
- 每个测试集50个相对比较对
- 总共150个比较测试每个实验
- 预计每个实验耗时约30分钟 