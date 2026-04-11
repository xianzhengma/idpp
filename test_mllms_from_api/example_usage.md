# 动态输出目录功能使用示例

## 概述

系统现在支持根据模型版本自动生成输出目录，不再固定使用 `gpt_physics_analysis_results`。

## 支持的模型版本和对应输出目录

### GPT模型系列
- `gpt-4o` → `gpt_physics_analysis_results`
- `gpt-4o-mini` → `gpt_mini_physics_analysis_results`
- `gpt-4-turbo` → `gpt_turbo_physics_analysis_results`
- 其他gpt模型 → `gpt_physics_analysis_results`

### Gemini模型系列
- `gemini-2.5-pro` → `gemini_physics_analysis_results`
- `gemini-1.5-pro` → `gemini_physics_analysis_results`
- 其他gemini模型 → `gemini_physics_analysis_results`

### Claude模型系列
- `claude-3-opus` → `claude_physics_analysis_results`
- `claude-3-sonnet` → `claude_physics_analysis_results`
- 其他claude模型 → `claude_physics_analysis_results`

### 其他模型
- `llama-3-70b` → `llama_physics_analysis_results`
- 任何其他模型 → `{模型前缀}_physics_analysis_results`

## 使用示例

### 1. 使用默认输出目录（根据模型版本自动生成）

```bash
# 使用gpt-4o模型，输出到 gpt_physics_analysis_results
python run_gpt_physics_tests.py --model-version gpt-4o --dry-run

# 使用gemini模型，输出到 gemini_physics_analysis_results
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --dry-run

# 使用gpt-4o-mini模型，输出到 gpt_mini_physics_analysis_results
python run_gpt_physics_tests.py --model-version gpt-4o-mini --dry-run
```

### 2. 手动指定输出目录（覆盖默认行为）

```bash
# 即使使用gemini模型，也可以手动指定输出目录
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --output-dir my_custom_results --dry-run
```

### 3. 相对测试专用工具

```bash
# 使用默认输出目录（根据模型版本自动生成）
python test_relative_only.py --version v125_vjepa --model-version gemini-2.5-pro --samples 5

# 手动指定输出目录
python test_relative_only.py --version v125_vjepa --model-version gemini-2.5-pro --output-dir gemini_relative_test --samples 5
```

## 优势

1. **自动组织结果**：不同模型的测试结果自动分开存储，避免混淆
2. **灵活性**：用户仍可以手动指定输出目录
3. **一致性**：主测试运行器和相对测试工具都支持相同的逻辑
4. **扩展性**：容易添加新的模型支持

## 注意事项

- 如果不指定 `--output-dir`，系统会根据 `--model-version` 自动生成目录名
- 如果指定了 `--output-dir`，会使用用户指定的目录，忽略自动生成的目录
- 目录名中的特殊字符（如 `-` 和 `.`）会被转换为下划线（`_`）以确保目录名的有效性 