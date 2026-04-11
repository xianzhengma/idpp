# 输出目录命名更新总结

## 🔄 修改内容

已将所有6个脚本中的输出目录命名从包含任务名改为不包含任务名：

### 修改前 ❌
```
gemini_physics_analysis_results_{task}_{type}_100samples[_{feature}]
```

示例：
- `gemini_physics_analysis_results_friction_absolute_100samples`
- `gemini_physics_analysis_results_viscosity_relative_100samples_frameindex`
- `gemini_physics_analysis_results_elasticity_absolute_100samples_oracle`

### 修改后 ✅
```
gemini_physics_analysis_results_{type}_100samples[_{feature}]
```

示例：
- `gemini_physics_analysis_results_absolute_100samples`
- `gemini_physics_analysis_results_relative_100samples_frameindex`
- `gemini_physics_analysis_results_absolute_100samples_oracle`

## 📋 影响的脚本

所有6个脚本都已更新：

### Absolute实验脚本
1. `run_friction_absolute_ablation_100samples.sh`
2. `run_viscosity_absolute_ablation_100samples.sh`
3. `run_elasticity_absolute_ablation_100samples.sh`

所有输出目录现在都使用：
- `gemini_physics_analysis_results_absolute_100samples`
- `gemini_physics_analysis_results_absolute_100samples_frameindex`
- `gemini_physics_analysis_results_absolute_100samples_oracle`

### Relative实验脚本
4. `run_friction_relative_ablation_100samples.sh`
5. `run_viscosity_relative_ablation_100samples.sh`
6. `run_elasticity_relative_ablation_100samples.sh`

所有输出目录现在都使用：
- `gemini_physics_analysis_results_relative_100samples`
- `gemini_physics_analysis_results_relative_100samples_frameindex`
- `gemini_physics_analysis_results_relative_100samples_icl`
- `gemini_physics_analysis_results_relative_100samples_concat`

## ⚠️ 重要说明

### 目录重用
由于移除了任务名，现在**不同任务的相同配置会使用同一个输出目录**：

- Friction、Viscosity、Elasticity的Baseline实验都会使用 `gemini_physics_analysis_results_absolute_100samples`
- 这意味着后运行的实验会**覆盖**先运行的实验结果

### 建议的运行策略

1. **串行运行不同类型的实验**：
   ```bash
   # 先运行所有absolute实验
   ./run_friction_absolute_ablation_100samples.sh
   ./run_viscosity_absolute_ablation_100samples.sh
   ./run_elasticity_absolute_ablation_100samples.sh
   
   # 再运行所有relative实验
   ./run_friction_relative_ablation_100samples.sh
   ./run_viscosity_relative_ablation_100samples.sh
   ./run_elasticity_relative_ablation_100samples.sh
   ```

2. **或者手动备份结果**：
   每个任务完成后立即备份结果目录。

## 🎯 优势

- **目录结构更简洁**：不包含冗余的任务名信息
- **统一的输出格式**：所有实验使用一致的目录命名
- **便于批处理**：可以更容易地对所有任务的结果进行统一处理

修改已完成，所有脚本都已更新并准备运行！