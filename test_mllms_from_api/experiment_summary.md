# 对比实验计划总结

## 实验配置

### 基本设置
- **模型**: Gemini-2.5-pro
- **测试属性**: viscosity, elasticity
- **样本数**: 每个实验20个绝对 + 20个相对 = 40个样本
- **总实验数**: 10个

### 实验列表

#### 高优先级实验（4个）
1. **基础模式** - 标准prompt，作为baseline
2. **ICL示例模式** - 使用in-context learning示例
3. **Oracle指导模式** - 提供具体的分析步骤指导
4. **ICL + Oracle组合** - 两个重要功能的组合

#### 中优先级实验（5个）
5. **拼接模式** - 相对比较时使用拼接模式
6. **帧索引模式** - 为每帧添加显式索引
7. **ICL + 拼接组合** - ICL示例 + 拼接模式
8. **Oracle + 拼接组合** - Oracle指导 + 拼接模式
9. **ICL + Oracle + 帧索引** - 三个主要功能的组合

#### 低优先级实验（1个）
10. **全功能模式** - 所有功能的组合

## 实验统计

### 样本分布
- **每个实验**: 40个样本（20个绝对 + 20个相对）
- **总样本数**: 400个样本
- **测试集**: 每个属性3个测试集（test1, test2, test3）
- **总测试数**: 120个测试（10个实验 × 12个测试）

### 时间估算
- **每个实验**: 约30分钟
- **总时间**: 约5小时
- **建议**: 可以分阶段执行，先运行高优先级实验

## 执行命令

### 批量执行
```bash
bash run_recommended_experiments.sh
```

### 单个实验示例
```bash
# 基础模式
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --properties viscosity elasticity --auto-confirm

# ICL模式
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --icl-example --properties viscosity elasticity --auto-confirm

# Oracle模式
python run_gpt_physics_tests.py --model-version gemini-2.5-pro --absolute-samples 20 --relative-pairs 20 --follow-oracle-test --properties viscosity elasticity --auto-confirm
```

## 输出目录结构

实验完成后将生成以下目录：
```
gemini_physics_analysis_results/                                    # 基础模式
gemini_physics_analysis_results_icl-example/                       # ICL模式
gemini_physics_analysis_results_follow-oracle-test/                # Oracle模式
gemini_physics_analysis_results_icl-example_follow-oracle-test/    # ICL+Oracle
gemini_physics_analysis_results_concat-mode/                       # 拼接模式
gemini_physics_analysis_results_frame-index/                       # 帧索引模式
gemini_physics_analysis_results_icl-example_concat-mode/           # ICL+拼接
gemini_physics_analysis_results_follow-oracle-test_concat-mode/    # Oracle+拼接
gemini_physics_analysis_results_icl-example_follow-oracle-test_frame-index/ # ICL+Oracle+帧索引
gemini_physics_analysis_results_icl-example_follow-oracle-test_concat-mode_frame-index/ # 全功能
```

## 分析建议

### 第一阶段分析（高优先级实验）
1. 比较基础模式与ICL模式的效果
2. 评估Oracle指导的有效性
3. 分析ICL+Oracle组合的协同效应

### 第二阶段分析（中优先级实验）
1. 评估拼接模式对相对比较的影响
2. 分析帧索引对时序理解的作用
3. 测试不同功能组合的效果

### 第三阶段分析（全功能实验）
1. 评估所有功能组合的最终效果
2. 确定最佳prompt策略

## 预期结果

通过这些实验，您将能够：
1. **识别最佳单功能**: 哪个prompt方式对viscosity和elasticity最有效
2. **发现功能协同**: 哪些功能组合效果更好
3. **优化策略**: 为不同物理属性选择最佳prompt组合
4. **建立基准**: 为后续研究建立性能基准 