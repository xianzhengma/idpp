# 论文 Table 1 — 复现与开源对照基准

本文档由论文主表 **Table 1. Results for relative value comparison and absolute value prediction.** 整理而成，作为开源清理、自检与 README 的**数值对照权威**（以论文发表版本为准）。

---

## 1. 指标说明

| 列块 | 指标 | 取值范围 | 含义（与论文一致） |
|------|------|----------|-------------------|
| **Rel. Test-1/2/3** | ROC AUC | [0, 1] | 相对大小比较 |
| **Abs. Test-1/2/3** | Pearson Corr. | [-1, 1] | 绝对值预测与真值相关 |
| **\*** | — | — | 使用与测试不相交的真实域数据做 **domain adaptation** 后的结果 |

**测试划分（论文定义）**

- **Test-1**：合成数据、分布内测试划分  
- **Test-2**：合成数据、分布外测试划分  
- **Test-3**：真实世界测试划分  

---

## 2. 与本仓库版本代号对应

开源计划与内部索引文档中：

| 论文表中的方法 | 本仓库代号 | 说明与索引 |
|----------------|------------|------------|
| **DynamiCrafter** | **E** | DynamicCrafter；论文表数字与 [`paper_scores_paths_and_checkpoints.md`](../paper_scores_paths_and_checkpoints.md) **§3 newdata（`e729_nd_*`）** 对齐（见该文档与 [`paper_scores_E_7.29.md`](../paper_scores_E_7.29.md)）。 |
| **V-JEPA-2** | **B** | V-JEPA mirror 评测；与 [`paper_scores_B_sources_and_checkpoints.md`](../paper_scores_B_sources_and_checkpoints.md) **B 列** 对齐。 |

**自检约定（与当前开源计划一致）**：优先以 **Elasticity、Test-3、B 与 E、相对+绝对** 四条线为门禁；全表数值仍以下文各小节为准，便于扩展其它 property。

---

## 3. Elasticity

| Method | Rel. Test-1 | Rel. Test-2 | Rel. Test-3 | Abs. Test-1 | Abs. Test-2 | Abs. Test-3 |
|--------|------------:|------------:|------------:|------------:|------------:|------------:|
| Oracle | 1.00 | 0.99 | 1.00 | 0.99 | 0.98 | 0.87 |
| DynamiCrafter **(E)** | 1.00 | 0.99 | 0.61 | 0.92 | 0.78 | 0.10 |
| V-JEPA-2 **(B)** | 1.00 | 1.00 | 0.99 | 0.89 | 0.65 | 0.37 |
| Qwen2.5VL-max | 0.59 | 0.50 | 0.54 | -0.05 | 0.11 | 0.16 |
| GPT-4o | 0.51 | 0.66 | 0.62 | 0.19 | 0.11 | 0.30 |
| Gemini-2.5-pro | 0.64 | 0.80 | 0.47 | 0.04 | 0.15 | 0.24 |

---

## 4. Viscosity

| Method | Rel. Test-1 | Rel. Test-2 | Rel. Test-3 | Abs. Test-1 | Abs. Test-2 | Abs. Test-3 |
|--------|------------:|------------:|------------:|------------:|------------:|------------:|
| Oracle | 0.99 | 0.99 | 1.00 | 0.99 | 0.98 | 0.80 |
| DynamiCrafter **(E)** | 1.00 | 1.00 | 0.93 | 0.99 | 0.95 | 0.44 |
| V-JEPA-2 **(B)** | 1.00 | 0.94 | 0.99 | 0.98 | 0.69 | 0.61 |
| Qwen2.5VL-max | 0.64 | 0.61 | 0.86 | 0.16 | 0.06 | 0.02 |
| GPT-4o | 0.63 | 0.59 | 0.99 | 0.18 | 0.08 | 0.55 |
| Gemini-2.5-pro | 0.48 | 0.69 | 0.95 | -0.06 | -0.05 | 0.60 |

---

## 5. Friction

| Method | Rel. Test-1 | Rel. Test-2 | Rel. Test-3 | Abs. Test-1 | Abs. Test-2 | Abs. Test-3 |
|--------|------------:|------------:|------------:|------------:|------------:|------------:|
| Oracle | 1.00 | 1.00 | 0.87 | 0.99 | 1.00 | 0.83 |
| DynamiCrafter **(E)** | 0.96 | 0.90 | 0.97 | 0.95 | 0.78 | 0.25 |
| *+ Domain Adaptation* (DynamiCrafter) | – | – | – | – | – | **0.80\*** |
| V-JEPA-2 **(B)** | 0.97 | 0.90 | 0.48 | 0.87 | 0.56 | 0.21 |
| *+ Domain Adaptation* (V-JEPA-2) | – | – | **0.87\*** | – | – | **0.73\*** |
| Qwen2.5VL-max | 0.52 | 0.48 | 0.80 | -0.03 | 0.03 | 0.06 |
| GPT-4o | 0.49 | 0.40 | 0.67 | 0.07 | 0.10 | 0.38 |
| Gemini-2.5-pro | 0.56 | 0.52 | 0.97 | 0.01 | 0.10 | 0.12 |

---

## 6. 维护说明

- 若论文勘误或相机稿更新 Table 1，应**先改论文**，再同步更新本文件，并在 changelog 中记录日期与变更单元格。  
- 内部 `paper_scores_*.md` 若与主表出现系统性差异，以 **本文件（论文表）** 为对外宣称基准，并在技术说明中解释舍入、子集或协议差异（例如 E 侧 newdata 与表中一位小数口径）。
- **E（DynamiCrafter）**：工程上跑 `e729_nd_*` 与 `calculate_e729_newdata_dc_scores.py` 时，若得到的数值与上表中 **DynamiCrafter** 行不一致，仍以 **本文件表格** 为论文事实，并排查数据子集、后处理或与 camera-ready 表是否同源。

---

## Changelog

- **2026-04-11**：从论文 Table 1 转录全文；建立与仓库 **B / E** 代号及索引文档的交叉引用。
