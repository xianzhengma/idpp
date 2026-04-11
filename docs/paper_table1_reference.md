# Paper Table 1 — reproduction and open-source reference

This document transcribes the main paper table **Table 1. Results for relative value comparison and absolute value prediction.** It is the **authoritative numeric reference** for open-source checks, self-tests, and README claims (paper-published version).

---

## 1. Metrics

| Block | Metric | Range | Meaning (as in the paper) |
|------|--------|-------|-----------------------------|
| **Rel. Test-1/2/3** | ROC AUC | [0, 1] | Relative ordering / pairwise comparison |
| **Abs. Test-1/2/3** | Pearson Corr. | [-1, 1] | Absolute value prediction vs. ground truth |
| **\*** | — | — | Result after **domain adaptation** on real-domain data disjoint from the test set |

**Test splits (paper)**

- **Test-1**: synthetic, in-distribution test split  
- **Test-2**: synthetic, out-of-distribution test split  
- **Test-3**: real-world test split  

---

## 2. Mapping to repository codenames

In the open-source plan and internal index notes:

| Method in the table | Codename here | Notes |
|---------------------|---------------|--------|
| **DynamiCrafter** | **E** | Aligns with internal `e729_nd_*` newdata runs and checkpoint notes. |
| **V-JEPA-2** | **B** | V-JEPA mirror evaluation; aligns with internal B-line checkpoints. |

**Gate (current open-source plan)**: prioritize **Elasticity, Test-3, B and E, relative + absolute** as the acceptance gate; full-table values below remain the reference for other properties.

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

## 6. Maintenance

- If the paper corrects Table 1, **update the paper first**, then sync this file and record the date and cells in a changelog.  
- If internal score logs diverge from this table, **this file (paper table)** is the external reference; document rounding, subset, or protocol differences in technical notes.  
- **E (DynamiCrafter)**: if `e729_nd_*` runs and `calculate_e729_newdata_dc_scores.py` disagree with the **DynamiCrafter** row here, treat **this table** as ground truth and debug data subset, post-processing, or camera-ready rounding.

---

## Changelog

- **2026-04-11**: Full transcription from paper Table 1; cross-links to B/E codenames.
