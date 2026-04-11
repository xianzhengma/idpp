# SVD Physics（公开复现树）

**GitHub 仓库**：<https://github.com/xianzhengma/idpp>

本目录为与 **`Inferring_dynamic_physics/SVD` 原工程隔离** 的开源复现仓库：训练 / 评测 DynamiCrafter（论文表 **E**）与 V-JEPA-2（论文表 **B**），并以 **论文 Table 1** 为最终验收基准。

## 当前阶段：GitHub 代码先行版（无权重、无数据集）

**本仓库推送内容仅包含**：代码、`configs/` 模板与生成脚本、文档、`environment.yml` / `requirements.txt`。  
**不包含** checkpoint、评测视频或 Hugging Face 上的模型/数据发布；合作者克隆后即可浏览结构与接口。完整可复现评测需在本地配置 **`SVD_ROOT`** 下的权重与数据；**Hugging Face 上的模型与数据集将后续单独上传**（见 [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md)）。

- **主表对照**：[docs/paper_table1_reference.md](docs/paper_table1_reference.md)
- **HF 占位说明**：[docs/HUGGINGFACE.md](docs/HUGGINGFACE.md)
- **环境**：`environment.yml`（由 `dynamicrafter_vjepa` conda 环境导出并去除 `prefix`）；或 `pip install -r requirements.txt`

## 目录约定：`SVD_ROOT`

所有权重、数据、评测 JSON 输出建议放在同一根目录下，并 **导出环境变量**：

```bash
export SVD_ROOT=/path/to/asset_root   # 见 configs/paths.example.yaml
```

布局生成可使用：

```bash
python scripts/resolve_eval_configs.py --svd-root "$SVD_ROOT"
# 写出 configs/reproduce/generated/*.yaml
```

## 下载权重与数据（Hugging Face，后续步骤）

待 Hub 仓库就绪后，将仓库 ID 写入环境变量再执行（当前占位 ID 会跳过下载，仅打印提示）：

```bash
export HF_SVD_CHECKPOINTS_REPO=your-org/svd-physics-checkpoints
export HF_SVD_DATA_REPO=your-org/svd-physics-eval-data
python scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

解压后请保证 **列表 JSON** 与 **视频根目录** 与模板中一致（`data/elasticity_*` 与 `data/lists/`）。

## 运行评测（Elasticity Test-3）

在 **GPU 机器**、激活环境后，于 `main/` 目录执行（示例：E 绝对 v124）：

```bash
cd main
python trainer.py --base ../configs/reproduce/generated/config_e729_nd_v124_test3_el_abs.yaml --val \
  --gpus 1 --precision 16
```

其余三条（E v125 rel、B v124 abs、B v125 rel）将 `--base` 换成对应生成 yaml。完整跑完后，JSON 应分别落在：

- ` $SVD_ROOT/results/e729_newdata_20260327/e729_nd_v124_test3_el_abs` 等（**E**）
- ` $SVD_ROOT/results/mirror_eval/v124_vjepa_test3` 等（**B**）

## 算分（E / newdata）

```bash
export SVD_ROOT=/path/to/asset_root
python calculate_e729_newdata_dc_scores.py
```

（或通过 `SVD_E_NEWDATA_RESULTS` 指向含 `e729_nd_*` 子目录的根。）

## 论文 Table 1 — Elasticity Test-3 自检

```bash
export SVD_ROOT=/path/to/asset_root
python scripts/verify_paper_elasticity_test3.py --atol 0.03
```

目标（主表）：**E** — Pearson 0.10、ROC-AUC 0.61；**B** — Pearson 0.37、ROC-AUC 0.99。若目录缺失或分数超容差，脚本以非零退出码结束。

## MLLM（`test_mllms_from_api/`）

- 将 `secrets/gpt_key.json.example` 复制为 `secrets/gpt_key.json` 并填入密钥，或设置 `GPT_KEY_JSON`。
- `vjepa_data_dirs.json` 已替换为占位路径；请按本地数据修改。
- 散点图输出目录由环境变量 `MLLM_SCATTER_OUT` 控制（默认 `outputs/mllm_scatter`）。

## 说明

- `lvdm/data/webvid.py` 中部分 **历史数据集类**仍含注释或旧默认路径，仅供开发参考；**正式评测以 `configs/reproduce/generated` 中 `data_dir` / `video_list_json` 为准**。
- 训练 smoke：可保留 `--train` 并减小 `max_steps`（需自行改 lightning 段）；本 README 以 **复现主表评测**为主。

## 许可证

见 [LICENSE](LICENSE)。数据集与第三方权重请单独遵守其许可。
