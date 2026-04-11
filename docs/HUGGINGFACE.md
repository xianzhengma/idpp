# Hugging Face 发布说明

## 仓库占位

在 **`scripts/download_hf_assets.py`** 中默认使用占位 ID：

- `HF_SVD_CHECKPOINTS_REPO` — 存放 `training_1024_v124_7.29` / `v125` / `v124_vjepa_norm` / `v125_vjepa_norm` 等子目录与 ckpt 文件。
- `HF_SVD_DATA_REPO` — 存放评测视频与 `lists/*.json`（与 `configs/reproduce` 模板一致）。

发布前请创建真实 Hub 仓库，将文件布局为本文档与 [`configs/paths.example.yaml`](../configs/paths.example.yaml) 所述，并在 README 中写明 **revision**（commit hash）与可选 `sha256sum` 清单。

## 推荐快照命令（维护者）

```bash
export HF_SVD_CHECKPOINTS_REPO=your-org/svd-physics-checkpoints
export HF_SVD_DATA_REPO=your-org/svd-physics-eval-data
export SVD_ROOT=/path/to/local_asset_root
pip install huggingface_hub
python scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

## 与论文 Table 1 对齐

自检脚本 **`scripts/verify_paper_elasticity_test3.py`** 以 [`paper_table1_reference.md`](paper_table1_reference.md) 中 **Elasticity / Test-3** 四格为基准；公开数据与权重应能复现该四格（默认容差 `--atol 0.03`）。
