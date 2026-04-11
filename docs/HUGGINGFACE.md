# Hugging Face release notes

**Status**: tracked separately from the **code-first** GitHub release. **Model weights and the evaluation dataset are not on the Hub yet**; after upload, replace placeholder IDs in this document with real repos and pin **revision** in the README.

## Placeholder repos

`scripts/download_hf_assets.py` defaults to placeholder IDs:

- `HF_SVD_CHECKPOINTS_REPO` — holds subfolders such as `training_1024_v124_7.29`, `v125`, `v124_vjepa_norm`, `v125_vjepa_norm`, and checkpoint files.
- `HF_SVD_DATA_REPO` — holds evaluation videos and `lists/*.json` (consistent with `configs/reproduce` templates).

Before release, create real Hub repositories, lay out files as described here and in [`configs/paths.example.yaml`](../configs/paths.example.yaml), and document **revision** (commit hash) and optional `sha256sum` lists in the README.

## Maintainer snapshot commands

```bash
export HF_SVD_CHECKPOINTS_REPO=your-org/svd-physics-checkpoints
export HF_SVD_DATA_REPO=your-org/svd-physics-eval-data
export SVD_ROOT=/path/to/local_asset_root
pip install huggingface_hub
python scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

## Alignment with paper Table 1

Self-check script `scripts/verify_paper_elasticity_test3.py` uses **Elasticity / Test-3** (four cells) in [`paper_table1_reference.md`](paper_table1_reference.md); public weights and data should reproduce those within default `--atol 0.03`.
