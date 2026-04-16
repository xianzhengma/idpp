# Hugging Face (datasets & checkpoints)

This file is a **short hub** for anything hosted on Hugging Face. **Readers** who only want to download and run eval should start with the root [`README.md`](../README.md), then [`HF_DATASET_UPLOAD.md`](HF_DATASET_UPLOAD.md) and [`HF_ASSETS_PAPER_ALIGNMENT.md`](HF_ASSETS_PAPER_ALIGNMENT.md). **Table 1 numbers** for self-checks: [`paper_table1_reference.md`](paper_table1_reference.md).

## Published repositories

| Role | Default ID (override with env vars) |
|------|-------------------------------------|
| **Dataset** (video zips) | [`Oliver-Ma/IDPP-Data`](https://huggingface.co/datasets/Oliver-Ma/IDPP-Data) — `HF_SVD_DATA_REPO` |
| **Checkpoints** | [`Oliver-Ma/IDPP-Checkpoints`](https://huggingface.co/Oliver-Ma/IDPP-Checkpoints) — `HF_SVD_CHECKPOINTS_REPO` |

Which files correspond to which paper rows is documented in [**HF_ASSETS_PAPER_ALIGNMENT.md**](HF_ASSETS_PAPER_ALIGNMENT.md). After any upload batch, **pin the dataset/model revision** (Hub commit hash) in your release notes or in [`IDPP_Data_README.md`](IDPP_Data_README.md) frontmatter where applicable.

## Download (smoke or mirror)

From the **repository root** (`SVD_public/`):

```bash
export HF_SVD_CHECKPOINTS_REPO=Oliver-Ma/IDPP-Checkpoints
export HF_SVD_DATA_REPO=Oliver-Ma/IDPP-Data
export SVD_ROOT=/path/to/local_asset_root
pip install -U huggingface_hub
python scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

`scripts/download_hf_assets.py` reads the same env vars; layout under `SVD_ROOT` should stay consistent with [`HF_ASSETS_PAPER_ALIGNMENT.md`](HF_ASSETS_PAPER_ALIGNMENT.md).