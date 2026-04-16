# Hugging Face dataset — how to use it

This page is for **readers who reproduce the paper**: where the data lives on the Hub, what the tree means, and how to get it onto disk. (Maintainer upload steps are not documented here.)

## Dataset repository

**[`Oliver-Ma/IDPP-Data`](https://huggingface.co/datasets/Oliver-Ma/IDPP-Data)** — physics-benchmark videos for **friction**, **viscosity**, and **elasticity**, each in **absolute** and **relative** lanes, with **train / test1 / test2 / test3** splits where applicable.

Set this when using the download helper in the code repo:

```bash
export HF_SVD_DATA_REPO=Oliver-Ma/IDPP-Data
```

## What you get after download

The Hub stores **archives**, not an unpacked frame tree. Under the repo root you will see:

```text
README.md
video_zips/
  elasticity_absolute/*.zip
  elasticity_relative/*.zip
  friction_absolute/*.zip
  friction_relative/*.zip
  viscosity_absolute/*.zip
  viscosity_relative/*.zip
```

Each split is one or more **`.zip`** files. The evaluation / training loaders expect a **directory** as `data_dir`, so you must **unzip** the bundles you need and point YAML `data_dir` at the **folder that contains the video files** (typically `*_rgb.mp4` trees inside the archive).

## Download options

**A. This repo’s helper** (writes under your `SVD_ROOT`, see root `README.md`):

```bash
python3 scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

**B. Hugging Face CLI** (any local folder):

```bash
huggingface-cli download Oliver-Ma/IDPP-Data --repo-type dataset --local-dir ./idpp_data
```

## Elasticity test-3 convenience staging

If you use the default Hub zips for the public elasticity **test3** setup, you can unpack into the layout expected by the bundled eval templates with:

```bash
python3 scripts/stage_elasticity_test3_from_hf.py \
  --svd-root "$SVD_ROOT" \
  --data-hf "$SVD_ROOT/data_hf"
```

## More detail

- **Full zip index and revision notes:** [IDPP_Data_README.md](IDPP_Data_README.md) (also used as the Hub dataset card source).
- **Which blobs align with which paper rows:** [HF_ASSETS_PAPER_ALIGNMENT.md](HF_ASSETS_PAPER_ALIGNMENT.md).
