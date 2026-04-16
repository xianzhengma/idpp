---
license: apache-2.0
language: en
pretty_name: IDPP-Data
tags:
  - physics
  - video
  - benchmark
---

# IDPP-Data

Video archives for **IDPP** / **SVD** physics-benchmark reproduction: **friction**, **viscosity**, and **elasticity**, each with **absolute** and **relative** splits (train / test1 / test2 / test3 where applicable).

**Repository:** [`Oliver-Ma/IDPP-Data`](https://huggingface.co/datasets/Oliver-Ma/IDPP-Data)  
**Snapshot (README revision):** 2026-04-16 — documents **26** `.zip` blobs under `video_zips/` (mirroring local `dataset/<lane>/`). Elasticity **test3** defaults align with **`data_global_map.md`** (merged absolute + `2.16_elasticity_test3` relative).

**Public / partner policy:** released archives should **not** ship `_seg.mp4` or `*red_circle*.mp4` (RGB `*_rgb.mp4` only for video). **test1** and **test2** sample sets are trimmed to the **Oracle JSON** lists (`ORACLE_JSON_DIR`, see `data/oracle_lists/README.md`). Regenerate zips with `scripts/sanitize_repack_idpp_zips.py` before bulk re-upload; training code in `SVD_public/lvdm/data/webvid.py` supports **RGB-only** folders when all `red_circle*` flags are off.

## Important: archives, not frame folders

Everything under **`video_zips/`** is a **`.zip` file**. Training and evaluation code (e.g. WebVid-style loaders) expect a **directory** as `data_dir`. **Unzip** each bundle you need, then point `data_dir` at the **extracted folder** (usually the top-level directory inside the zip).

Example:

```bash
huggingface-cli download Oliver-Ma/IDPP-Data --repo-type dataset \
  --local-dir ./idpp_data
unzip ./idpp_data/video_zips/friction_absolute/4.29_friction_synthetic_test1.zip -d ./idpp_unpacked
# Then set data_dir to the inner folder that contains the rgb/mp4 tree.
```

Cross-check names and semantics with the project **`data_global_map.md`** (paths and splits).

## Layout on the Hub

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

## Index: `video_zips/` (26 files)

| Path on Hub |
|-------------|
| `video_zips/elasticity_absolute/4.11_mpm_bounce_ball_videos_test1_sorted.zip` |
| `video_zips/elasticity_absolute/4.11_mpm_bounce_ball_videos_test2_sorted.zip` |
| `video_zips/elasticity_absolute/4.11_mpm_bounce_ball_videos_train2_sorted.zip` |
| `video_zips/elasticity_absolute/7.15_new_elasticity_video_clips_merged.zip` |
| `video_zips/elasticity_relative/5.24_elasticity_synthetic_test1.zip` |
| `video_zips/elasticity_relative/5.24_elasticity_synthetic_test2.zip` |
| `video_zips/elasticity_relative/5.24_elasticity_synthetic_train.zip` |
| `video_zips/elasticity_relative/2.16_elasticity_test3.zip` |
| `video_zips/friction_absolute/4.29_friction_synthetic_test1.zip` |
| `video_zips/friction_absolute/4.29_friction_synthetic_test2.zip` |
| `video_zips/friction_absolute/4.29_friction_synthetic_train.zip` |
| `video_zips/friction_absolute/7.18_friction_test3_rgb_seg_videos_152frames_abs_test.zip` |
| `video_zips/friction_absolute/7.18_friction_test3_rgb_seg_videos_152frames_abs_train.zip` |
| `video_zips/friction_relative/5.24_friction_synthetic_test1.zip` |
| `video_zips/friction_relative/5.24_friction_synthetic_test2.zip` |
| `video_zips/friction_relative/5.24_friction_synthetic_train.zip` |
| `video_zips/friction_relative/7.18_friction_test3_rgb_seg_videos_152frames_relative_test.zip` |
| `video_zips/friction_relative/7.18_friction_test3_rgb_seg_videos_152frames_relative_train.zip` |
| `video_zips/viscosity_absolute/5.2_viscosity_synthetic_test1.zip` |
| `video_zips/viscosity_absolute/5.2_viscosity_synthetic_test2.zip` |
| `video_zips/viscosity_absolute/5.2_viscosity_synthetic_train.zip` |
| `video_zips/viscosity_absolute/7.15_new_viscosity_video_clips_16frames_merged.zip` |
| `video_zips/viscosity_relative/5.24_viscosity_synthetic_test1.zip` |
| `video_zips/viscosity_relative/5.24_viscosity_synthetic_test2.zip` |
| `video_zips/viscosity_relative/5.24_viscosity_synthetic_train.zip` |
| `video_zips/viscosity_relative/7.15_new_viscosity_video_clips_152frames_relative_merged_revised2.zip` |

**Elasticity test3 (Table 1 / §3):** Hub ships **`7.15_new_elasticity_video_clips_merged.zip`** and **`2.16_elasticity_test3.zip`**, matching **`data_global_map.md`** genesis paths. Bundled list JSONs under `SVD_public/data/lists/` are copied from **`/work/guanqi/Genesis/used_video_list_3.22/`**. Deprecated blobs **`7.15_new_elasticity_video_clips_16frames.zip`** and **`7.14_elasticity_test3.zip`** should be removed from the dataset repo after the new zips are uploaded so `upload_idpp_hub_zips.py` does not republish stale splits.

Pack genesis trees locally: [`scripts/repack_elasticity_test3_paper_zips.py`](scripts/repack_elasticity_test3_paper_zips.py) → then [`scripts/sanitize_repack_idpp_zips.py`](scripts/sanitize_repack_idpp_zips.py) (RGB-only policy) → upload via [`scripts/upload_idpp_hub_zips.py`](scripts/upload_idpp_hub_zips.py) (`--single-zip` or full scan).

**Notes**

- **Friction** `7.18_*_train.zip` under `friction_*` are the **test3-related train / finetune** style bundles (two zips per abs/rel lane); align with your paper / config naming.
- **Viscosity absolute** includes `7.15_new_viscosity_video_clips_16frames_merged.zip` (16-frame variant). If your paper uses the **152-frame merged** absolute test3 split, confirm against **`data_global_map.md`** and obtain that bundle from the canonical genesis copy if it is not in this repo.

## Environment variable

For download helpers in the public code tree:

```bash
export HF_SVD_DATA_REPO=Oliver-Ma/IDPP-Data
```

## Code

- Upstream / paper code: [idpp](https://github.com/xianzhengma/idpp).
- **Using the Hub snapshot (download + layout):** [`docs/HF_DATASET_UPLOAD.md`](HF_DATASET_UPLOAD.md). Maintainer bulk zip scripts: `scripts/upload_idpp_hub_zips.py`, `scripts/sbatch_upload_idpp_hub_zips.sh`.

