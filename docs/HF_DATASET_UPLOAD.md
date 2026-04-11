# Hugging Face dataset upload (efficient workflow)

Use this together with **`data_global_map.md`** (project root) and **`configs/hf_dataset_manifest.example.yaml`**.

## What goes on the Hub

- **24 video bundles**: 3 properties × (absolute | relative) × (train | test1 | test2 | test3).
- **List JSON** under `used_video_list_3.22` (see `data_global_map.md` §使用说明) — ship as `lists/` (or `metadata/lists/`) in the same dataset repo.
- **Auxiliary GT / homography JSON** referenced in `data_global_map.md` (friction test3 real, etc.) — ship under `metadata/` so configs can use relative paths inside the snapshot.

**Canonical eval copies** in your notes live under **`/scratch/shared/beegfs/guanqi/genesis_data/`**; **training + some mirrors** live under **`/work/guanqi/proj5/Inferring_dynamic_physics/dataset/`** (`datasets_0906` is a symlink to `dataset`). Staging should **merge** both roots into **one Hub-friendly tree** (no machine-specific prefixes).

## Recommended Hub layout (repo-relative)

```text
friction/
  absolute/{train,test1,test2,test3}/
  relative/{train,test1,test2,test3}/
viscosity/
  absolute/{train,test1,test2,test3}/
  relative/{train,test1,test2,test3}/
elasticity/
  absolute/{train,test1,test2,test3}/
  relative/{train,test1,test2,test3}/
lists/          # used_video_list_3.22 copies or generated lists
metadata/       # bounce_analysis_results.json, etc.
README.md       # dataset card
```

Unpack zips **before** upload if training configs expect directories of `*_rgb.mp4`.

## Efficient upload options

### A. `upload-large-folder` (many small files)

Best when the staging folder already looks like the layout above.

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli upload-large-folder YOUR_ORG/svd-physics-eval-data ./hf_dataset_staging \
  --repo-type=dataset
```

Supports resume and multipart uploads; good for millions of frames.

### B. Fewer, larger blobs (tar per split)

If the CLI is slow on metadata-heavy trees:

```bash
tar -cf friction_relative_train.tar -C /path/to/5.24_friction_synthetic_train .
# upload the .tar with huggingface-cli upload or the Hub UI; document unpack in README
```

Trade-off: users must extract before training; smaller file count, simpler re-upload.

### C. Two-phase upload

1. **Phase 1 — tests (collaborator / genesis):** rsync or manifest `src` = `genesis_data/...` into `staging/.../test*`.
2. **Phase 2 — train (`dataset/`):** append `train` rows from `dataset/` (see `data_global_map.md` **Train** sections).

Re-run `upload-large-folder` with `--exclude` patterns if you only add `**/train/**`.

## Staging helper

```bash
cp configs/hf_dataset_manifest.example.yaml hf_dataset_manifest.yaml
# Edit SRC paths to match your machine (genesis + dataset)
python scripts/stage_hf_dataset.py --manifest hf_dataset_manifest.yaml --out ./hf_dataset_staging --symlink
# Inspect, then remove --symlink and use --copy for a self-contained tree if needed
huggingface-cli upload-large-folder YOUR_ORG/svd-physics-eval-data ./hf_dataset_staging --repo-type=dataset
```

`--symlink` is fast and saves disk; some filesystems disallow symlinks in upload — if upload fails, use `--copy`.

## After upload

- Set `HF_SVD_DATA_REPO` and pin **revision** in `SVD_public/README.md` / dataset card.
- Point `SVD_ROOT/data` (or yaml `data_dir`) at the downloaded snapshot layout **matching** this tree so `paper_scores_paths_and_checkpoints.md` §3 yamls stay valid.
