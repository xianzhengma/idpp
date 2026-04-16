# IDPP / SVD Physics (public release)

Training and evaluation code for **inferring dynamic physics from video**, in the same spirit as the public **IDPP** implementation: [xianzhengma/idpp](https://github.com/xianzhengma/idpp). This repository is the version we open-source; large weights and video archives are distributed via **Hugging Face** (see **Data** below).

---

## Environment

- **Python:** 3.9 (see `environment.yml`).
- **Install (recommended):**

  ```bash
  cd /path/to/this/repo
  conda env create -f environment.yml
  conda activate svd_physics_release
  ```

  Alternatively: `pip install -r requirements.txt` (you may need to resolve a few system libraries manually).

- **Runtime path:** from the repo root, point Python at `main/` and the repo root, e.g.

  ```bash
  export PYTHONPATH="/path/to/this/repo/main:/path/to/this/repo:${PYTHONPATH}"
  ```

- **Hardware:** GPU is expected for `trainer.py` training and validation.

---

## Data

We keep a single on-disk asset root, **`SVD_ROOT`**, for checkpoints, HF snapshot trees, unpacked videos, and run outputs (none of the large blobs are committed to git).

1. **Set the root**

   ```bash
   export SVD_ROOT=/path/to/your/svd_assets
   mkdir -p "$SVD_ROOT"
   ```

2. **Download from Hugging Face** (defaults are overridable via env vars):

   ```bash
   export HF_SVD_CHECKPOINTS_REPO=Oliver-Ma/IDPP-Checkpoints
   export HF_SVD_DATA_REPO=Oliver-Ma/IDPP-Data
   python3 scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
   ```

3. **Lay out evaluation data** under `$SVD_ROOT/data/` so it matches the `data_dir` / list paths in your YAML. For the **elasticity / test-3** split used in our public checks, you can stage from the HF zips with:

   ```bash
   python3 scripts/stage_elasticity_test3_from_hf.py \
     --svd-root "$SVD_ROOT" \
     --data-hf "$SVD_ROOT/data_hf"
   ```

4. **Resolver for bundled templates:** after data and checkpoints are in place, generate concrete configs (paths filled with `$SVD_ROOT`):

   ```bash
   python3 scripts/resolve_eval_configs.py --svd-root "$SVD_ROOT"
   ```

More detail on zip names, directory layout, and dataset cards: [docs/IDPP_Data_README.md](docs/IDPP_Data_README.md). How Hub assets map to paper rows: [docs/HF_ASSETS_PAPER_ALIGNMENT.md](docs/HF_ASSETS_PAPER_ALIGNMENT.md).

---

## Training

Entry point: **`main/trainer.py`** (PyTorch Lightning), same CLI style as the upstream IDPP stack.

```bash
cd /path/to/this/repo/main
export PYTHONPATH="/path/to/this/repo/main:/path/to/this/repo:${PYTHONPATH}"

python3 trainer.py --base /path/to/your_train_config.yaml --train --gpus 1 --precision 16
```

- **Config:** your YAML must define `model`, `data`, and `lightning` blocks compatible with this tree (`lvdm` models and `webvid`-style dataloaders). This repo ships **evaluation-oriented** templates under `configs/reproduce/templates/`; for full training recipes, start from configs in [idpp](https://github.com/xianzhengma/idpp) and adjust paths to your `SVD_ROOT` / dataset layout.
- **Resume:** `--auto_resume` / `--auto_resume_weight_only` are supported (see `trainer.py`).

---

## Testing (evaluation)

**1. Run validation / inference pass** (writes metrics and outputs according to your yaml):

```bash
cd /path/to/this/repo/main
export PYTHONPATH="/path/to/this/repo/main:/path/to/this/repo:${PYTHONPATH}"

python3 trainer.py --base ../configs/reproduce/generated/your_eval_config.yaml --val --gpus 1 --precision 16
```

Example resolved configs for **elasticity test-3** (four cells: E abs/rel, B abs/rel) appear as `configs/reproduce/generated/config_e729_nd_v124_test3_el_abs.yaml`, `..._v125_..._rel.yaml`, `config_v124_vjepa_test3.yaml`, `config_v125_vjepa_test3.yaml` after `resolve_eval_configs.py`.

**2. Aggregate scores** (E-side newdata protocol used in our paper table tooling):

```bash
cd /path/to/this/repo
export SVD_E_NEWDATA_RESULTS="${SVD_ROOT}/results/<your_e729_newdata_run_dir>"
python3 calculate_e729_newdata_dc_scores.py
```

**3. Optional sanity checks:** `scripts/verify_paper_elasticity_test3.py`, `scripts/smoke_elasticity_test3_dataloaders.py` (CPU-only dataloader smoke). Table-1 reference numbers: [docs/paper_table1_reference.md](docs/paper_table1_reference.md).

Cluster helpers (edit partitions for your site): `scripts/submit_hf_elasticity_test3_four_jobs.sh`, etc.

---

## Documentation & license

- Hub notes for maintainers: [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md). **Dataset download and folder layout (readers):** [docs/HF_DATASET_UPLOAD.md](docs/HF_DATASET_UPLOAD.md).
- **License:** [LICENSE](LICENSE). Third-party checkpoints and datasets remain under their own terms.

Optional subfolder **`test_mllms_from_api/`** is for separate MLLM experiments; do not commit API keys (see `.gitignore`).
