# SVD Physics (public reproduction)

**Repository:** <https://github.com/xianzhengma/idpp>

This tree is a standalone open-source reproduction codebase (separate from the internal `Inferring_dynamic_physics/SVD` workspace). It supports training and evaluation of **DynamiCrafter** (paper table **E**) and **V-JEPA-2** (paper table **B**). The acceptance target is **Table 1** in the paper.

## Current release: code-only on GitHub (no weights, no datasets)

This push includes **only** source code, `configs/` templates and generation scripts, documentation, `environment.yml`, and `requirements.txt`. It does **not** include checkpoints, evaluation videos, or published Hugging Face model/dataset artifacts. Collaborators can clone and inspect the structure and APIs. Full reproduction requires local assets under **`SVD_ROOT`**. **Model and dataset uploads to Hugging Face are planned as a follow-up** (see [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md)).

- **Table 1 reference:** [docs/paper_table1_reference.md](docs/paper_table1_reference.md)
- **Hugging Face notes:** [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md)
- **Environment:** `environment.yml` (exported from a `dynamicrafter_vjepa`-style conda env, `prefix` removed) or `pip install -r requirements.txt`

## Layout: `SVD_ROOT`

Place checkpoints, data, and evaluation JSON outputs under one root and set:

```bash
export SVD_ROOT=/path/to/asset_root   # see configs/paths.example.yaml
```

Generate resolved configs:

```bash
python scripts/resolve_eval_configs.py --svd-root "$SVD_ROOT"
# writes configs/reproduce/generated/*.yaml
```

## Download weights and data (Hugging Face, later)

**Which ckpts and which data dirs match the paper:** see [docs/HF_ASSETS_PAPER_ALIGNMENT.md](docs/HF_ASSETS_PAPER_ALIGNMENT.md). **E** is anchored to **`SVD/paper_scores_paths_and_checkpoints.md` §3** (new-data `e729_nd_*`) and **`calculate_e729_newdata_dc_scores.py`**; **B** to **`SVD/paper_scores_B_sources_and_checkpoints.md`** and mirror configs under the internal `SVD/` tree.

After Hub repositories exist, set their IDs and run (placeholder IDs skip download and only print hints):

```bash
export HF_SVD_CHECKPOINTS_REPO=your-org/svd-physics-checkpoints
export HF_SVD_DATA_REPO=your-org/svd-physics-eval-data
python scripts/download_hf_assets.py --svd-root "$SVD_ROOT"
```

After extraction, align **list JSON** files and **video roots** with the templates (`data/elasticity_*` and `data/lists/`).

## Run evaluation (Elasticity Test-3)

On a **GPU** machine with the environment activated, from `main/` (example: E, absolute, v124):

```bash
cd main
python trainer.py --base ../configs/reproduce/generated/config_e729_nd_v124_test3_el_abs.yaml --val \
  --gpus 1 --precision 16
```

For the other three runs (E v125 relative, B v124 absolute, B v125 relative), point `--base` to the matching generated YAML. Outputs should land under:

- `$SVD_ROOT/results/e729_newdata_20260327/e729_nd_v124_test3_el_abs` (and siblings) for **E**
- `$SVD_ROOT/results/mirror_eval/v124_vjepa_test3` (and siblings) for **B**

## Scoring (E / newdata protocol)

```bash
export SVD_ROOT=/path/to/asset_root
python calculate_e729_newdata_dc_scores.py
```

Alternatively set `SVD_E_NEWDATA_RESULTS` to the root that contains `e729_nd_*` subfolders.

## Paper Table 1 — Elasticity Test-3 self-check

```bash
export SVD_ROOT=/path/to/asset_root
python scripts/verify_paper_elasticity_test3.py --atol 0.03
```

Targets (Table 1): **E** — Pearson **0.10**, ROC-AUC **0.61**; **B** — Pearson **0.37**, ROC-AUC **0.99**. Missing folders or scores outside tolerance exit non-zero.

## MLLMs (`test_mllms_from_api/`)

- Copy `secrets/gpt_key.json.example` to `secrets/gpt_key.json` and fill in keys, or set `GPT_KEY_JSON`.
- `vjepa_data_dirs.json` uses placeholder paths; edit for your local or downloaded data.
- Scatter output directory: `MLLM_SCATTER_OUT` (default `outputs/mllm_scatter`).

## Notes

- Some legacy dataset classes in `lvdm/data/webvid.py` still mention old paths in comments or defaults; **use `data_dir` / `video_list_json` in `configs/reproduce/generated` for real runs**.
- Training smoke tests: use `--train` with a small `max_steps` (edit the lightning section). This README focuses on **Table 1 evaluation reproduction**.

## License

See [LICENSE](LICENSE). Datasets and third-party checkpoints remain under their own licenses.
