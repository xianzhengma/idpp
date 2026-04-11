# Multimodal LLM physics benchmarks (GPT / Qwen / Gemini)

Optional tooling to score vision-language models on friction, viscosity, and elasticity using the same video layout as the main project. Training checkpoints live under `SVD_ROOT/checkpoints`; eval videos under `SVD_ROOT/data` (see repo root README and `configs/paths.example.yaml`).

## Layout

- `gpt_physics_analyzer.py` — main analyzer
- `gpt_api_third_party.py` — third-party chat completions
- `vjepa_data_dirs.json` — per-version dataset roots (edit `REPLACE_ME` or your paths)
- `run_gpt_physics_tests.py` — batch driver
- `API_KEYS.md` — required environment variables

## Setup

```bash
pip install opencv-python numpy
```

Configure API keys as described in `API_KEYS.md` (e.g. copy `secrets/gpt_key.json.example` to `secrets/gpt_key.json` or set `GPT_KEY_JSON`).

## Quick start

```bash
cd test_mllms_from_api
python run_gpt_physics_tests.py
```

Result text files use English section headers (`Sample`, `Comparison`, `GPT full response`, …) so downstream parsers expect the same labels.

## Versions in `vjepa_data_dirs.json`

- **v119_vjepa** — viscosity absolute  
- **v110_vjepa** — viscosity relative  
- **v120_vjepa** — friction absolute  
- **v121_vjepa** — friction relative  
- **v124_vjepa** — elasticity absolute  
- **v125_vjepa** — elasticity relative  

Each version has `test1`, `test2`, `test3` folders under the paths you set.

## Paper Table 1 gate

Elasticity Test-3 numbers for trained models (E / B) are checked with:

```bash
python scripts/verify_paper_elasticity_test3.py --svd-root "$SVD_ROOT"
```

Targets are in `docs/paper_table1_reference.md`.
