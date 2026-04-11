#!/usr/bin/env bash
# Run four Elasticity Test-3 validation jobs in sequence (GPU required) after SVD_ROOT is set and YAMLs are generated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SVD_ROOT="${SVD_ROOT:-$ROOT/asset_root}"
python "$ROOT/scripts/resolve_eval_configs.py" --svd-root "$SVD_ROOT"
cd "$ROOT/main"
for cfg in \
  config_e729_nd_v124_test3_el_abs.yaml \
  config_e729_nd_v125_test3_el_rel.yaml \
  config_v124_vjepa_test3.yaml \
  config_v125_vjepa_test3.yaml
do
  echo "=== $cfg ==="
  python trainer.py --base "../configs/reproduce/generated/$cfg" --val --gpus 1 --precision 16
done
python "$ROOT/scripts/verify_paper_elasticity_test3.py" --svd-root "$SVD_ROOT"
