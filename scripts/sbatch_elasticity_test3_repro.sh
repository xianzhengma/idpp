#!/usr/bin/env bash
# 在 low-prio-gpu 分区跑 elasticity test3 四条公开评测（E v124/v125 + B v124/v125），单作业内串行。
# 前置：已设置 SVD_ROOT（含 checkpoints/）、并已运行 stage_elasticity_test3_from_hf.py 与 resolve_eval_configs.py。
#
# 提交示例:
#   export SVD_ROOT=/path/to/asset_root
#   sbatch /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_elasticity_test3_repro.sh

#SBATCH -p low-prio-gpu
# 仅 A40 / A6000 / Quadro RTX 8000。
#SBATCH --constraint=a40|a6000|rtx8000
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -J el_t3_repro
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/%x_%j.err

set -eu
set -o pipefail
export PYTHONUNBUFFERED=1

REPO="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public"
MAIN="${REPO}/main"
GEN="${REPO}/configs/reproduce/generated"
LOGROOT="${REPO}/scripts/logs_elasticity_repro"
mkdir -p "${LOGROOT}"

if [[ -z "${SVD_ROOT:-}" ]]; then
  echo "ERROR: export SVD_ROOT=... before sbatch (and use sbatch --export=ALL,SVD_ROOT)" >&2
  exit 1
fi

run_one() {
  local name="$1"
  local cfg="$2"
  echo "[$(date -Iseconds)] ${name} start"
  cd "${MAIN}"
  python3 -u trainer.py --base "${cfg}" --val --gpus 1 --precision 16
  echo "[$(date -Iseconds)] ${name} done"
}

run_one "E_v124_abs" "${GEN}/config_e729_nd_v124_test3_el_abs.yaml"
run_one "E_v125_rel" "${GEN}/config_e729_nd_v125_test3_el_rel.yaml"
run_one "B_v124_abs" "${GEN}/config_v124_vjepa_test3.yaml"
run_one "B_v125_rel" "${GEN}/config_v125_vjepa_test3.yaml"

echo "[$(date -Iseconds)] All four runs finished. Next: python3 scripts/verify_paper_elasticity_test3.py --svd-root \"\${SVD_ROOT}\""
