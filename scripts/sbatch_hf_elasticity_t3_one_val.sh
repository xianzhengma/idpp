#!/usr/bin/env bash
# 单条 elasticity test3 的 `trainer.py --val`（low-prio-gpu，一条作业 = 一个配置）。
# 需已通过 `submit_hf_elasticity_test3_four_jobs.sh` 或手工完成 stage + resolve。
#
# 环境变量（sbatch --export=ALL,...）:
#   HF_ELASTICITY_VAL_CFG  — 绝对路径，指向 generated 下的 yaml，例如
#       .../config_e729_nd_v124_test3_el_abs.yaml  (DynamicCrafter / E, absolute)
#       .../config_e729_nd_v125_test3_el_rel.yaml (DynamicCrafter / E, relative)
#       .../config_v124_vjepa_test3.yaml          (V-JEPA / B, absolute)
#       .../config_v125_vjepa_test3.yaml          (V-JEPA / B, relative)
#   SVD_ROOT — 资产根目录（与 resolve 时一致）

#SBATCH -p low-prio-gpu
# 仅调度 A40 / A6000 / Quadro RTX 8000（与 Slurm AVAIL_FEATURES 一致），排除 Blackwell 等不兼容 PyTorch 的型号。
#SBATCH --constraint=a40|a6000|rtx8000
#SBATCH -c 4
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/hf_oneval_%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/hf_oneval_%x_%j.err

set -eu
set -o pipefail
export PYTHONUNBUFFERED=1

PROJ="/work/guanqi/proj5/Inferring_dynamic_physics"
REPO="${PROJ}/SVD_public"
MAIN="${REPO}/main"
SCR="${REPO}/scripts"
export TMPDIR="${TMPDIR:-${PROJ}/.tmp_hf_elasticity_t3}"
mkdir -p "${TMPDIR}" "${SCR}/logs_elasticity_repro"

if [[ -z "${HF_ELASTICITY_VAL_CFG:-}" ]]; then
  echo "ERROR: set HF_ELASTICITY_VAL_CFG to the yaml path (use sbatch --export=ALL,HF_ELASTICITY_VAL_CFG,...)" >&2
  exit 1
fi
if [[ ! -f "${HF_ELASTICITY_VAL_CFG}" ]]; then
  echo "ERROR: HF_ELASTICITY_VAL_CFG is not a file: ${HF_ELASTICITY_VAL_CFG}" >&2
  exit 1
fi

SVD_ROOT="${SVD_ROOT:-${PROJ}/svd_hf_elasticity_t3}"
export SVD_ROOT

if [[ -f "/users/guanqi/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/users/guanqi/anaconda3/etc/profile.d/conda.sh"
  conda activate dynamicrafter_vjepa
fi

export PYTHONPATH="${MAIN}:${REPO}:${PYTHONPATH:-}"

echo "[$(date -Iseconds)] host=$(hostname) cfg=${HF_ELASTICITY_VAL_CFG} SVD_ROOT=${SVD_ROOT}"
cd "${MAIN}"
exec python3 -u trainer.py --base "${HF_ELASTICITY_VAL_CFG}" --val --gpus 1 --precision 16
