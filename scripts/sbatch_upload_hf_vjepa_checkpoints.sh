#!/usr/bin/env bash
# 在 **单个** compute 节点上传 B 轨 V-JEPA（mirror）.ckpt 到 IDPP-Checkpoints；勿在登录节点跑大包上传。
#
# 全量 B（8 个文件，含 v121 test3* scratch）:
#   export HF_TOKEN=hf_...
#   unset UPLOAD_LIMIT
#   sbatch --export=ALL,HF_TOKEN /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_upload_hf_vjepa_checkpoints.sh
#
# 仅弹性 v124 / v125（2 个文件，先探路）:
#   sbatch --export=ALL,HF_TOKEN,UPLOAD_VJEPA_ELASTICITY_ONLY=1 sbatch_upload_hf_vjepa_checkpoints.sh
#
# 可选:
#   UPLOAD_SORT_SMALL_FIRST=1   — 先传小文件
#   CKPT_UPLOAD_LIMIT=N         — 只传前 N 个（smoke；勿与生产全量混淆，提交前 unset）

#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -J hf_vjepa_ckpt
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public"
SCRIPT_DIR="${REPO_ROOT}/scripts"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: compute job needs HF_TOKEN or HUGGINGFACE_HUB_TOKEN." >&2
  echo "Submit with: export HF_TOKEN=...; sbatch --export=ALL $0" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs_hf_upload"

EXTRA=()
if [[ -n "${CKPT_UPLOAD_LIMIT:-}" ]]; then
  EXTRA+=(--limit "${CKPT_UPLOAD_LIMIT}")
fi
if [[ "${UPLOAD_SORT_SMALL_FIRST:-0}" == "1" ]]; then
  EXTRA+=(--sort-by-size)
fi
if [[ "${UPLOAD_VJEPA_ELASTICITY_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--elasticity-only)
fi

cd "${REPO_ROOT}"
echo "[$(date -Iseconds)] V-JEPA ckpt upload on $(hostname)"
exec python3 -u "${SCRIPT_DIR}/upload_hf_vjepa_checkpoints.py" "${EXTRA[@]}"
