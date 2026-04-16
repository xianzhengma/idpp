#!/usr/bin/env bash
# 单个 compute 作业：上传 DynamiCrafter 7.29（E 轨）9 个 .ckpt 到 IDPP-Checkpoints 的 E_7.29/。
#
#   export HF_TOKEN=hf_...
#   unset CKPT_UPLOAD_LIMIT
#   sbatch --export=ALL,HF_TOKEN \
#     /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_upload_hf_dynamicrafter_729.sh
#
# 可选: CKPT_UPLOAD_LIMIT=N, UPLOAD_SORT_SMALL_FIRST=1, UPLOAD_DC_SKIP_SWEEP=1（跳过 v120 sweep）

#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -J hf_dc729_up
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public"
SCRIPT_DIR="${REPO_ROOT}/scripts"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: need HF_TOKEN or HUGGINGFACE_HUB_TOKEN" >&2
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
if [[ "${UPLOAD_DC_SKIP_SWEEP:-0}" == "1" ]]; then
  EXTRA+=(--skip-sweep)
fi

cd "${REPO_ROOT}"
echo "[$(date -Iseconds)] DynamiCrafter 7.29 upload on $(hostname)"
exec python3 -u "${SCRIPT_DIR}/upload_hf_dynamicrafter_729_checkpoints.py" "${EXTRA[@]}"
