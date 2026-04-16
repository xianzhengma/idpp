#!/usr/bin/env bash
# 在 compute 节点批量上传 IDPP-Data（video_zips/*.zip）。不要在登录节点跑大包上传。
#
# 提交（登录节点执行；Token 仅出现在当前 shell，勿写入仓库）:
#   export HF_TOKEN=hf_...
#   unset UPLOAD_LIMIT   # 若曾用 UPLOAD_LIMIT 做 smoke，务必清掉，否则 --export=ALL 会把 3 带进作业
#   export UPLOAD_SORT_SMALL_FIRST=1   # 可选
#   sbatch --export=ALL,HF_TOKEN,UPLOAD_SORT_SMALL_FIRST /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_upload_idpp_hub_zips.sh
#
# 可选覆盖:
#   sbatch --export=ALL,UPLOAD_LIMIT=3,UPLOAD_SORT_SMALL_FIRST=1 sbatch_upload_idpp_hub_zips.sh
#   UPLOAD_SORT_SMALL_FIRST=1 时先传小 zip，便于确认 compute 能访问 huggingface.co

#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH -J hf_idpp_up
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_hf_upload/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Slurm 会把脚本拷到 spool，勿用 BASH_SOURCE 推路径（会变成 /var/spool/slurmd/...）。
REPO_ROOT="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public"
SCRIPT_DIR="${REPO_ROOT}/scripts"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: compute job needs HF_TOKEN or HUGGINGFACE_HUB_TOKEN in environment." >&2
  echo "Submit with: export HF_TOKEN=...; sbatch --export=ALL $0" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs_hf_upload"

EXTRA=()
if [[ -n "${UPLOAD_LIMIT:-}" ]]; then
  EXTRA+=(--limit "${UPLOAD_LIMIT}")
fi
if [[ "${UPLOAD_SORT_SMALL_FIRST:-0}" == "1" ]]; then
  EXTRA+=(--sort-by-size)
fi

cd "${REPO_ROOT}"
echo "[$(date -Iseconds)] Starting upload on $(hostname)"
exec python3 -u "${SCRIPT_DIR}/upload_idpp_hub_zips.py" --repo-id Oliver-Ma/IDPP-Data "${EXTRA[@]}"
