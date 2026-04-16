#!/usr/bin/env bash
# 在 compute 节点对 dataset/<lane>/*.zip 做公开清洗（默认：仅删 _seg / red_circle* mp4）并原地重打包。
# 解压临时目录放在 /work 下，避免 /tmp 或登录节点根分区写满。
#
# 提交（登录节点）:
#   sbatch /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_sanitize_repack_idpp_zips.sh
#
# 每成功清洗一个 zip 就立刻启动 HF 上传（与后续 sanitize 并行；作业结束前会等待全部上传结束）:
#   export HF_TOKEN=hf_...
#   sbatch --export=ALL,HF_TOKEN,HF_UPLOAD_EACH=1 /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_sanitize_repack_idpp_zips.sh
#
# 可选（同一作业内紧接着跑 Oracle 全量对齐，需 Genesis JSON 已挂载）:
#   sbatch --export=ALL,SANITIZE_ORACLE_AFTER=1 sbatch_sanitize_repack_idpp_zips.sh
#   或: export ORACLE_JSON_DIR=/work/guanqi/Genesis/oracle_py_files
#
# 日志: SVD_public/scripts/logs_dataset_sanitize/hf_san_idpp_%j.out

#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -J san_idpp_z
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_dataset_sanitize/%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_dataset_sanitize/%x_%j.err

set -eu
set -o pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public"
SCRIPT_DIR="${REPO_ROOT}/scripts"
PROJ="/work/guanqi/proj5/Inferring_dynamic_physics"
export TMPDIR="${TMPDIR:-${PROJ}/.tmp_idpp_sanitize}"
mkdir -p "${TMPDIR}"
mkdir -p "${SCRIPT_DIR}/logs_dataset_sanitize"

echo "[$(date -Iseconds)] sanitize+repack on $(hostname) TMPDIR=${TMPDIR}"

cd "${REPO_ROOT}"
LOG="${SCRIPT_DIR}/logs_dataset_sanitize/san_pass1_${SLURM_JOB_ID:-local}.log"

PASS1_EXTRA=()
if [[ "${HF_UPLOAD_EACH:-0}" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: HF_UPLOAD_EACH=1 requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN (use: sbatch --export=ALL,HF_TOKEN,HF_UPLOAD_EACH=1 ...)" >&2
    exit 1
  fi
  PASS1_EXTRA+=(--hf-upload-each)
fi

python3 -u "${SCRIPT_DIR}/sanitize_repack_idpp_zips.py" \
  --dataset-root "${PROJ}/dataset" \
  --in-place \
  --apply \
  --sensitive-mp4-only \
  "${PASS1_EXTRA[@]}" 2>&1 | tee "${LOG}"

ORACLE_ROOT="${ORACLE_JSON_DIR:-/work/guanqi/Genesis/oracle_py_files}"
shopt -s nullglob
_oracle_jsons=( "${ORACLE_ROOT}"/*.json )
shopt -u nullglob
if [[ "${SANITIZE_ORACLE_AFTER:-0}" == "1" ]] && [[ -d "${ORACLE_ROOT}" ]] && [[ ${#_oracle_jsons[@]} -gt 0 ]]; then
  echo "[$(date -Iseconds)] Pass 2: Oracle rgb prune + count check (ORACLE_JSON_DIR=${ORACLE_ROOT})"
  export ORACLE_JSON_DIR="${ORACLE_ROOT}"
  LOG2="${SCRIPT_DIR}/logs_dataset_sanitize/san_pass2_oracle_${SLURM_JOB_ID:-local}.log"
  python3 -u "${SCRIPT_DIR}/sanitize_repack_idpp_zips.py" \
    --dataset-root "${PROJ}/dataset" \
    --in-place \
    --apply \
    "${PASS1_EXTRA[@]}" 2>&1 | tee "${LOG2}"
else
  echo "[$(date -Iseconds)] Skip Oracle pass 2 (set SANITIZE_ORACLE_AFTER=1 and valid ORACLE_JSON_DIR)."
fi

echo "[$(date -Iseconds)] Done."
