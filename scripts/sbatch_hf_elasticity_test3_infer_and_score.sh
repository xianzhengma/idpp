#!/usr/bin/env bash
# **单条 Slurm 作业**内顺序跑四次 val + 算分（一条 job 占一张 GPU 很久）。
# 若要 **四个独立作业**（E abs / E rel / B abs / B rel）、且用 **low-prio-gpu**，请用登录节点执行:
#   bash scripts/submit_hf_elasticity_test3_four_jobs.sh
#
# 本脚本保留兼容：可选 snapshot 下载 → stage → resolve → 四次 val → calculate。
#
# 登录节点提交示例:
#   export SVD_ROOT=/work/guanqi/proj5/Inferring_dynamic_physics/svd_hf_elasticity_t3
#   export HF_TOKEN=hf_...   # 若需从 Hub 拉取
#   sbatch --export=ALL,SVD_ROOT,HF_TOKEN /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/sbatch_hf_elasticity_test3_infer_and_score.sh
#
# 已有人工放置 checkpoints/data_hf 时:
#   export SKIP_HF_DOWNLOAD=1
#   sbatch --export=ALL,SVD_ROOT,SKIP_HF_DOWNLOAD /path/to/sbatch_hf_elasticity_test3_infer_and_score.sh

#SBATCH -p low-prio-gpu
# 仅 A40 / A6000 / Quadro RTX 8000（见 AVAIL_FEATURES: a40, a6000, rtx8000）。
#SBATCH --constraint=a40|a6000|rtx8000
#SBATCH -c 4
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -J hf_el_t3
#SBATCH -o /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/hf_infer_%x_%j.out
#SBATCH -e /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro/hf_infer_%x_%j.err

set -eu
set -o pipefail
export PYTHONUNBUFFERED=1

PROJ="/work/guanqi/proj5/Inferring_dynamic_physics"
REPO="${PROJ}/SVD_public"
MAIN="${REPO}/main"
GEN="${REPO}/configs/reproduce/generated"
SCR="${REPO}/scripts"
LOGROOT="${SCR}/logs_elasticity_repro"
export TMPDIR="${TMPDIR:-${PROJ}/.tmp_hf_elasticity_t3}"
mkdir -p "${TMPDIR}" "${LOGROOT}"

SVD_ROOT="${SVD_ROOT:-${PROJ}/svd_hf_elasticity_t3}"
export SVD_ROOT
mkdir -p "${SVD_ROOT}"

if [[ -f "/users/guanqi/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/users/guanqi/anaconda3/etc/profile.d/conda.sh"
  conda activate dynamicrafter_vjepa
fi

export PYTHONPATH="${MAIN}:${REPO}:${PYTHONPATH:-}"
export HF_SVD_CHECKPOINTS_REPO="${HF_SVD_CHECKPOINTS_REPO:-Oliver-Ma/IDPP-Checkpoints}"
export HF_SVD_DATA_REPO="${HF_SVD_DATA_REPO:-Oliver-Ma/IDPP-Data}"

echo "[$(date -Iseconds)] SVD_ROOT=${SVD_ROOT} HOST=$(hostname)"

if [[ "${SKIP_HF_DOWNLOAD:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] snapshot_download checkpoints + dataset"
  python3 -u "${SCR}/download_hf_assets.py" --svd-root "${SVD_ROOT}"
else
  echo "[$(date -Iseconds)] SKIP_HF_DOWNLOAD=1 — using existing ${SVD_ROOT}/checkpoints and ${SVD_ROOT}/data_hf"
fi

echo "[$(date -Iseconds)] stage test3 from Hub zips (merged abs + 2.16 rel; lists from repo)"
python3 -u "${SCR}/stage_elasticity_test3_from_hf.py" \
  --svd-root "${SVD_ROOT}" \
  --data-hf "${SVD_ROOT}/data_hf"

echo "[$(date -Iseconds)] resolve eval yaml"
python3 -u "${SCR}/resolve_eval_configs.py" --svd-root "${SVD_ROOT}"

run_val() {
  local tag="$1"
  local cfg="$2"
  echo "[$(date -Iseconds)] val ${tag}"
  cd "${MAIN}"
  python3 -u trainer.py --base "${cfg}" --val --gpus 1 --precision 16
}

run_val "E_v124_abs" "${GEN}/config_e729_nd_v124_test3_el_abs.yaml"
run_val "E_v125_rel" "${GEN}/config_e729_nd_v125_test3_el_rel.yaml"
run_val "B_v124_abs" "${GEN}/config_v124_vjepa_test3.yaml"
run_val "B_v125_rel" "${GEN}/config_v125_vjepa_test3.yaml"

echo "[$(date -Iseconds)] score (E newdata table; elasticity test3 rows)"
cd "${REPO}"
export SVD_E_NEWDATA_RESULTS="${SVD_ROOT}/results/e729_newdata_20260327"
python3 -u calculate_e729_newdata_dc_scores.py | tee "${LOGROOT}/scores_${SLURM_JOB_ID:-local}.txt"
grep -E '^\| v124 \| elasticity \| absolute \| test3 \|' "${LOGROOT}/scores_${SLURM_JOB_ID:-local}.txt" || true
grep -E '^\| v125 \| elasticity \| relative \| test3 \|' "${LOGROOT}/scores_${SLURM_JOB_ID:-local}.txt" || true

echo "[$(date -Iseconds)] done. JSON under: ${SVD_ROOT}/results/e729_newdata_20260327/ and ${SVD_ROOT}/results/mirror_eval/"
