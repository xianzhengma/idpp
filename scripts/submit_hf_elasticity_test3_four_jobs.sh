#!/usr/bin/env bash
# 在登录节点：先 stage + resolve 一次，再 **并行提交 4 个** low-prio-gpu 作业（E abs、E rel、B abs、B rel）。
#
# 用法:
#   export SVD_ROOT=/work/guanqi/proj5/Inferring_dynamic_physics/svd_hf_elasticity_t3
#   export SKIP_HF_DOWNLOAD=1   # 可选；与单作业脚本语义一致
#   bash /work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/submit_hf_elasticity_test3_four_jobs.sh
#
# 四个作业对应论文/配置里的 elasticity test3：
#   DynamicCrafter（表 E / 仓库里常称 newdata e729_nd_*）v124 absolute、v125 relative
#   V-JEPA-2（表 B / mirror）v124 absolute、v125 relative

set -eu
set -o pipefail

PROJ="/work/guanqi/proj5/Inferring_dynamic_physics"
REPO="${PROJ}/SVD_public"
GEN="${REPO}/configs/reproduce/generated"
SCR="${REPO}/scripts"
ONE="${SCR}/sbatch_hf_elasticity_t3_one_val.sh"

SVD_ROOT="${SVD_ROOT:-${PROJ}/svd_hf_elasticity_t3}"
export SVD_ROOT
mkdir -p "${SVD_ROOT}" "${SCR}/logs_elasticity_repro"
chmod +x "${ONE}"

if [[ "${SKIP_HF_DOWNLOAD:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] snapshot_download (needs HF_TOKEN if private)"
  python3 -u "${SCR}/download_hf_assets.py" --svd-root "${SVD_ROOT}"
else
  echo "[$(date -Iseconds)] SKIP_HF_DOWNLOAD=1"
fi

echo "[$(date -Iseconds)] stage + resolve (once)"
python3 -u "${SCR}/stage_elasticity_test3_from_hf.py" \
  --svd-root "${SVD_ROOT}" \
  --data-hf "${SVD_ROOT}/data_hf"
python3 -u "${SCR}/resolve_eval_configs.py" --svd-root "${SVD_ROOT}"

submit_one() {
  local job_name="$1"
  local cfg_path="$2"
  sbatch -J "${job_name}" \
    --export=ALL,SVD_ROOT="${SVD_ROOT}",SKIP_HF_DOWNLOAD="${SKIP_HF_DOWNLOAD:-0}",HF_ELASTICITY_VAL_CFG="${cfg_path}" \
    "${ONE}"
}

echo "[$(date -Iseconds)] submitting 4 Slurm jobs (low-prio-gpu, GPU constraint a40|a6000|rtx8000, one val each)"
submit_one "el_E124_abs" "${GEN}/config_e729_nd_v124_test3_el_abs.yaml"
submit_one "el_E125_rel" "${GEN}/config_e729_nd_v125_test3_el_rel.yaml"
submit_one "el_B124_abs" "${GEN}/config_v124_vjepa_test3.yaml"
submit_one "el_B125_rel" "${GEN}/config_v125_vjepa_test3.yaml"

echo "[$(date -Iseconds)] done. After all finish, score with:"
echo "  cd ${REPO} && export SVD_ROOT=${SVD_ROOT} SVD_E_NEWDATA_RESULTS=${SVD_ROOT}/results/e729_newdata_20260327"
echo "  python3 calculate_e729_newdata_dc_scores.py | grep elasticity"
