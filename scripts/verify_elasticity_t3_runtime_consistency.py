#!/usr/bin/env python3
"""
模块作用：自动核验 elasticity test3 四条作业的配置注入与运行日志是否一致。

输入：SVD_ROOT、日志目录、每条作业对应 out/err 日志名（可选，默认使用当前已验证批次）。
输出：在终端打印逐项核验结果；若存在不一致，返回非零退出码。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class JobSpec:
    """作用：描述单条核验任务。输入：配置路径和期望字段。输出：不可变对象。"""

    key: str
    cfg_path: str
    out_log: str
    err_log: str
    expected_ckpt_suffix: str
    expected_dataset: str


def _read_text(path: str) -> str:
    """作用：读取文本文件内容。输入：路径。输出：字符串。"""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _extract_first(pattern: str, text: str) -> str:
    """作用：按正则提取首个匹配组。输入：pattern 与文本。输出：匹配字符串。"""
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_cfg_fields(cfg_text: str) -> Tuple[str, str, str]:
    """作用：从 YAML 文本提取 checkpoint/data_dir/video_list_json。输入：yaml 文本。输出：三元组。"""
    ckpt = _extract_first(r"^\s*pretrained_checkpoint:\s*(.+)$", cfg_text)
    data_dir = _extract_first(r"^\s*data_dir:\s*(.+)$", cfg_text)
    list_json = _extract_first(r"^\s*video_list_json:\s*(.+)$", cfg_text)
    return ckpt, data_dir, list_json


def _build_default_specs(svd_root: str, log_dir: str) -> Dict[str, JobSpec]:
    """作用：构建默认四条核验目标。输入：SVD_ROOT 与日志目录。输出：key->JobSpec。"""
    cfg_root = "/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/configs/reproduce/generated"
    return {
        "e124_abs": JobSpec(
            key="e124_abs",
            cfg_path=os.path.join(cfg_root, "config_e729_nd_v124_test3_el_abs.yaml"),
            out_log=os.path.join(log_dir, "hf_oneval_el_E124_abs_606613.out"),
            err_log=os.path.join(log_dir, "hf_oneval_el_E124_abs_606613.err"),
            expected_ckpt_suffix="training_1024_v124_7.29/checkpoints/epoch=4-step=2625.ckpt",
            expected_dataset="WebVidv2Viscosity",
        ),
        "e125_rel": JobSpec(
            key="e125_rel",
            cfg_path=os.path.join(cfg_root, "config_e729_nd_v125_test3_el_rel.yaml"),
            out_log=os.path.join(log_dir, "hf_oneval_el_E125_rel_dbg4_606695.out"),
            err_log=os.path.join(log_dir, "hf_oneval_el_E125_rel_dbg4_606695.err"),
            expected_ckpt_suffix="training_1024_v125_7.29/checkpoints/epoch=3-step=500.ckpt",
            expected_dataset="WebVidv3ElasticityTest",
        ),
        "b124_abs": JobSpec(
            key="b124_abs",
            cfg_path=os.path.join(cfg_root, "config_v124_vjepa_test3.yaml"),
            out_log=os.path.join(log_dir, "hf_oneval_el_B124_abs_606615.out"),
            err_log=os.path.join(log_dir, "hf_oneval_el_B124_abs_606615.err"),
            expected_ckpt_suffix="training_1024_v124_vjepa_norm/checkpoints/epoch=21-step=8375.ckpt",
            expected_dataset="WebVidv2Viscosity",
        ),
        "b125_rel": JobSpec(
            key="b125_rel",
            cfg_path=os.path.join(cfg_root, "config_v125_vjepa_test3.yaml"),
            out_log=os.path.join(log_dir, "hf_oneval_el_B125_rel_fix2_606698.out"),
            err_log=os.path.join(log_dir, "hf_oneval_el_B125_rel_fix2_606698.err"),
            expected_ckpt_suffix="training_1024_v125_vjepa_norm/checkpoints/epoch=126-step=15875.ckpt",
            expected_dataset="WebVidv3ElasticityTest",
        ),
    }


def _validate_one(spec: JobSpec, svd_root: str) -> Tuple[bool, str]:
    """作用：核验单条作业一致性。输入：JobSpec 与 SVD_ROOT。输出：(是否通过, 详情)。"""
    cfg_text = _read_text(spec.cfg_path)
    out_text = _read_text(spec.out_log)
    err_text = _read_text(spec.err_log)

    ckpt_cfg, data_dir, list_json = _parse_cfg_fields(cfg_text)
    ckpt_log = _extract_first(r"Loaded weights from pretrained checkpoint:\s*(.+)$", err_text)
    dataset_line = _extract_first(r"^validation,\s*([^,]+),\s*(\d+)$", err_text)
    cfg_in_out = _extract_first(r"cfg=(\S+)\s+SVD_ROOT=", out_text)

    expected_data_dir = os.path.join(svd_root, "data", "elasticity_relative_test3") if "rel" in spec.key else os.path.join(svd_root, "data", "elasticity_absolute_test3")
    expected_list = os.path.join(svd_root, "data", "lists", "elasticity_relative_test3.json") if "rel" in spec.key else os.path.join(svd_root, "data", "lists", "elasticity_absolute_test3.json")

    checks = [
        ckpt_cfg.endswith(spec.expected_ckpt_suffix),
        ckpt_log.endswith(spec.expected_ckpt_suffix),
        data_dir == expected_data_dir,
        list_json == expected_list,
        cfg_in_out == spec.cfg_path,
        spec.expected_dataset in dataset_line,
    ]
    ok = all(checks)
    msg = (
        f"[{spec.key}] ok={ok} "
        f"dataset={dataset_line} "
        f"cfg_ckpt={ckpt_cfg} "
        f"log_ckpt={ckpt_log} "
        f"data_dir={data_dir} "
        f"list_json={list_json}"
    )
    return ok, msg


def main() -> int:
    """作用：脚本入口。输入：命令行参数。输出：退出码。"""
    parser = argparse.ArgumentParser(description="Verify elasticity test3 runtime consistency from configs + logs.")
    parser.add_argument("--svd-root", required=True, help="SVD_ROOT used by eval jobs.")
    parser.add_argument(
        "--log-dir",
        default="/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/logs_elasticity_repro",
        help="Directory containing hf_oneval logs.",
    )
    args = parser.parse_args()

    specs = _build_default_specs(args.svd_root, args.log_dir)
    all_ok = True
    for key in ("e124_abs", "e125_rel", "b124_abs", "b125_rel"):
        ok, msg = _validate_one(specs[key], args.svd_root)
        print(msg)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
