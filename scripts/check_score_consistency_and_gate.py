#!/usr/bin/env python3
"""
模块作用：对比 SVD_public 与 SVD 的 elasticity test3 算分结果是否一致，并执行 paper gate。

输入：--svd-root、--atol。
输出：打印一致性结果与 gate 结果；不通过时返回非零退出码。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from types import ModuleType
from typing import Tuple


def _load_module(module_path: str, module_name: str) -> ModuleType:
    """作用：按路径加载 Python 模块。输入：文件路径与模块名。输出：模块对象。"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_pair(module: ModuleType, folder: str, kind: str) -> float:
    """作用：用目标模块计算单目录分数。输入：模块、目录、任务类型。输出：浮点分数。"""
    preds, gts = module.load_results(folder)
    if preds is None:
        raise RuntimeError(f"No valid JSON in {folder}")
    if kind == "absolute":
        return float(module.metric_elasticity_absolute(preds, gts))
    if kind == "relative":
        return float(module.metric_elasticity_relative(preds, gts))
    raise RuntimeError(f"Unsupported kind: {kind}")


def _assert_close(name: str, left: float, right: float, atol: float) -> None:
    """作用：检查两个分数是否在容差内。输入：名称、两个分数、容差。输出：无。"""
    delta = abs(left - right)
    print(f"[consistency] {name}: public={left:.6f} internal={right:.6f} delta={delta:.6f}")
    if delta > atol:
        raise RuntimeError(f"{name} mismatch exceeds atol={atol}")


def _run_gate(verify_module: ModuleType, svd_root: str, atol: float) -> int:
    """作用：调用 verify 脚本主逻辑。输入：verify 模块、SVD_ROOT、容差。输出：退出码。"""
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "verify_paper_elasticity_test3.py",
            "--svd-root",
            svd_root,
            "--atol",
            str(atol),
        ]
        verify_module.main()
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = argv_backup


def main() -> int:
    """作用：脚本入口。输入：命令行参数。输出：退出码。"""
    parser = argparse.ArgumentParser(description="Check score consistency between SVD_public and SVD.")
    parser.add_argument("--svd-root", required=True)
    parser.add_argument("--atol", type=float, default=1e-8, help="Tolerance for public vs internal score consistency.")
    parser.add_argument("--gate-atol", type=float, default=0.03, help="Tolerance used by paper gate.")
    args = parser.parse_args()

    public_score_path = "/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/calculate_e729_newdata_dc_scores.py"
    internal_score_path = "/work/guanqi/proj5/Inferring_dynamic_physics/SVD/calculate_e729_newdata_dc_scores.py"
    gate_path = "/work/guanqi/proj5/Inferring_dynamic_physics/SVD_public/scripts/verify_paper_elasticity_test3.py"

    public_mod = _load_module(public_score_path, "score_public")
    internal_mod = _load_module(internal_score_path, "score_internal")
    gate_mod = _load_module(gate_path, "verify_gate")

    e_abs_dir = os.path.join(args.svd_root, "results", "e729_newdata_20260327", "e729_nd_v124_test3_el_abs")
    e_rel_dir = os.path.join(args.svd_root, "results", "e729_newdata_20260327", "e729_nd_v125_test3_el_rel")
    b_abs_dir = os.path.join(args.svd_root, "results", "mirror_eval", "v124_vjepa_test3")
    b_rel_dir = os.path.join(args.svd_root, "results", "mirror_eval", "v125_vjepa_test3")

    for name, folder, kind in (
        ("E_abs_t3", e_abs_dir, "absolute"),
        ("E_rel_t3", e_rel_dir, "relative"),
        ("B_abs_t3", b_abs_dir, "absolute"),
        ("B_rel_t3", b_rel_dir, "relative"),
    ):
        public_score = _score_pair(public_mod, folder, kind)
        internal_score = _score_pair(internal_mod, folder, kind)
        _assert_close(name, public_score, internal_score, args.atol)

    gate_code = _run_gate(gate_mod, args.svd_root, args.gate_atol)
    if gate_code != 0:
        raise RuntimeError(f"Paper gate failed with exit code {gate_code}")
    print("[gate] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
