#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级说明：在已有评测 JSON 输出目录上计算 Elasticity Test-3 四格指标，并与论文 Table 1 对照。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from calculate_e729_newdata_dc_scores import (
        load_results,
        metric_elasticity_absolute,
        metric_elasticity_relative,
    )
    from svd_paths import get_results_newdata_root, get_svd_root
except ImportError as exc:
    raise SystemExit(f"导入失败: {exc}") from exc


def _score_folder(folder: str, kind: str) -> Optional[float]:
    """
    输入: folder — 含 per-sample JSON；kind — absolute 或 relative。
    输出: 标量指标或 None。
    作用: load_results + 对应 metric。
    """
    try:
        preds, gts = load_results(folder)
        if preds is None:
            return None
        if kind == "absolute":
            return float(metric_elasticity_absolute(preds, gts))
        if kind == "relative":
            return float(metric_elasticity_relative(preds, gts))
        return None
    except (ValueError, TypeError, OSError):
        return None


def main() -> None:
    """
    输入: CLI（SVD_ROOT、容差、可选各目录覆盖）。
    输出: 打印比对结果并以非零码表示未通过。
    作用: 论文 Table 1 弹性 Test-3 自检门禁。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd-root", default=None)
    parser.add_argument("--atol", type=float, default=0.03, help="与主表允许的绝对误差")
    parser.add_argument("--e-abs-dir", default=None, help="覆盖 E v124 absolute 输出目录")
    parser.add_argument("--e-rel-dir", default=None, help="覆盖 E v125 relative 输出目录")
    parser.add_argument("--b-abs-dir", default=None, help="覆盖 B v124 输出目录")
    parser.add_argument("--b-rel-dir", default=None, help="覆盖 B v125 输出目录")
    args = parser.parse_args()

    root = get_svd_root(args.svd_root)
    os.environ["SVD_ROOT"] = root
    nd_root = get_results_newdata_root()

    e_abs = args.e_abs_dir or os.path.join(nd_root, "e729_nd_v124_test3_el_abs")
    e_rel = args.e_rel_dir or os.path.join(nd_root, "e729_nd_v125_test3_el_rel")
    b_abs = args.b_abs_dir or os.path.join(root, "results", "mirror_eval", "v124_vjepa_test3")
    b_rel = args.b_rel_dir or os.path.join(root, "results", "mirror_eval", "v125_vjepa_test3")

    expected = {
        "E DynamiCrafter abs (Pearson)": (0.10, _score_folder(e_abs, "absolute")),
        "E DynamiCrafter rel (ROC-AUC)": (0.61, _score_folder(e_rel, "relative")),
        "B V-JEPA-2 abs (Pearson)": (0.37, _score_folder(b_abs, "absolute")),
        "B V-JEPA-2 rel (ROC-AUC)": (0.99, _score_folder(b_rel, "relative")),
    }

    failed = False
    for name, (paper_val, got) in expected.items():
        if got is None:
            print(f"[MISSING] {name}: 无有效 JSON（检查目录是否存在: 见上文路径）")
            failed = True
            continue
        ok = abs(got - paper_val) <= args.atol
        flag = "OK" if ok else "FAIL"
        print(f"[{flag}] {name}: paper={paper_val:.2f} got={got:.4f} (atol={args.atol})")
        if not ok:
            failed = True

    if failed:
        sys.exit(1)
    print("All checks passed (within tolerance). See docs/paper_table1_reference.md.")


if __name__ == "__main__":
    main()
