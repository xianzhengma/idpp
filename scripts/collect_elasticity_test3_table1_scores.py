#!/usr/bin/env python3
"""
作用：在给定 SVD_ROOT 下汇总 Table 1 所需的四条 elasticity test3 分数（E124 abs、E125 rel、B124 abs、B125 rel）。

输入：命令行 `--rows` 为多组 `TAG=SVD_ROOT`（空格分隔）。
输出：打印 TSV 表头与各行分数；失败时跳过该行并打印警告。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    from calculate_e729_newdata_dc_scores import (
        load_results,
        metric_elasticity_absolute,
        metric_elasticity_relative,
    )
except ImportError as exc:
    raise SystemExit(f'Import failed: {exc}') from exc


def _one_metric(folder: str, kind: str) -> Tuple[Optional[float], int]:
    """
    作用：从单目录加载预测并计算弹性度量。

    输入：folder 为 JSON 目录；kind 为 absolute 或 relative。
    输出：(标量分数或 None, 样本条数)。
    """
    try:
        preds, gts = load_results(folder)
        if preds is None:
            return None, 0
        n = len(preds)
        if kind == 'absolute':
            return float(metric_elasticity_absolute(preds, gts)), n
        return float(metric_elasticity_relative(preds, gts)), n
    except (ValueError, TypeError, OSError, FloatingPointError):
        return None, 0


def _score_svd_root(tag: str, root: str) -> None:
    """
    作用：打印单行 TSV（tag、四分数、四样本数）。

    输入：tag 为实验标签；root 为 SVD_ROOT 绝对路径。
    输出：写入 stdout。
    """
    nd = os.path.join(root, 'results', 'e729_newdata_20260327')
    e_abs = os.path.join(nd, 'e729_nd_v124_test3_el_abs')
    e_rel = os.path.join(nd, 'e729_nd_v125_test3_el_rel')
    b_abs = os.path.join(root, 'results', 'mirror_eval', 'v124_vjepa_test3')
    b_rel = os.path.join(root, 'results', 'mirror_eval', 'v125_vjepa_test3')
    se, ne = _one_metric(e_abs, 'absolute')
    sr, nr = _one_metric(e_rel, 'relative')
    sb, nb = _one_metric(b_abs, 'absolute')
    sbr, nbr = _one_metric(b_rel, 'relative')
    vals = [se, sr, sb, sbr]
    nums = [ne, nr, nb, nbr]
    def fmt(v: Optional[float]) -> str:
        if v is None:
            return 'NA'
        return f'{v:.6f}'
    line = '\t'.join([tag, root] + [fmt(v) for v in vals] + [str(n) for n in nums])
    print(line)


def main() -> int:
    """作用：CLI 入口。输入：argv。输出：退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rows',
        nargs='+',
        required=True,
        help="e.g. M00=/path/to/svd_hf_elasticity_t3 M11=/path/to/svd_ablate_M11",
    )
    args = parser.parse_args()
    try:
        print('tag\tSVD_ROOT\tE_abs_Pearson\tE_rel_ROCAUC\tB_abs_Pearson\tB_rel_ROCAUC\tn_E_abs\tn_E_rel\tn_B_abs\tn_B_rel')
        for item in args.rows:
            if '=' not in item:
                print(f'skip bad row (use TAG=PATH): {item}', file=sys.stderr)
                continue
            tag, root = item.split('=', 1)
            root = os.path.abspath(os.path.expanduser(root))
            _score_svd_root(tag.strip(), root)
        return 0
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
