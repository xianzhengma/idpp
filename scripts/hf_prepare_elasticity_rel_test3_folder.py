#!/usr/bin/env python3
"""
模块作用：为 Hugging Face 解压后的 elasticity **relative** test3 目录生成
`gt_restitution1.json`（basename → 由文件名解析的 restitution）以及
`elasticity_relative_test3.json`（每个 video_id 组取前两条 `_rgb` 组成一对），
供 WebVidv3ElasticityTest + 显式 pair list 使用（不依赖 genesis 2.16 命名）。

输入：CLI `--data-dir`、`--list-out`、`--write-gt-in-dir`（通常与 data-dir 相同）。
输出：写入两个 JSON；失败时非零退出。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    parser = argparse.ArgumentParser(description="Prepare gt + pair list for HF relative elasticity test3 folder.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Unpacked video root (contains *_rgb.mp4).")
    parser.add_argument(
        "--list-out",
        type=Path,
        required=True,
        help="Write elasticity_relative_test3-style pair list JSON here.",
    )
    parser.add_argument(
        "--write-gt-in-dir",
        type=Path,
        default=None,
        help="Write gt_restitution1.json under this dir (default: same as --data-dir).",
    )
    return parser.parse_args()


def _restitution_from_basename(basename: str) -> float:
    """作用：从仿真命名 basename 中解析 objRestitution 浮点值。输入：basename。输出：float。"""
    match = re.search(r"objRestitution_([0-9]+\.[0-9eE+-]+)", basename)
    if not match:
        raise ValueError(f"Cannot parse restitution from basename: {basename}")
    return float(match.group(1))


def _group_rgb_basenames(data_dir: Path) -> dict[int, list[str]]:
    """
    作用：按 WebVidv3 规则将 `_rgb.mp4` 按 video_id 分组。
    输入：数据根目录。
    输出：video_id → basename 列表。
    """
    groups: dict[int, list[str]] = {}
    try:
        names = sorted(p.name for p in data_dir.iterdir() if p.is_file() and p.name.endswith("_rgb.mp4"))
    except OSError as exc:
        raise RuntimeError(f"Cannot list {data_dir}: {exc}") from exc
    for name in names:
        basename = name[: -len("_rgb.mp4")]
        parts = basename.split("_")
        if len(parts) < 3 or parts[0] != "video":
            continue
        try:
            vid = int(parts[1])
        except ValueError:
            continue
        groups.setdefault(vid, []).append(basename)
    return groups


def _pairs_from_groups(groups: dict[int, list[str]]) -> list[list[str]]:
    """作用：每组至少两条时取排序后前两条组成一对。输入：分组。输出：pair 列表。"""
    pairs: list[list[str]] = []
    for basenames in groups.values():
        if len(basenames) < 2:
            continue
        basenames = sorted(basenames)
        pairs.append([basenames[0], basenames[1]])
    return pairs


def _write_gt_file(gt_path: Path, basenames: list[str]) -> None:
    """作用：写出 gt_restitution1.json。输入：路径、所有 basename。输出：无。"""
    data: dict[str, float] = {}
    for bn in basenames:
        try:
            data[bn] = _restitution_from_basename(bn)
        except ValueError:
            continue
    try:
        gt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {gt_path}: {exc}") from exc


def main() -> int:
    """作用：入口。输出：退出码。"""
    args = parse_args()
    data_dir = args.data_dir.resolve()
    list_out = args.list_out.resolve()
    gt_dir = (args.write_gt_in_dir or args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"ERROR: not a directory: {data_dir}", file=sys.stderr)
        return 1
    try:
        groups = _group_rgb_basenames(data_dir)
        pairs = _pairs_from_groups(groups)
        if not pairs:
            raise RuntimeError("No valid pairs (need video_* groups with >=2 _rgb clips).")
        all_bn = sorted({b for pair in pairs for b in pair})
        _write_gt_file(gt_dir / "gt_restitution1.json", all_bn)
        list_out.parent.mkdir(parents=True, exist_ok=True)
        list_out.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        print(f"OK: wrote {gt_dir / 'gt_restitution1.json'} ({len(all_bn)} keys) and {list_out} ({len(pairs)} pairs)")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
