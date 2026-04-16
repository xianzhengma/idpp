#!/usr/bin/env python3
# 模块作用：对单个已解压的 IDPP 数据目录做「公开版」清洗：删除 `_seg.mp4`、`*red_circle*.mp4` 等敏感 mp4；
# 可选按 Oracle JSON 将不在列表中的 `*_rgb.mp4` 从目录树中删除。
# 输入：CLI（--root、--oracle-json、--prune-rgb-to-oracle、--dry-run / --apply）。
# 输出：stdout 打印摘要 JSON；--apply 时执行删除。

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    parser = argparse.ArgumentParser(description="Sanitize IDPP public tree (remove seg/red mp4; optional oracle prune).")
    parser.add_argument("--root", type=Path, required=True, help="Unpacked dataset root (walked recursively)")
    parser.add_argument("--oracle-json", type=Path, default=None, help="Oracle list JSON (test1/test2)")
    parser.add_argument(
        "--prune-rgb-to-oracle",
        action="store_true",
        help="Delete *_rgb.mp4 whose basename stem is not in oracle set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not delete files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletions (mutually exclusive with dry-run for writes)",
    )
    return parser.parse_args()


def is_sensitive_public_mp4(name: str) -> bool:
    """
    作用：判断是否属于合作方不希望公开的 mp4（需删除）。
    输入：文件名（不含路径）。
    输出：True 表示应删；**始终保留** `*_rgb.mp4`。
    """
    lower = name.lower()
    if not lower.endswith(".mp4"):
        return False
    if lower.endswith("_rgb.mp4"):
        return False
    if lower.endswith("_seg.mp4"):
        return True
    if lower.endswith("red_circle2.mp4"):
        return True
    if lower.endswith("red_circle.mp4"):
        return True
    return False


def load_oracle_basenames(oracle_path: Path) -> set[str]:
    """
    作用：从 Oracle JSON 解析允许的 basename（无 `_rgb.mp4` 后缀）。
    输入：JSON 文件路径。
    输出：basename 集合。
    """
    try:
        raw = json.loads(oracle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read oracle JSON {oracle_path}: {exc}") from exc

    out: set[str] = set()

    def add_one(item: object) -> None:
        if item is None:
            return
        if isinstance(item, str):
            s = item.strip()
            if s.endswith("_rgb.mp4"):
                out.add(s[: -len("_rgb.mp4")])
            else:
                out.add(s)
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            add_one(item[0])
            if len(item) > 1:
                add_one(item[1])

    if isinstance(raw, dict):
        for key in raw.keys():
            if isinstance(key, str):
                k = key.strip()
                if k.endswith("_rgb.mp4"):
                    out.add(k[: -len("_rgb.mp4")])
                else:
                    out.add(k)
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, (list, tuple)):
                for x in entry:
                    add_one(x)
            else:
                add_one(entry)
    else:
        raise RuntimeError(f"Unsupported oracle JSON top type: {type(raw)}")

    return out


def collect_mp4_paths(root: Path) -> list[Path]:
    """作用：枚举 root 下所有 .mp4。输入：根路径。输出：Path 列表。"""
    paths: list[Path] = []
    try:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(".mp4"):
                    paths.append(Path(dirpath) / fn)
    except OSError as exc:
        raise RuntimeError(f"os.walk failed under {root}: {exc}") from exc
    return paths


def run_sanitize(
    root: Path,
    oracle_path: Path | None,
    prune_rgb: bool,
    dry_run: bool,
) -> dict:
    """
    作用：执行清洗逻辑并返回统计 dict。
    输入：根目录、oracle 路径、是否裁剪 rgb、是否 dry-run。
    输出：摘要字典。
    """
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"--root is not a directory: {root}")

    allowed: set[str] | None = None
    if prune_rgb:
        if oracle_path is None or not oracle_path.is_file():
            raise RuntimeError("--prune-rgb-to-oracle requires existing --oracle-json")
        allowed = load_oracle_basenames(oracle_path)

    removed_sensitive: list[str] = []
    removed_rgb: list[str] = []

    for path in collect_mp4_paths(root):
        name = path.name
        try:
            if is_sensitive_public_mp4(name):
                removed_sensitive.append(str(path))
                if not dry_run:
                    path.unlink()
                continue
            if prune_rgb and allowed is not None and name.endswith("_rgb.mp4"):
                stem = name[: -len("_rgb.mp4")]
                if stem not in allowed:
                    removed_rgb.append(str(path))
                    if not dry_run:
                        path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Failed processing {path}: {exc}") from exc

    rgb_left = sum(1 for p in collect_mp4_paths(root) if p.name.endswith("_rgb.mp4"))
    seg_left = sum(1 for p in collect_mp4_paths(root) if "_seg.mp4" in p.name.lower())

    return {
        "root": str(root),
        "oracle_json": str(oracle_path) if oracle_path else None,
        "prune_rgb_to_oracle": prune_rgb,
        "dry_run": dry_run,
        "removed_sensitive_mp4_count": len(removed_sensitive),
        "removed_extra_rgb_mp4_count": len(removed_rgb),
        "remaining_rgb_mp4_count": rgb_left,
        "remaining_seg_like_mp4_count": seg_left,
        "removed_sensitive_sample": removed_sensitive[:5],
        "removed_rgb_sample": removed_rgb[:5],
    }


def main() -> int:
    """作用：CLI 入口。输出：退出码。"""
    args = parse_args()
    if args.apply and args.dry_run:
        print("ERROR: use either --apply or --dry-run", file=sys.stderr)
        return 1
    dry_run = not args.apply

    try:
        report = run_sanitize(
            args.root,
            args.oracle_json,
            bool(args.prune_rgb_to_oracle),
            dry_run,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
