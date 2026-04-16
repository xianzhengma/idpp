#!/usr/bin/env python3
"""
模块作用：把 Hugging Face 快照里的 elasticity test3 zip 解压并编排为 `resolve_eval_configs.py` 期望的
`$SVD_ROOT/data/elasticity_{absolute,relative}_test3` 与 `$SVD_ROOT/data/lists/*.json` 布局。

默认 zip（与 data_global_map.md 合作者路径一致）：`7.15_new_elasticity_video_clips_merged.zip`、`2.16_elasticity_test3.zip`。
输入：CLI（--svd-root、可选 zip 路径、可选视频根覆盖）。
输出：创建/覆盖目录与符号链接；打印校验摘要；失败时非零退出。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    here = Path(__file__).resolve().parent
    repo = here.parent
    default_hf = os.environ.get("SVD_DATA_HF_ROOT", "").strip()
    parser = argparse.ArgumentParser(
        description="Stage elasticity test3 videos + lists under SVD_ROOT for public repro.",
    )
    parser.add_argument(
        "--svd-root",
        type=Path,
        required=True,
        help="Asset root (must contain checkpoints/; receives data/ subtree).",
    )
    parser.add_argument(
        "--data-hf",
        type=Path,
        default=Path(default_hf) if default_hf else None,
        help="snapshot_download root (…/video_zips/...). Default: <svd-root>/data_hf",
    )
    parser.add_argument(
        "--rel-video-root",
        type=Path,
        default=None,
        help="If set: symlink this directory as relative test3 (matches bundled elasticity_relative_test3.json; paper §3 genesis layout).",
    )
    parser.add_argument(
        "--rel-zip",
        type=Path,
        default=None,
        help="Relative test3 zip when --rel-video-root not set (default: …/2.16_elasticity_test3.zip; genesis, matches data_global_map.md).",
    )
    parser.add_argument(
        "--abs-zip",
        type=Path,
        default=None,
        help="Override absolute test3 zip (default: …/elasticity_absolute/7.15_new_elasticity_video_clips_merged.zip)",
    )
    parser.add_argument(
        "--abs-video-root",
        type=Path,
        default=None,
        help="If set: symlink this directory as absolute test3 (paper §3 uses merged tree; see docs).",
    )
    parser.add_argument(
        "--gt-rel-json",
        type=Path,
        default=None,
        help="gt_restitution1.json for WebVidv3ElasticityTest (default: bundled under repo data/elasticity_aux/).",
    )
    parser.add_argument(
        "--rel-hf-infer-prep",
        action="store_true",
        help="Legacy: when using an old Hub relative zip without gt/list, run hf_prepare_elasticity_rel_test3_folder.py. "
        "Do not use with default 2.16_elasticity_test3.zip (gt and lists already present).",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    """作用：定位 SVD_public 根目录。输入：无。输出：Path。"""
    return Path(__file__).resolve().parents[1]


def _default_hf_root(svd_root: Path, data_hf: Path | None) -> Path:
    """作用：解析 HF 快照根路径。输入：svd_root、可选覆盖。输出：Path。"""
    if data_hf is not None:
        return data_hf.resolve()
    return (svd_root / "data_hf").resolve()


def _extract_zip_to_unpack(zip_path: Path, unpack_dir: Path) -> Path:
    """
    作用：清空 unpack_dir、解压 zip，并返回「唯一顶层目录」的绝对路径。
    输入：zip 路径、解压目录（持久化于 SVD_ROOT 下，供 symlink 指向）。
    输出：解压后的单根子目录 Path。
    """
    try:
        shutil.rmtree(unpack_dir, ignore_errors=True)
        unpack_dir.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unpack_dir)
        children = [p for p in unpack_dir.iterdir() if p.name not in (".", "..")]
        if len(children) == 1 and children[0].is_dir():
            return children[0].resolve()
        raise RuntimeError(f"Expected single top-level dir in {zip_path}, got {len(children)} entries")
    except Exception:
        shutil.rmtree(unpack_dir, ignore_errors=True)
        raise


def _link_replace(target: Path, link_path: Path) -> None:
    """作用：创建或替换指向 target 的符号链接。输入：目标、链接路径。输出：无。"""
    try:
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(f"Symlink failed {link_path} -> {target}: {exc}") from exc


def _copy_file(src: Path, dst: Path) -> None:
    """作用：安全复制单文件。输入：源、目标路径。输出：无。"""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except OSError as exc:
        raise RuntimeError(f"Copy failed {src} -> {dst}: {exc}") from exc


def _count_rgb(root: Path) -> int:
    """作用：统计目录下 *_rgb.mp4 数量。输入：根目录。输出：个数。"""
    try:
        return sum(1 for p in root.iterdir() if p.is_file() and p.name.endswith("_rgb.mp4"))
    except OSError as exc:
        raise RuntimeError(f"Cannot scan {root}: {exc}") from exc


def _validate_rel_lists(data_dir: Path, list_path: Path) -> None:
    """作用：抽样校验 list JSON 中若干 basename 在 data_dir 存在。输入：视频根、json。输出：无或抛错。"""
    try:
        raw = json.loads(list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Bad JSON {list_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RuntimeError("elasticity_relative_test3.json must be a list")
    checked = 0
    for entry in raw[:8]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            for base in (str(entry[0]), str(entry[1])):
                rgb = data_dir / f"{base}_rgb.mp4"
                if not rgb.is_file():
                    raise RuntimeError(f"Missing video for list entry: {rgb}")
                checked += 1
    if checked == 0:
        raise RuntimeError("No valid pairs found in elasticity_relative_test3.json (unexpected format).")


def _validate_abs_lists(data_dir: Path, list_path: Path) -> None:
    """作用：抽样校验 absolute list 中若干 basename 存在 *_rgb.mp4。输入：视频根、json。输出：无或抛错。"""
    try:
        raw = json.loads(list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Bad JSON {list_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RuntimeError("elasticity_absolute_test3.json must be a list")
    for name in raw[:12]:
        if not isinstance(name, str):
            continue
        rgb = data_dir / f"{name}_rgb.mp4"
        if not rgb.is_file():
            raise RuntimeError(f"Missing absolute clip: {rgb}")
    bounce = data_dir / "bounce_analysis_results.json"
    if not bounce.is_file():
        raise RuntimeError(f"Missing {bounce} (required by WebVidv2Viscosity / E eval).")


def main() -> int:
    """作用：编排入口。输出：退出码。"""
    args = parse_args()
    svd_root = args.svd_root.resolve()
    repo = _repo_root()
    hf_root = _default_hf_root(svd_root, args.data_hf)
    lists_src = repo / "data" / "lists"
    list_abs = lists_src / "elasticity_absolute_test3.json"
    list_rel = lists_src / "elasticity_relative_test3.json"
    for p in (list_abs, list_rel):
        if not p.is_file():
            print(f"ERROR: bundled list missing: {p}", file=sys.stderr)
            return 1

    rel_link: Path | None = None
    rel_zip: Path | None = None
    if args.rel_video_root is not None:
        rel_link = args.rel_video_root.resolve()
        if not rel_link.is_dir():
            print(f"ERROR: --rel-video-root is not a directory: {rel_link}", file=sys.stderr)
            return 1
    else:
        rel_zip = args.rel_zip
        if rel_zip is None:
            rel_zip = (hf_root / "video_zips" / "elasticity_relative" / "2.16_elasticity_test3.zip").resolve()
        else:
            rel_zip = rel_zip.resolve()
        if not rel_zip.is_file():
            print(f"ERROR: relative zip not found: {rel_zip}", file=sys.stderr)
            return 1

    abs_link: Path | None = None
    if args.abs_video_root is not None:
        abs_link = args.abs_video_root.resolve()
        if not abs_link.is_dir():
            print(f"ERROR: --abs-video-root is not a directory: {abs_link}", file=sys.stderr)
            return 1
    else:
        abs_zip = args.abs_zip
        if abs_zip is None:
            abs_zip = (
                hf_root / "video_zips" / "elasticity_absolute" / "7.15_new_elasticity_video_clips_merged.zip"
            ).resolve()
        else:
            abs_zip = abs_zip.resolve()
        if not abs_zip.is_file():
            print(f"ERROR: absolute zip not found: {abs_zip}", file=sys.stderr)
            return 1

    out_lists = svd_root / "data" / "lists"
    out_lists.mkdir(parents=True, exist_ok=True)
    _copy_file(list_abs, out_lists / "elasticity_absolute_test3.json")
    _copy_file(list_rel, out_lists / "elasticity_relative_test3.json")

    unpack_rel = svd_root / "data" / ".unpack_elasticity_relative_test3"
    unpack_abs = svd_root / "data" / ".unpack_elasticity_absolute_test3"

    try:
        if rel_link is not None:
            inner_rel = rel_link
            gt_in_tree = inner_rel / "gt_restitution1.json"
            if not gt_in_tree.is_file():
                print(
                    f"ERROR: {gt_in_tree} missing under --rel-video-root; "
                    "use a tree that includes gt_restitution1.json (e.g. genesis 2.16_elasticity_test3) "
                    "or stage from --rel-zip instead.",
                    file=sys.stderr,
                )
                return 1
        else:
            assert rel_zip is not None
            inner_rel = _extract_zip_to_unpack(rel_zip, unpack_rel)
            if args.rel_hf_infer_prep:
                prep = repo / "scripts" / "hf_prepare_elasticity_rel_test3_folder.py"
                if not prep.is_file():
                    print(f"ERROR: missing helper script {prep}", file=sys.stderr)
                    return 1
                cmd = [
                    sys.executable,
                    "-u",
                    str(prep),
                    "--data-dir",
                    str(inner_rel),
                    "--list-out",
                    str(out_lists / "elasticity_relative_test3.json"),
                    "--write-gt-in-dir",
                    str(inner_rel),
                ]
                try:
                    subprocess.run(cmd, check=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    print(f"ERROR: HF relative prep failed: {exc}", file=sys.stderr)
                    return 1
            else:
                gt_in_zip_tree = inner_rel / "gt_restitution1.json"
                if not gt_in_zip_tree.is_file():
                    gt_src = args.gt_rel_json
                    if gt_src is None:
                        gt_src = repo / "data" / "elasticity_aux" / "gt_restitution1_relative_test3.json"
                    gt_src = gt_src.resolve()
                    if not gt_src.is_file():
                        print(f"ERROR: gt json missing in zip and fallback: {gt_src}", file=sys.stderr)
                        return 1
                    _copy_file(gt_src, inner_rel / "gt_restitution1.json")
        rel_out = svd_root / "data" / "elasticity_relative_test3"
        _link_replace(inner_rel, rel_out)
        _validate_rel_lists(rel_out, out_lists / "elasticity_relative_test3.json")

        if abs_link is not None:
            abs_out = svd_root / "data" / "elasticity_absolute_test3"
            _link_replace(abs_link, abs_out)
        else:
            inner_abs = _extract_zip_to_unpack(abs_zip, unpack_abs)
            abs_out = svd_root / "data" / "elasticity_absolute_test3"
            _link_replace(inner_abs, abs_out)
        _validate_abs_lists(svd_root / "data" / "elasticity_absolute_test3", out_lists / "elasticity_absolute_test3.json")

        print("OK: staged elasticity test3 under", svd_root / "data")
        print("  relative ->", rel_out.resolve(), "rgb_count=", _count_rgb(rel_out))
        print("  absolute ->", (svd_root / "data" / "elasticity_absolute_test3").resolve(), "rgb_count=", _count_rgb(svd_root / "data" / "elasticity_absolute_test3"))
        if args.rel_video_root is None:
            print(
                "NOTE: default relative zip is genesis 2.16_elasticity_test3 (Table 1 / data_global_map.md). "
                "Override with --rel-video-root or --rel-zip if needed.",
            )
        if args.abs_video_root is None:
            print(
                "NOTE: default absolute zip is genesis 7.15_new_elasticity_video_clips_merged (Table 1 / data_global_map.md). "
                "Override with --abs-video-root or --abs-zip if needed.",
            )
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
