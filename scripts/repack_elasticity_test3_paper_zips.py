#!/usr/bin/env python3
"""
模块作用：从合作者 genesis 目录（与 data_global_map.md 一致）打包 Table 1 对齐的弹性 test3 zip，
写入本地 dataset/，供 `upload_idpp_hub_zips.py` 上传到 `Oliver-Ma/IDPP-Data` 的 `video_zips/`。

输入：CLI 路径（默认 scratch genesis）；输出两个 zip 文件；失败时非零退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _zip_directory_tree(src_dir: Path, out_zip: Path) -> None:
    """
    作用：将目录树打包为 zip，归档名保留顶层文件夹名（与 stage 脚本「单根解压」约定一致）。

    输入：src_dir 为要打包的根目录；out_zip 为输出 .zip 路径。
    输出：写入 out_zip；若 OSError/zip 失败则抛出。
    """
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")
    parent = src_dir.parent
    base = src_dir.name
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in src_dir.rglob("*"):
                if fp.is_file():
                    arc = fp.relative_to(parent)
                    zf.write(fp, arc.as_posix())
    except (OSError, zipfile.BadZipFile) as exc:
        try:
            if out_zip.is_file():
                out_zip.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Zip failed {src_dir} -> {out_zip}: {exc}") from exc


def main() -> int:
    """作用：CLI 入口。输入：argv。输出：退出码。"""
    here = Path(__file__).resolve().parent
    proj = here.parent.parent
    default_abs = Path("/scratch/shared/beegfs/guanqi/genesis_data/7.15_new_elasticity_video_clips_merged")
    default_rel = Path("/scratch/shared/beegfs/guanqi/genesis_data/2.16_elasticity_test3")
    default_out_abs = proj / "dataset" / "elasticity_absolute" / "7.15_new_elasticity_video_clips_merged.zip"
    default_out_rel = proj / "dataset" / "elasticity_relative" / "2.16_elasticity_test3.zip"
    parser = argparse.ArgumentParser(description="Pack paper-aligned elasticity test3 zips for Hub upload.")
    parser.add_argument("--abs-src", type=Path, default=default_abs, help="Genesis absolute test3 root")
    parser.add_argument("--rel-src", type=Path, default=default_rel, help="Genesis relative test3 root")
    parser.add_argument("--out-abs-zip", type=Path, default=default_out_abs, help="Output zip for absolute")
    parser.add_argument("--out-rel-zip", type=Path, default=default_out_rel, help="Output zip for relative")
    parser.add_argument("--rel-only", action="store_true", help="Only pack relative zip")
    parser.add_argument("--abs-only", action="store_true", help="Only pack absolute zip")
    args = parser.parse_args()
    try:
        if not args.rel_only:
            print("Packing absolute ->", args.out_abs_zip)
            _zip_directory_tree(args.abs_src, args.out_abs_zip)
        if not args.abs_only:
            print("Packing relative ->", args.out_rel_zip)
            _zip_directory_tree(args.rel_src, args.out_rel_zip)
        print("OK. Next: sanitize (RGB-only policy), then upload, e.g.:")
        print(
            "  python3 scripts/upload_idpp_hub_zips.py --single-zip dataset/elasticity_absolute/7.15_new_elasticity_video_clips_merged.zip",
        )
        print(
            "  python3 scripts/upload_idpp_hub_zips.py --single-zip dataset/elasticity_relative/2.16_elasticity_test3.zip",
        )
        print("Remove deprecated zips from dataset/ before full-folder upload if you no longer want them on Hub.")
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
