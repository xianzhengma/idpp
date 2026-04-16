#!/usr/bin/env python3
# 模块作用：将本地 dataset 各 lane 目录下的 .zip 批量上传到 Hugging Face dataset 仓库的 video_zips/ 前缀下。
# 输入：CLI（--dataset-root、--repo-id、--dry-run、--limit 等）；鉴权使用环境变量 HF_TOKEN 或 huggingface-cli 已登录缓存。
# 输出：每个文件上传后在 stdout 打印一行；失败时 stderr + 非零退出码。

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    default_root = "/work/guanqi/proj5/Inferring_dynamic_physics/dataset"
    script_dir = Path(__file__).resolve().parent
    default_readme = script_dir.parent / "docs" / "IDPP_Data_README.md"
    parser = argparse.ArgumentParser(description="Upload dataset lane zips to HF dataset repo (video_zips/…).")
    parser.add_argument("--dataset-root", type=Path, default=Path(default_root), help="Root containing lane subdirs")
    parser.add_argument("--repo-id", default="Oliver-Ma/IDPP-Data", help="HF dataset repo id")
    parser.add_argument("--prefix", default="video_zips", help="Path prefix inside the repo")
    parser.add_argument("--dry-run", action="store_true", help="List uploads only")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N files (0 = all)")
    parser.add_argument(
        "--readme",
        type=Path,
        default=default_readme if default_readme.is_file() else None,
        help="README.md for the dataset card (default: SVD_public/docs/IDPP_Data_README.md if present)",
    )
    parser.add_argument("--no-readme", action="store_true", help="Skip README upload")
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        help="Upload smaller zips first (smoke test / faster feedback)",
    )
    parser.add_argument(
        "--single-zip",
        type=Path,
        default=None,
        help="Upload only this file; must exist under --dataset-root (skips directory scan).",
    )
    return parser.parse_args()


def validate_zip_under_dataset_root(dataset_root: Path, single_zip: Path) -> Path:
    """
    作用：校验待上传的单个 zip 位于 dataset 根下且为普通文件。
    输入：dataset_root、用户给出的 single_zip（可相对路径）。
    输出：resolve 后的 Path。
    """
    root = dataset_root.resolve()
    try:
        candidate = single_zip.expanduser().resolve()
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve path {single_zip}: {exc}") from exc
    if not candidate.is_file():
        raise RuntimeError(f"Not a file: {candidate}")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Zip must be under dataset-root ({root}), got {candidate}"
        ) from exc
    return candidate


def collect_zip_paths(dataset_root: Path) -> list[Path]:
    """
    作用：收集 dataset/<lane>/*.zip（仅一层子目录），按路径排序。
    输入：dataset_root。
    输出：Path 列表。
    """
    try:
        zips: list[Path] = []
        for child in sorted(dataset_root.iterdir()):
            if not child.is_dir():
                continue
            for zp in sorted(child.glob("*.zip")):
                if zp.is_file():
                    zips.append(zp)
        return zips
    except OSError as exc:
        raise RuntimeError(f"Failed to scan {dataset_root}: {exc}") from exc


def path_in_repo(prefix: str, dataset_root: Path, file_path: Path) -> str:
    """作用：生成仓库内相对路径。输入：prefix、根目录、本地 zip 路径。输出：posix 风格字符串。"""
    rel = file_path.resolve().relative_to(dataset_root.resolve())
    return f"{prefix}/{rel.as_posix()}"


def upload_readme(api: object, repo_id: str, readme_path: Path) -> None:
    """作用：上传数据集 README。输入：HfApi 实例、repo_id、本地 README 路径。输出：无。"""
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset README",
        )
    except Exception as exc:
        raise RuntimeError(f"README upload failed: {exc}") from exc


def upload_one_zip(api: object, repo_id: str, local_path: Path, dest: str) -> None:
    """作用：上传单个 zip。输入：HfApi、repo_id、本地路径、仓库内路径。输出：无。"""
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {dest}",
        )
    except Exception as exc:
        raise RuntimeError(f"Upload failed {local_path} -> {dest}: {exc}") from exc


def main() -> int:
    """作用：入口：鉴权、枚举 zip、逐个上传。输出：退出码。"""
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install -U huggingface_hub", file=sys.stderr)
        return 1

    try:
        if args.single_zip is not None:
            zips = [validate_zip_under_dataset_root(args.dataset_root, args.single_zip)]
        else:
            zips = collect_zip_paths(args.dataset_root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not zips:
        print(f"No zip files under {args.dataset_root}/*/", file=sys.stderr)
        return 1

    if args.sort_by_size and args.single_zip is None:
        try:
            zips.sort(key=lambda p: p.stat().st_size)
        except OSError as exc:
            print(f"ERROR: cannot stat zips for sort: {exc}", file=sys.stderr)
            return 1

    if args.dry_run:
        if (
            (not args.no_readme)
            and args.readme
            and args.readme.is_file()
            and args.single_zip is None
        ):
            print(f"[dry-run] README: {args.readme.resolve()} -> README.md")
        limit = args.limit if args.limit and args.limit > 0 else len(zips)
        for idx, zp in enumerate(zips[:limit]):
            dest = path_in_repo(args.prefix, args.dataset_root, zp)
            print(f"[dry-run] {zp} -> {dest}")
        print(f"OK: dry-run, {min(limit, len(zips))} zip(s).")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        api = HfApi(token=token if token else None)
        api.whoami()
    except Exception as exc:
        print(
            f"ERROR: Hugging Face 鉴权失败: {exc}\n"
            "请设置环境变量 HF_TOKEN 或 HUGGINGFACE_HUB_TOKEN（勿泄露），"
            "或在 **同一终端** 执行 huggingface-cli login。"
            "注意：你在 SSH/本机 export 的变量不会自动进入 Cursor Agent 的子进程；"
            "Agent 内上传需在 Cursor 的 Environment 里配置，或直接在集群登录节点运行本脚本。",
            file=sys.stderr,
        )
        return 1

    skip_readme_for_single = args.single_zip is not None
    if (
        (not args.no_readme)
        and args.readme
        and args.readme.is_file()
        and (not skip_readme_for_single)
    ):
        dest_readme = args.readme.resolve()
        print(f"README: {dest_readme} -> README.md", flush=True)
        if not args.dry_run:
            try:
                upload_readme(api, args.repo_id, dest_readme)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

    limit = args.limit if args.limit and args.limit > 0 else len(zips)
    for idx, zp in enumerate(zips[:limit]):
        dest = path_in_repo(args.prefix, args.dataset_root, zp)
        print(f"[{idx + 1}/{min(limit, len(zips))}] {zp} -> {dest}", flush=True)
        try:
            upload_one_zip(api, args.repo_id, zp, dest)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    print("OK: upload pass complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
