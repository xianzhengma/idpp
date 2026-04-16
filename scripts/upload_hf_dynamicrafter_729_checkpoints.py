#!/usr/bin/env python3
# 模块作用：将论文 E 轨（DynamiCrafter 7.29 / `e729_nd_*`，见 `docs/HF_ASSETS_PAPER_ALIGNMENT.md`）共 9 个唯一 `.ckpt`
# 上传到 Hugging Face **model** 仓库；大文件走 LFS。
# 输入：CLI；鉴权 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` 或 `huggingface-cli login`。
# 输出：逐文件进度；失败 stderr + 非零退出码。

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# v120 test3 新数据扫点（E2），与主 v120（E1）不同文件；本地为 staging 绝对路径。
V120_SWEEP_LOCAL = Path(
    "/work/guanqi/proj5/Inferring_dynamic_physics/SVD/guanqi_eval_scripts/"
    "e729_ckpt_sweep_test3/staging_7.29_sweep/training_1024_v120_7.29/checkpoints/"
    "epoch=0-step=125.ckpt"
)

V120_SWEEP_DEST_REL = "training_1024_v120_7.29/checkpoints/epoch=0-step=125.ckpt"

# 相对 `CKPT_ROOT` 的其余 7 个（E1 主 v120 + E3–E9，顺序与清单一致）。
BACKUP_REL_PATHS_ORDERED: tuple[str, ...] = (
    "training_1024_v120_7.29/checkpoints/epoch=18-step=4500.ckpt",
    "training_1024_v120_ft_7.29/checkpoints/epoch=3299-step=3300.ckpt",
    "training_1024_v121_7.29/checkpoints/epoch=17-step=2250.ckpt",
    "training_1024_v121_ft_7.29/checkpoints/epoch=378-step=2650.ckpt",
    "training_1024_v122_7.29/checkpoints/epoch=3-step=2375.ckpt",
    "training_1024_v123_7.29/checkpoints/epoch=3-step=500.ckpt",
    "training_1024_v124_7.29/checkpoints/epoch=4-step=2625.ckpt",
    "training_1024_v125_7.29/checkpoints/epoch=3-step=500.ckpt",
)


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    default_root = "/work/guanqi/proj5/Inferring_dynamic_physics/backup_dynamicrafer_ckpt"
    parser = argparse.ArgumentParser(
        description="Upload DynamiCrafter 7.29 (E / e729_nd) checkpoints to HF model repo."
    )
    parser.add_argument("--ckpt-root", type=Path, default=Path(default_root), help="backup_dynamicrafer_ckpt")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_SVD_CHECKPOINTS_REPO", "Oliver-Ma/IDPP-Checkpoints"),
        help="HF model repo id",
    )
    parser.add_argument(
        "--prefix",
        default="E_7.29",
        help="Top folder in repo (keeps training_*/checkpoints/ layout under it)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip E2 (v120 test3 staging sweep epoch=0-step=125)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List uploads only")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N files (0 = all)")
    parser.add_argument("--sort-by-size", action="store_true", help="Smaller files first")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing locals instead of failing",
    )
    return parser.parse_args()


def repo_dest(prefix: str, rel_under_root: str) -> str:
    """作用：Hub 内路径。输入：前缀、相对 ckpt 根的相对路径。输出：posix 字符串。"""
    rel_clean = rel_under_root.lstrip("/")
    return f"{prefix.strip().rstrip('/')}/{rel_clean}"


def collect_upload_pairs(
    ckpt_root: Path,
    prefix: str,
    skip_sweep: bool,
) -> list[tuple[Path, str]]:
    """
    作用：组装 (本地路径, path_in_repo)；顺序 E1 → E2（可选）→ E3…E9。
    输入：ckpt 根、前缀、是否跳过扫点。
    输出：列表。
    """
    try:
        root = ckpt_root.resolve()
    except OSError as exc:
        raise RuntimeError(f"Invalid ckpt root {ckpt_root}: {exc}") from exc

    pairs: list[tuple[Path, str]] = []
    first_rel = BACKUP_REL_PATHS_ORDERED[0]
    pairs.append((root / first_rel, repo_dest(prefix, first_rel)))

    if not skip_sweep:
        pairs.append((V120_SWEEP_LOCAL, repo_dest(prefix, V120_SWEEP_DEST_REL)))

    for rel in BACKUP_REL_PATHS_ORDERED[1:]:
        pairs.append((root / rel, repo_dest(prefix, rel)))

    return pairs


def filter_existing_pairs(
    pairs: list[tuple[Path, str]],
    skip_missing: bool,
) -> list[tuple[Path, str]]:
    """
    作用：按存在性过滤本地文件。
    输入：候选对、是否跳过缺失。
    输出：可上传列表；不跳过且缺失则抛错。
    """
    missing = [p for p, _ in pairs if not p.is_file()]
    if not missing:
        return pairs

    if skip_missing:
        for path in missing:
            print(f"SKIP missing: {path}", flush=True)
        return [(p, d) for p, d in pairs if p.is_file()]

    raise RuntimeError("Local checkpoint missing: " + ", ".join(str(m) for m in missing))


def sort_pairs_by_size(pairs: list[tuple[Path, str]]) -> None:
    """作用：按体积升序排序。输入：pairs；输出：原地修改。"""
    try:
        pairs.sort(key=lambda item: item[0].stat().st_size)
    except OSError as exc:
        raise RuntimeError(f"Cannot stat for sort: {exc}") from exc


def hf_api_from_env() -> object:
    """
    作用：构造已鉴权 HfApi。
    输入：环境变量。
    输出：HfApi。
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("pip install -U huggingface_hub") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        api = HfApi(token=token if token else None)
        api.whoami()
        return api
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face 鉴权失败。请 export HF_TOKEN 或 huggingface-cli login。"
        ) from exc


def upload_one_checkpoint(api: object, repo_id: str, local_path: Path, dest: str) -> None:
    """
    作用：上传单个 .ckpt。
    输入：HfApi、repo_id、本地路径、path_in_repo。
    输出：无；失败抛 RuntimeError。
    """
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {dest}",
        )
    except Exception as exc:
        raise RuntimeError(f"Upload failed {local_path} -> {dest}: {exc}") from exc


def main() -> int:
    """作用：入口。输出：退出码。"""
    args = parse_args()

    try:
        pairs = collect_upload_pairs(args.ckpt_root, args.prefix, args.skip_sweep)
        pairs = filter_existing_pairs(pairs, args.skip_missing)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not pairs:
        print("ERROR: no files to upload", file=sys.stderr)
        return 1

    if args.sort_by_size:
        try:
            sort_pairs_by_size(pairs)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    limit = args.limit if args.limit > 0 else len(pairs)
    todo = pairs[:limit]

    if args.dry_run:
        for local_path, dest in todo:
            print(f"[dry-run] {local_path} -> {dest}", flush=True)
        print(f"OK: dry-run, {len(todo)} file(s).", flush=True)
        return 0

    try:
        api = hf_api_from_env()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for idx, (local_path, dest) in enumerate(todo, start=1):
        print(f"[{idx}/{len(todo)}] {local_path} -> {dest}", flush=True)
        try:
            upload_one_checkpoint(api, args.repo_id, local_path, dest)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    print("OK: DynamiCrafter 7.29 (E) checkpoint upload pass complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
