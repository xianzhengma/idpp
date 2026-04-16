#!/usr/bin/env python3
# 模块作用：将论文 B 轨（V-JEPA mirror，`paper_scores_B_sources_and_checkpoints.md`）对应的 `.ckpt`
# 批量上传到 Hugging Face **model** 仓库；大文件由 `huggingface_hub` 走 LFS。
# 输入：CLI（`--ckpt-root`、`--repo-id`、`--elasticity-only`、`--dry-run` 等）；鉴权 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`。
# 输出：逐文件打印进度；失败时 stderr + 非零退出码。

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# 相对 `CKPT_ROOT` 的 7 个权重（不含 v121 test3* scratch）。
BACKUP_REL_PATHS: tuple[str, ...] = (
    "training_1024_v110_vjepa_norm/checkpoints/epoch=120-step=15125.ckpt",
    "training_1024_v119_vjepa_norm/checkpoints/epoch=21-step=13000.ckpt",
    "training_1024_v120_vjepa_norm/checkpoints/epoch=116-step=14000.ckpt",
    "training_1024_v121_vjepa_norm/checkpoints/epoch=120-step=15125.ckpt",
    "training_1024_v120_vjepa_ft/checkpoints/epoch=5499-step=5500.ckpt",
    "training_1024_v124_vjepa_norm/checkpoints/epoch=21-step=8375.ckpt",
    "training_1024_v125_vjepa_norm/checkpoints/epoch=126-step=15875.ckpt",
)

ELASTICITY_REL_PATHS: tuple[str, ...] = (
    "training_1024_v124_vjepa_norm/checkpoints/epoch=21-step=8375.ckpt",
    "training_1024_v125_vjepa_norm/checkpoints/epoch=126-step=15875.ckpt",
)

V121_TEST3STAR_SCRATCH = Path(
    "/scratch/shared/beegfs/guanqi/dynamicrafter_ckpt_backup_3.1/logs/"
    "e_729_v121_vjepa_ft_t3train_norm15125_trainon_test_20260409/"
    "training_1024_v121_vjepa_ft_t3train_norm15125_trainon_test/checkpoints/"
    "epoch=1-step=10.ckpt"
)

def scratch_upload_pair(prefix: str) -> tuple[Path, str]:
    """
    作用：v121 test3*（norm15125_ttest）scratch 权重在 Hub 上的目标路径。
    输入：仓库顶层前缀（与 --prefix 一致）。
    输出：(本地 Path, path_in_repo)。
    """
    pfx = prefix.strip().rstrip("/")
    dest = f"{pfx}/v121_fr_rel_test3star_norm15125_ft/epoch=1-step=10.ckpt"
    return (V121_TEST3STAR_SCRATCH, dest)


def parse_args() -> argparse.Namespace:
    """作用：解析命令行。输入：sys.argv。输出：Namespace。"""
    default_root = "/work/guanqi/proj5/Inferring_dynamic_physics/backup_dynamicrafer_ckpt"
    parser = argparse.ArgumentParser(
        description="Upload V-JEPA (B / mirror) checkpoints to HF model repo."
    )
    parser.add_argument("--ckpt-root", type=Path, default=Path(default_root), help="backup_dynamicrafer_ckpt")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_SVD_CHECKPOINTS_REPO", "Oliver-Ma/IDPP-Checkpoints"),
        help="HF model repo id",
    )
    parser.add_argument(
        "--prefix",
        default="B_vjepa_mirror",
        help="Top folder in repo (files also keep training_*/checkpoints/ under it)",
    )
    parser.add_argument(
        "--elasticity-only",
        action="store_true",
        help="Only v124 + v125 (elasticity abs/rel); skip other B ckpts and v121 scratch",
    )
    parser.add_argument("--dry-run", action="store_true", help="List uploads only")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N files (0 = all)")
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        help="Upload smaller files first (smoke / faster feedback)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip local paths that do not exist instead of failing",
    )
    return parser.parse_args()


def repo_dest(prefix: str, rel_under_root: str) -> str:
    """作用：生成仓库内路径。输入：前缀、相对 ckpt 根的路径。输出：posix 风格字符串。"""
    rel_clean = rel_under_root.lstrip("/")
    return f"{prefix.strip().rstrip('/')}/{rel_clean}"


def collect_upload_pairs(
    ckpt_root: Path,
    prefix: str,
    elasticity_only: bool,
) -> list[tuple[Path, str]]:
    """
    作用：组装 (本地绝对路径, 仓库 path_in_repo) 列表。
    输入：ckpt 根目录、Hub 前缀、是否仅弹性 v124/v125。
    输出：有序列表。
    """
    try:
        root = ckpt_root.resolve()
    except OSError as exc:
        raise RuntimeError(f"Invalid ckpt root {ckpt_root}: {exc}") from exc

    pairs: list[tuple[Path, str]] = []
    rels = ELASTICITY_REL_PATHS if elasticity_only else BACKUP_REL_PATHS

    for rel in rels:
        local_path = root / rel
        pairs.append((local_path, repo_dest(prefix, rel)))

    if not elasticity_only:
        pairs.append(scratch_upload_pair(prefix))

    return pairs


def filter_existing_pairs(
    pairs: list[tuple[Path, str]],
    skip_missing: bool,
) -> list[tuple[Path, str]]:
    """
    作用：按存在性过滤本地路径。
    输入：候选列表、是否跳过缺失。
    输出：可上传列表；若不跳过且存在缺失则抛错。
    """
    missing = [p for p, _ in pairs if not p.is_file()]
    if not missing:
        return pairs

    if skip_missing:
        kept = [(p, d) for p, d in pairs if p.is_file()]
        for path in missing:
            print(f"SKIP missing: {path}", flush=True)
        return kept

    raise RuntimeError("Local checkpoint missing: " + ", ".join(str(m) for m in missing))


def sort_pairs_by_size(pairs: list[tuple[Path, str]]) -> None:
    """作用：按文件体积升序原地排序。输入：pairs；输出：无（原地修改）。"""
    try:
        pairs.sort(key=lambda item: item[0].stat().st_size)
    except OSError as exc:
        raise RuntimeError(f"Cannot stat files for sort: {exc}") from exc


def hf_api_from_env() -> object:
    """
    作用：用环境变量 Token 构造已鉴权的 HfApi。
    输入：环境变量。
    输出：HfApi 实例。
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
            "Hugging Face 鉴权失败。请 export HF_TOKEN 或在同环境 huggingface-cli login。"
        ) from exc


def upload_one_checkpoint(api: object, repo_id: str, local_path: Path, dest: str) -> None:
    """
    作用：上传单个 .ckpt 到 model 仓库。
    输入：HfApi、repo_id、本地路径、仓库内路径。
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
    """作用：入口：收集路径、鉴权、逐个上传。输出：退出码。"""
    args = parse_args()

    try:
        pairs = collect_upload_pairs(args.ckpt_root, args.prefix, args.elasticity_only)
        pairs = filter_existing_pairs(pairs, args.skip_missing)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not pairs:
        print("ERROR: no checkpoint files to upload after filtering", file=sys.stderr)
        return 1

    if args.sort_by_size:
        try:
            sort_pairs_by_size(pairs)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    limit = args.limit if args.limit and args.limit > 0 else len(pairs)
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

    print("OK: V-JEPA checkpoint upload pass complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
