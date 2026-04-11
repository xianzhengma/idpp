#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级说明：从 Hugging Face 拉取公开权重与数据到 `SVD_ROOT` 约定目录（需自行创建 Hub 仓库并填入 ID）。
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _download_repo(repo_id: str, local_dir: str, repo_type: str) -> None:
    """
    输入: repo_id — 如 org/name；local_dir — 本地目录；repo_type — model 或 dataset。
    输出: 无。
    作用: 调用 huggingface_hub.snapshot_download。
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("请 pip install huggingface_hub") from exc
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(f"下载失败 {repo_id}: {exc}") from exc


def main() -> None:
    """
    输入: CLI 参数（HF 仓库 ID 与 SVD_ROOT）。
    输出: 无。
    作用: 将 checkpoint 与数据集快照到本地（占位 ID 需替换为你的发布仓库）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svd-root",
        default=os.environ.get("SVD_ROOT", os.path.join(os.getcwd(), "asset_root")),
        help="资产根目录",
    )
    parser.add_argument(
        "--checkpoints-repo",
        default=os.environ.get("HF_SVD_CHECKPOINTS_REPO", "YOUR_ORG/svd-physics-checkpoints"),
        help="Hugging Face Model 仓库 ID（权重）",
    )
    parser.add_argument(
        "--data-repo",
        default=os.environ.get("HF_SVD_DATA_REPO", "YOUR_ORG/svd-physics-eval-data"),
        help="Hugging Face Dataset 仓库 ID（评测视频与 list json）",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.svd_root)
    os.environ["SVD_ROOT"] = root
    ckpt_dir = os.path.join(root, "checkpoints")
    data_dir = os.path.join(root, "data_hf")
    try:
        os.makedirs(root, exist_ok=True)
        if "YOUR_ORG" in args.checkpoints_repo:
            print(
                "跳过权重下载：请设置 HF_SVD_CHECKPOINTS_REPO 或 --checkpoints-repo 为真实 Hub ID。",
                file=sys.stderr,
            )
        else:
            _download_repo(args.checkpoints_repo, ckpt_dir, "model")
        if "YOUR_ORG" in args.data_repo:
            print(
                "跳过数据下载：请设置 HF_SVD_DATA_REPO 或 --data-repo 为真实 Hub ID。",
                file=sys.stderr,
            )
        else:
            _download_repo(args.data_repo, data_dir, "dataset")
        print("SVD_ROOT=", root)
        print("若已手动放置文件，请按 configs/paths.example.yaml 调整目录。")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
