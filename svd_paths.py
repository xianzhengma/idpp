#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块级说明：统一解析环境变量 `SVD_ROOT`，供算分脚本与构建配置脚本使用。
布局约定见 `docs/HUGGINGFACE.md` 与 `configs/paths.example.yaml`。
"""

from __future__ import annotations

import os
from typing import Optional


def get_svd_root(explicit_root: Optional[str] = None) -> str:
    """
    输入: explicit_root — 若非 None 则优先使用该路径（测试用）。
    输出: 仓库根下的资产根目录绝对路径（含 checkpoints、data、results）。
    作用: 从环境变量 `SVD_ROOT` 读取；未设置时默认为当前工作目录下的 `./asset_root`。
    """
    try:
        if explicit_root is not None:
            return os.path.abspath(os.path.expanduser(explicit_root))
        env_root = os.environ.get("SVD_ROOT", "").strip()
        if env_root:
            return os.path.abspath(os.path.expanduser(env_root))
        return os.path.abspath(os.path.join(os.getcwd(), "asset_root"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法解析 SVD_ROOT: {exc}") from exc


def get_results_newdata_root(root: Optional[str] = None) -> str:
    """
    输入: root — 可选，传给 get_svd_root。
    输出: E 版 newdata 评测 JSON 根目录（其下含 `e729_nd_*` 子目录）。
    作用: 与 `calculate_e729_newdata_dc_scores.py` 中 RESULTS_ROOT 一致。
    """
    try:
        base = get_svd_root(root)
        return os.path.join(base, "results", "e729_newdata_20260327")
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"无法拼接 results 路径: {exc}") from exc


def get_checkpoint_path(relative_under_checkpoints: str, root: Optional[str] = None) -> str:
    """
    输入: relative_under_checkpoints — 相对于 `checkpoints/` 的相对路径片段。
    输出: 绝对路径。
    作用: 拼装 `SVD_ROOT/checkpoints/<relative>`。
    """
    try:
        base = get_svd_root(root)
        return os.path.join(base, "checkpoints", relative_under_checkpoints)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"无法拼接 checkpoint 路径: {exc}") from exc
