#!/usr/bin/env python3
"""
模块作用：在不加载 Lightning 模型的前提下，实例化四个评测 yaml 中的 validation Dataset，
并各取一个样本，用于快速验证路径、json、RGB-only 与 gt 文件是否就绪。

输入：--svd-root、--configs-dir（默认 SVD_public/configs/reproduce/generated）。
输出：打印每套 len 与一次 __getitem__ 成功信息；失败非零退出。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """作用：解析 CLI。输入：sys.argv。输出：Namespace。"""
    here = Path(__file__).resolve().parent
    default_cfg = here.parent / "configs" / "reproduce" / "generated"
    parser = argparse.ArgumentParser(description="Smoke-test elasticity test3 dataloaders from resolved YAMLs.")
    parser.add_argument("--svd-root", type=Path, required=True, help="SVD_ROOT with data/ and lists/")
    parser.add_argument("--configs-dir", type=Path, default=default_cfg, help="Directory with four generated yaml files")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    """作用：读取 YAML 为 dict。输入：路径。输出：dict。"""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install PyYAML") from exc
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping root in {path}")
    return data


def _instantiate_dataset(cfg_path: Path) -> object:
    """作用：从 yaml 的 data.validation 实例化 Dataset。输入：yaml 路径。输出：Dataset 实例。"""
    here = Path(__file__).resolve().parent
    repo = here.parent
    main_dir = repo / "main"
    try:
        os.chdir(main_dir)
    except OSError as exc:
        raise RuntimeError(f"Cannot chdir to {main_dir}: {exc}") from exc
    for p in (str(main_dir), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from utils.utils import instantiate_from_config
    except ImportError as exc:
        raise RuntimeError(f"Cannot import instantiate_from_config: {exc}") from exc
    cfg = _load_yaml(cfg_path)
    data = cfg.get("data") or {}
    params = data.get("params") or {}
    val = params.get("validation")
    if not isinstance(val, dict):
        raise RuntimeError(f"No data.validation in {cfg_path}")
    return instantiate_from_config(val)


def _smoke_one(name: str, cfg_path: Path) -> None:
    """作用：对单个 yaml 跑 len + __getitem__(0)。输入：标签、路径。输出：无或抛错。"""
    if not cfg_path.is_file():
        raise RuntimeError(f"Missing config: {cfg_path}")
    ds = _instantiate_dataset(cfg_path)
    n = len(ds)
    if n <= 0:
        raise RuntimeError(f"{name}: dataset length is 0 ({cfg_path})")
    _ = ds[0]
    print(f"OK {name}: len={n} cfg={cfg_path}")


def main() -> int:
    """作用：入口。输出：退出码。"""
    args = parse_args()
    root = args.svd_root.resolve()
    os.environ["SVD_ROOT"] = str(root)
    cfg_dir = args.configs_dir.resolve()
    files = [
        ("E_v124_abs", cfg_dir / "config_e729_nd_v124_test3_el_abs.yaml"),
        ("E_v125_rel", cfg_dir / "config_e729_nd_v125_test3_el_rel.yaml"),
        ("B_v124_abs", cfg_dir / "config_v124_vjepa_test3.yaml"),
        ("B_v125_rel", cfg_dir / "config_v125_vjepa_test3.yaml"),
    ]
    try:
        for label, path in files:
            _smoke_one(label, path)
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
