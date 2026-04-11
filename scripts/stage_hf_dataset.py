#!/usr/bin/env python3
"""Stage video data for Hugging Face upload: merge genesis + local dataset into one tree."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:
    print('pip install pyyaml', file=sys.stderr)
    raise


def load_manifest(path: Path) -> dict:
    """Load YAML manifest; returns dict with key 'entries' (list of {dest, src})."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f'Failed to read manifest: {exc}') from exc


def link_or_copy_tree(src: Path, dst: Path, use_symlink: bool) -> None:
    """Mirror directory src into dst using symlink tree or copytree."""
    try:
        if dst.exists():
            shutil.rmtree(dst)
        if use_symlink:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src.resolve(), dst, target_is_directory=True)
        else:
            shutil.copytree(src, dst, symlinks=True)
    except OSError as exc:
        raise RuntimeError(f'Failed to stage {src} -> {dst}: {exc}') from exc


def extract_zip_to(zip_path: Path, dst: Path) -> None:
    """Extract a zip archive into dst (created empty first)."""
    try:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dst)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f'Failed to extract {zip_path}: {exc}') from exc


def stage_one_entry(src: Path, dest_rel: str, staging: Path, use_symlink: bool) -> str:
    """Stage one manifest row; returns status string ok|skip|zip."""
    dst = staging / dest_rel
    if not src.exists():
        return f'skip missing: {src}'
    if src.is_file() and src.suffix.lower() == '.zip':
        extract_zip_to(src, dst)
        return f'zip -> {dest_rel}'
    if src.is_dir():
        link_or_copy_tree(src, dst, use_symlink)
        return f'dir -> {dest_rel}'
    return f'skip not dir/zip: {src}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage HF dataset from hf_dataset_manifest.yaml')
    parser.add_argument('--manifest', required=True, help='Path to manifest YAML')
    parser.add_argument('--out', required=True, help='Staging output directory')
    parser.add_argument('--symlink', action='store_true', help='Symlink dirs instead of copy')
    parser.add_argument('--dry-run', action='store_true', help='Print actions only')
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    staging = Path(args.out).resolve()
    data = load_manifest(manifest_path)
    entries = data.get('entries') or []
    try:
        if not args.dry_run:
            staging.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
    for row in entries:
        if not isinstance(row, dict):
            continue
        dest_rel = row.get('dest')
        src_raw = row.get('src')
        if not dest_rel or not src_raw:
            continue
        src = Path(os.path.expandvars(str(src_raw))).expanduser()
        if args.dry_run:
            print(f'would stage {src} -> {staging / dest_rel}')
            continue
        try:
            msg = stage_one_entry(src, dest_rel, staging, args.symlink)
            print(msg)
        except RuntimeError as exc:
            print(f'ERROR {dest_rel}: {exc}', file=sys.stderr)
    if not args.dry_run:
        print(f'Done. Upload with: huggingface-cli upload-large-folder <repo_id> {staging} --repo-type=dataset')


if __name__ == '__main__':
    main()
