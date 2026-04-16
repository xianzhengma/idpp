#!/usr/bin/env python3
# 模块作用：按 oracle_lane_manifest.yaml 批量处理 `dataset/<lane>/*.zip`：解压 → sanitize_idpp_public_tree → 校验 _rgb 数 → repack_zip_tree。
# 输入：CLI（--dataset-root、--manifest、--output-dir 或 --in-place、--oracle-json-dir）；默认 dry-run。
# 输出：每 zip 一行 JSON 摘要；失败非零退出。

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """作用：解析 CLI。输入：sys.argv。输出：Namespace。"""
    here = Path(__file__).resolve().parent
    default_manifest = here.parent / "data" / "oracle_lane_manifest.yaml"
    parser = argparse.ArgumentParser(description="Sanitize + repack IDPP dataset zips for public Hub.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/work/guanqi/proj5/Inferring_dynamic_physics/dataset"),
        help="Root containing <lane>/*.zip",
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest, help="oracle_lane_manifest.yaml")
    parser.add_argument(
        "--oracle-json-dir",
        type=Path,
        default=None,
        help="Override ORACLE_JSON_DIR / manifest default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write sanitized zips here (mirrors lane/); required unless --in-place",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace zip under dataset-root (creates .bak before replace)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually sanitize + repack (default: dry-run)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N entries (0=all)")
    parser.add_argument(
        "--sensitive-mp4-only",
        action="store_true",
        help="Only remove _seg / red_circle* mp4; skip Oracle rgb prune and expected_rgb checks (no ORACLE_JSON_DIR needed)",
    )
    parser.add_argument(
        "--hf-upload-each",
        action="store_true",
        help="After each successful zip (--apply), start HF upload in a subprocess; overlaps with next sanitize; needs HF_TOKEN.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="Oliver-Ma/IDPP-Data",
        help="HF dataset repo id when using --hf-upload-each",
    )
    parser.add_argument(
        "--hf-prefix",
        default="video_zips",
        help="Path prefix inside the repo when using --hf-upload-each",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    """作用：读取 YAML manifest。输入：路径。输出：dict。"""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install PyYAML") from exc
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot load manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or "entries" not in data:
        raise RuntimeError("manifest missing top-level 'entries'")
    return data


def oracle_dir_from_manifest(manifest: dict[str, Any], override: Path | None) -> Path:
    """作用：解析 Oracle JSON 根目录。输入：manifest、CLI 覆盖。输出：Path。"""
    if override is not None:
        return override.resolve()
    env = os.environ.get("ORACLE_JSON_DIR", "").strip()
    if env:
        return Path(env).resolve()
    default = manifest.get("default_oracle_json_dir", "/work/guanqi/Genesis/oracle_py_files")
    return Path(str(default)).resolve()


def zip_inner_root(extract_parent: Path, zip_path: Path) -> Path:
    """
    作用：解压后得到用于 sanitize / repack 的「单顶层目录」。
    输入：解压父目录、原始 zip 路径。
    输出：内容根 Path（与 repack_zip_tree 期望的 basename 一致）。
    """
    stem = zip_path.stem
    guess = extract_parent / stem
    if guess.is_dir():
        return guess

    children = [p for p in extract_parent.iterdir() if p.name not in (".", "..")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]

    # 多顶层目录或顶层直接是文件：包进 <stem>/ 以便 repack 与 Hub 历史布局一致。
    wrap = extract_parent / stem
    if wrap.exists():
        raise RuntimeError(f"Wrap dir already exists: {wrap}")
    wrap.mkdir(parents=True, exist_ok=False)
    try:
        for child in children:
            shutil.move(str(child), str(wrap / child.name))
    except OSError as exc:
        raise RuntimeError(f"Failed to normalize zip layout under {extract_parent}: {exc}") from exc
    return wrap


def count_rgb_mp4(root: Path) -> int:
    """作用：递归统计 *_rgb.mp4。输入：根目录。输出：个数。"""
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.name.endswith("_rgb.mp4"):
                total += 1
    except OSError as exc:
        raise RuntimeError(f"rglob failed {root}: {exc}") from exc
    return total


def run_subprocess(cmd: list[str]) -> None:
    """作用：运行子进程并检查退出码。输入：命令列表。输出：无或抛错。"""
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"subprocess failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def hf_token_present() -> bool:
    """作用：判断当前环境是否具备 HF Hub 上传所需 Token。输入：无。输出：bool。"""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def spawn_hf_upload_process(
    upload_script: Path,
    dataset_root: Path,
    output_zip: Path,
    repo_id: str,
    prefix: str,
) -> subprocess.Popen[Any]:
    """
    作用：为单个已落盘的 zip 启动 HF 上传子进程（可与后续 sanitize 并行）。
    输入：upload 脚本路径、dataset 根、zip 绝对路径、仓库 id、仓库内 prefix。
    输出：已启动、尚未 wait 的 Popen。
    """
    cmd = [
        sys.executable,
        "-u",
        str(upload_script),
        "--dataset-root",
        str(dataset_root),
        "--single-zip",
        str(output_zip),
        "--repo-id",
        repo_id,
        "--prefix",
        prefix,
        "--no-readme",
    ]
    try:
        return subprocess.Popen(cmd)
    except OSError as exc:
        raise RuntimeError(f"Cannot start HF upload subprocess: {exc}") from exc


def wait_hf_upload_processes(procs: list[subprocess.Popen[Any]]) -> list[tuple[int, int]]:
    """
    作用：按顺序等待所有 HF 上传子进程并收集退出码。
    输入：Popen 列表。
    输出：(索引, returncode) 列表。
    """
    results: list[tuple[int, int]] = []
    for idx, proc in enumerate(procs):
        try:
            code = int(proc.wait())
        except OSError as exc:
            raise RuntimeError(f"wait() failed for HF upload job #{idx}: {exc}") from exc
        results.append((idx, code))
    return results


def process_one_entry(
    entry: dict[str, Any],
    dataset_root: Path,
    oracle_root: Path,
    output_dir: Path | None,
    in_place: bool,
    apply: bool,
    sanitize_script: Path,
    repack_script: Path,
    sensitive_mp4_only: bool,
) -> dict[str, Any]:
    """
    作用：处理 manifest 中单条 zip。
    输入：entry 字典、路径、是否写入。
    输出：摘要 dict。
    """
    lane = str(entry["lane"])
    zip_name = str(entry["zip"])
    split_name = str(entry.get("split", ""))
    oracle_name = entry.get("oracle_json")
    expected = entry.get("expected_rgb")

    zip_path = (dataset_root / lane / zip_name).resolve()
    summary: dict[str, Any] = {
        "lane": lane,
        "zip": zip_name,
        "zip_path": str(zip_path),
        "exists": zip_path.is_file(),
    }
    if not zip_path.is_file():
        summary["status"] = "skipped_missing_zip"
        return summary

    prune_rgb = (not sensitive_mp4_only) and oracle_name is not None and split_name in ("test1", "test2")
    oracle_path: Path | None = None
    summary["oracle_exists"] = True
    summary["sensitive_mp4_only"] = bool(sensitive_mp4_only)
    if prune_rgb:
        oracle_path = (oracle_root / str(oracle_name)).resolve()
        summary["oracle_path"] = str(oracle_path)
        summary["oracle_exists"] = oracle_path.is_file()
        if not oracle_path.is_file():
            summary["status"] = "error_missing_oracle_json"
            return summary

    if not apply:
        summary["status"] = "dry_run_ok"
        summary["would_prune_rgb"] = prune_rgb
        summary["expected_rgb_after"] = expected if prune_rgb else None
        return summary

    tmp = Path(tempfile.mkdtemp(prefix="idpp_san_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmp)

        inner = zip_inner_root(tmp, zip_path)
        cmd_san = [
            sys.executable,
            str(sanitize_script),
            "--root",
            str(inner),
        ]
        if prune_rgb and oracle_path is not None:
            cmd_san += ["--oracle-json", str(oracle_path), "--prune-rgb-to-oracle", "--apply"]
        else:
            cmd_san += ["--apply"]
        run_subprocess(cmd_san)

        rgb_n = count_rgb_mp4(inner)
        summary["rgb_mp4_after"] = rgb_n
        if (not sensitive_mp4_only) and expected is not None and rgb_n != int(expected):
            summary["status"] = "error_rgb_count_mismatch"
            summary["expected_rgb"] = int(expected)
            return summary

        if in_place:
            out_zip = zip_path.with_suffix(zip_path.suffix + ".sanitized.partial")
            final_zip = zip_path
            backup = zip_path.with_suffix(zip_path.suffix + ".bak")
        else:
            if output_dir is None:
                raise RuntimeError("output_dir required when not --in-place")
            out_dir = (output_dir / lane).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_zip = (out_dir / (zip_name + ".partial")).resolve()
            final_zip = (out_dir / zip_name).resolve()
            backup = None

        cmd_zip = [
            sys.executable,
            str(repack_script),
            "--source-dir",
            str(inner),
            "--output-zip",
            str(out_zip),
        ]
        run_subprocess(cmd_zip)

        if in_place:
            if backup.exists():
                backup.unlink()
            shutil.move(str(zip_path), str(backup))
            shutil.move(str(out_zip), str(final_zip))
            summary["backup"] = str(backup)
        else:
            if final_zip.exists():
                final_zip.unlink()
            shutil.move(str(out_zip), str(final_zip))

        summary["status"] = "ok"
        summary["output_zip"] = str(final_zip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return summary


def main() -> int:
    """作用：批量入口。输出：退出码。"""
    args = parse_args()
    if args.apply and (not args.in_place) and args.output_dir is None:
        print("ERROR: --apply requires --output-dir or --in-place", file=sys.stderr)
        return 1
    if args.in_place and args.output_dir is not None:
        print("ERROR: do not pass both --in-place and --output-dir", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    sanitize_script = here / "sanitize_idpp_public_tree.py"
    repack_script = here.parent.parent / "scripts" / "repack_zip_tree.py"
    if not repack_script.is_file():
        repack_script = Path("/work/guanqi/proj5/Inferring_dynamic_physics/scripts/repack_zip_tree.py")

    upload_script = here / "upload_idpp_hub_zips.py"
    upload_procs: list[subprocess.Popen[Any]] = []
    if args.hf_upload_each:
        if not args.apply:
            print("ERROR: --hf-upload-each requires --apply", file=sys.stderr)
            return 1
        if not hf_token_present():
            print(
                "ERROR: --hf-upload-each requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment",
                file=sys.stderr,
            )
            return 1
        if not upload_script.is_file():
            print(f"ERROR: HF upload script not found: {upload_script}", file=sys.stderr)
            return 1

    try:
        manifest = load_manifest(args.manifest)
        oracle_root = oracle_dir_from_manifest(manifest, args.oracle_json_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        print("ERROR: entries must be a list", file=sys.stderr)
        return 1

    limit = args.limit if args.limit and args.limit > 0 else len(entries)
    failed = False
    for idx, entry in enumerate(entries[:limit]):
        if not isinstance(entry, dict):
            continue
        try:
            summary = process_one_entry(
                entry,
                args.dataset_root.resolve(),
                oracle_root,
                args.output_dir.resolve() if args.output_dir else None,
                bool(args.in_place),
                bool(args.apply),
                sanitize_script,
                repack_script,
                bool(args.sensitive_mp4_only),
            )
        except RuntimeError as exc:
            print(json.dumps({"entry": entry, "error": str(exc)}, ensure_ascii=False))
            failed = True
            continue
        print(json.dumps(summary, ensure_ascii=False))
        st = summary.get("status", "")
        if args.hf_upload_each and st == "ok" and summary.get("output_zip"):
            try:
                upload_procs.append(
                    spawn_hf_upload_process(
                        upload_script,
                        args.dataset_root.resolve(),
                        Path(str(summary["output_zip"])),
                        str(args.hf_repo_id),
                        str(args.hf_prefix),
                    )
                )
            except RuntimeError as exc:
                print(
                    json.dumps(
                        {
                            "hf_upload_spawn_error": str(exc),
                            "output_zip": summary.get("output_zip"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                failed = True
        if st.startswith("error"):
            failed = True

    if args.hf_upload_each and upload_procs:
        try:
            for job_index, return_code in wait_hf_upload_processes(upload_procs):
                if return_code != 0:
                    print(
                        json.dumps(
                            {
                                "hf_upload_failed": True,
                                "job_index": job_index,
                                "returncode": return_code,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    failed = True
        except RuntimeError as exc:
            print(json.dumps({"hf_upload_wait_error": str(exc)}, ensure_ascii=False), flush=True)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
