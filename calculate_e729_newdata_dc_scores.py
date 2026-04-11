#!/usr/bin/env python3
"""
计算 E 版 7.29 **新数据** batch（e729_newdata_20260327）下各结果目录的主指标（含 V-JEPA v121 续训 Test3 两行）。

**本文件为续训 / e729 newdata 评测的单一算分来源**：`load_results`、各 `metric_*`、`E_NEWDATA_ROWS`、`RESULTS_ROOT`
应与所有监控、扫点、delta 脚本一致；其它脚本只应 **import 并调用** 本模块函数，勿再复制粘贴算分实现。

与 `calculate_e_729_dc_scores.py` 的关系：摩擦/粘度 **绝对** = sigmoid(pred)×0.2 后与 gt 的 Pearson *r*；**相对** = ROC-AUC；
弹性 **绝对** = raw Pearson *r*；**相对** = ROC-AUC。公式相同；差别在于 **结果根目录与表格行**：
本文件使用 `RESULTS_ROOT` + `e729_nd_*` 子目录；`calculate_e_729_dc_scores.py` 使用 scatter 根下 `7.29_result_*` 等历史目录。
"""

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from svd_paths import get_results_newdata_root

# 优先 `SVD_ROOT/results/e729_newdata_20260327`；可用 `SVD_E_NEWDATA_RESULTS` 覆盖
try:
    RESULTS_ROOT = os.environ.get("SVD_E_NEWDATA_RESULTS") or get_results_newdata_root()
except RuntimeError:
    RESULTS_ROOT = os.environ.get(
        "SVD_E_NEWDATA_RESULTS",
        os.path.join(os.getcwd(), "results", "e729_newdata_20260327"),
    )


def load_results(folder_path):
    """从目录加载所有 JSON 的 pred/gt；失败返回 (None, None)。"""
    preds = []
    gts = []
    try:
        if not os.path.isdir(folder_path):
            return None, None
        for fn in os.listdir(folder_path):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(folder_path, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for _k, value in data.items():
                    if isinstance(value, dict) and "pred" in value and "gt" in value:
                        preds.append(value["pred"])
                        gts.append(value["gt"])
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
    except OSError:
        return None, None
    if not preds:
        return None, None
    return np.array(preds), np.array(gts)


def metric_friction_absolute(preds, gts):
    """Friction 绝对：sigmoid(pred)*0.2 后 Pearson r。"""
    try:
        scaled = torch.sigmoid(torch.tensor(preds)) * 0.2
        r, _ = pearsonr(gts, scaled.numpy())
        return float(r)
    except (ValueError, FloatingPointError):
        return float("nan")


def metric_friction_relative(preds, gts):
    """Friction 相对：ROC-AUC。"""
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float("nan")


def metric_viscosity_absolute(preds, gts):
    """Viscosity 绝对：sigmoid*0.2 后 Pearson r。"""
    try:
        scaled = torch.sigmoid(torch.tensor(preds)) * 0.2
        r, _ = pearsonr(gts, scaled.numpy())
        return float(r)
    except (ValueError, FloatingPointError):
        return float("nan")


def metric_viscosity_relative(preds, gts):
    """Viscosity 相对：ROC-AUC。"""
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float("nan")


def metric_elasticity_absolute(preds, gts):
    """Elasticity 绝对：raw Pearson r。"""
    try:
        r, _ = pearsonr(gts, preds)
        return float(r)
    except (ValueError, FloatingPointError):
        return float("nan")


def metric_elasticity_relative(preds, gts):
    """Elasticity 相对：ROC-AUC。"""
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float("nan")


# 行表：与 config_e729_nd_* 的 save_result_dir 子目录名一致（随 yaml 增加而扩展）
E_NEWDATA_ROWS = [
    ("v120", "friction", "absolute", "test1", "e729_nd_v120_test1_fr_abs", metric_friction_absolute),
    ("v120", "friction", "absolute", "test2", "e729_nd_v120_test2_fr_abs", metric_friction_absolute),
    ("v120", "friction", "absolute", "test3", "e729_nd_v120_test3_fr_abs", metric_friction_absolute),
    ("v120", "friction", "absolute", "test3*", "e729_nd_v120_test3star_fr_abs_ft", metric_friction_absolute),
    ("v121", "friction", "relative", "test1", "e729_nd_v121_test1_fr_rel", metric_friction_relative),
    ("v121", "friction", "relative", "test2", "e729_nd_v121_test2_fr_rel", metric_friction_relative),
    ("v121", "friction", "relative", "test3", "e729_nd_v121_test3_fr_rel", metric_friction_relative),
    (
        "v121vj_cont7325",
        "friction",
        "relative",
        "test3",
        "e729_nd_v121_vjepa_cont7325_test3",
        metric_friction_relative,
    ),
    (
        "v121vj_t3resume",
        "friction",
        "relative",
        "test3",
        "e729_nd_v121_vjepa_t3resume_last_test3",
        metric_friction_relative,
    ),
    ("v121", "friction", "relative", "test3*", "e729_nd_v121_test3star_fr_rel_ft", metric_friction_relative),
    ("v122", "viscosity", "absolute", "test1", "e729_nd_v122_test1_vis_abs", metric_viscosity_absolute),
    ("v122", "viscosity", "absolute", "test2", "e729_nd_v122_test2_vis_abs", metric_viscosity_absolute),
    ("v122", "viscosity", "absolute", "test3", "e729_nd_v122_test3_vis_abs", metric_viscosity_absolute),
    ("v123", "viscosity", "relative", "test1", "e729_nd_v123_test1_vis_rel", metric_viscosity_relative),
    ("v123", "viscosity", "relative", "test2", "e729_nd_v123_test2_vis_rel", metric_viscosity_relative),
    ("v123", "viscosity", "relative", "test3", "e729_nd_v123_test3_vis_rel", metric_viscosity_relative),
    ("v124", "elasticity", "absolute", "test1", "e729_nd_v124_test1_el_abs", metric_elasticity_absolute),
    ("v124", "elasticity", "absolute", "test2", "e729_nd_v124_test2_el_abs", metric_elasticity_absolute),
    ("v124", "elasticity", "absolute", "test3", "e729_nd_v124_test3_el_abs", metric_elasticity_absolute),
    ("v125", "elasticity", "relative", "test1", "e729_nd_v125_test1_el_rel", metric_elasticity_relative),
    ("v125", "elasticity", "relative", "test2", "e729_nd_v125_test2_el_rel", metric_elasticity_relative),
    ("v125", "elasticity", "relative", "test3", "e729_nd_v125_test3_el_rel", metric_elasticity_relative),
]


def metric_fn_for_model_test(model, test):
    """
    输入: model（如 v120）、test（如 test1、test3*）。
    输出: `E_NEWDATA_ROWS` 中对应指标函数；无匹配时为 None。
    作用: 供监控/扫点脚本统一取用，避免重复实现 metric_*。
    """
    for m, _p, _k, t, _sub, fn in E_NEWDATA_ROWS:
        if m == model and t == test:
            return fn
    return None


def subdir_for_model_test(model, test):
    """
    输入: model、test。
    输出: `RESULTS_ROOT` 下基线子目录名（如 e729_nd_v122_test1_vis_abs）；无匹配时为 None。
    作用: 与 `metric_fn_for_model_test` 配对，解析 baseline 路径。
    """
    for m, _p, _k, t, sub, _fn in E_NEWDATA_ROWS:
        if m == model and t == test:
            return sub
    return None


def metric_display_name(kind):
    """
    输入: kind（absolute 或 relative，与表格「Task」列一致）。
    输出: 人类可读的指标名；relative 一律为 ROC-AUC，absolute 为 Pearson r（与 metric_* 实现一致）。
    作用: Markdown 表头区分，避免把 friction relative 的 AUC 误称为 Pearson。
    """
    if kind == "relative":
        return "ROC-AUC"
    return "Pearson r"


def score_results_folder(folder_path, model, test):
    """
    输入: 含 per-video JSON（pred/gt）的目录路径、model、test。
    输出: 主指标标量（可为 nan）；未知 model/test、无 JSON 或计算失败时为 None。
    作用: 单目录算分，一条路径完成 load + metric。
    """
    fn = metric_fn_for_model_test(model, test)
    if fn is None:
        return None
    preds, gts = load_results(folder_path)
    if preds is None:
        return None
    try:
        return float(fn(preds, gts))
    except (ValueError, FloatingPointError, TypeError):
        return None


def main():
    """打印 Markdown 表；relative 行为 ROC-AUC，absolute 行为 Pearson r（列「Metric」标明）。"""
    rows = []
    try:
        for model, prop, kind, test, subdir, fn in E_NEWDATA_ROWS:
            path = os.path.join(RESULTS_ROOT, subdir)
            preds, gts = load_results(path)
            metric_name = metric_display_name(kind)
            if preds is None:
                rows.append((model, prop, kind, test, metric_name, "MISSING", subdir))
                continue
            val = fn(preds, gts)
            rows.append((model, prop, kind, test, metric_name, f"{val:.4f}", subdir))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("| Model | Property | Task | Test | Metric | E newdata | Folder |")
    print("|---|---|---|---|---|---:|---|")
    for model, prop, kind, test, metric_name, val, subdir in rows:
        print(f"| {model} | {prop} | {kind} | {test} | {metric_name} | {val} | `{subdir}` |")


if __name__ == "__main__":
    main()
