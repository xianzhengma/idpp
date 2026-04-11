import json
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from svd_paths import get_results_newdata_root
try:
    RESULTS_ROOT = os.environ.get('SVD_E_NEWDATA_RESULTS') or get_results_newdata_root()
except RuntimeError:
    RESULTS_ROOT = os.environ.get('SVD_E_NEWDATA_RESULTS', os.path.join(os.getcwd(), 'results', 'e729_newdata_20260327'))

def load_results(folder_path):
    preds = []
    gts = []
    try:
        if not os.path.isdir(folder_path):
            return (None, None)
        for fn in os.listdir(folder_path):
            if not fn.endswith('.json'):
                continue
            fp = os.path.join(folder_path, fn)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for _k, value in data.items():
                    if isinstance(value, dict) and 'pred' in value and ('gt' in value):
                        preds.append(value['pred'])
                        gts.append(value['gt'])
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
    except OSError:
        return (None, None)
    if not preds:
        return (None, None)
    return (np.array(preds), np.array(gts))

def metric_friction_absolute(preds, gts):
    try:
        scaled = torch.sigmoid(torch.tensor(preds)) * 0.2
        r, _ = pearsonr(gts, scaled.numpy())
        return float(r)
    except (ValueError, FloatingPointError):
        return float('nan')

def metric_friction_relative(preds, gts):
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float('nan')

def metric_viscosity_absolute(preds, gts):
    try:
        scaled = torch.sigmoid(torch.tensor(preds)) * 0.2
        r, _ = pearsonr(gts, scaled.numpy())
        return float(r)
    except (ValueError, FloatingPointError):
        return float('nan')

def metric_viscosity_relative(preds, gts):
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float('nan')

def metric_elasticity_absolute(preds, gts):
    try:
        r, _ = pearsonr(gts, preds)
        return float(r)
    except (ValueError, FloatingPointError):
        return float('nan')

def metric_elasticity_relative(preds, gts):
    try:
        return float(roc_auc_score(gts, preds))
    except ValueError:
        return float('nan')
E_NEWDATA_ROWS = [('v120', 'friction', 'absolute', 'test1', 'e729_nd_v120_test1_fr_abs', metric_friction_absolute), ('v120', 'friction', 'absolute', 'test2', 'e729_nd_v120_test2_fr_abs', metric_friction_absolute), ('v120', 'friction', 'absolute', 'test3', 'e729_nd_v120_test3_fr_abs', metric_friction_absolute), ('v120', 'friction', 'absolute', 'test3*', 'e729_nd_v120_test3star_fr_abs_ft', metric_friction_absolute), ('v121', 'friction', 'relative', 'test1', 'e729_nd_v121_test1_fr_rel', metric_friction_relative), ('v121', 'friction', 'relative', 'test2', 'e729_nd_v121_test2_fr_rel', metric_friction_relative), ('v121', 'friction', 'relative', 'test3', 'e729_nd_v121_test3_fr_rel', metric_friction_relative), ('v121vj_cont7325', 'friction', 'relative', 'test3', 'e729_nd_v121_vjepa_cont7325_test3', metric_friction_relative), ('v121vj_t3resume', 'friction', 'relative', 'test3', 'e729_nd_v121_vjepa_t3resume_last_test3', metric_friction_relative), ('v121', 'friction', 'relative', 'test3*', 'e729_nd_v121_test3star_fr_rel_ft', metric_friction_relative), ('v122', 'viscosity', 'absolute', 'test1', 'e729_nd_v122_test1_vis_abs', metric_viscosity_absolute), ('v122', 'viscosity', 'absolute', 'test2', 'e729_nd_v122_test2_vis_abs', metric_viscosity_absolute), ('v122', 'viscosity', 'absolute', 'test3', 'e729_nd_v122_test3_vis_abs', metric_viscosity_absolute), ('v123', 'viscosity', 'relative', 'test1', 'e729_nd_v123_test1_vis_rel', metric_viscosity_relative), ('v123', 'viscosity', 'relative', 'test2', 'e729_nd_v123_test2_vis_rel', metric_viscosity_relative), ('v123', 'viscosity', 'relative', 'test3', 'e729_nd_v123_test3_vis_rel', metric_viscosity_relative), ('v124', 'elasticity', 'absolute', 'test1', 'e729_nd_v124_test1_el_abs', metric_elasticity_absolute), ('v124', 'elasticity', 'absolute', 'test2', 'e729_nd_v124_test2_el_abs', metric_elasticity_absolute), ('v124', 'elasticity', 'absolute', 'test3', 'e729_nd_v124_test3_el_abs', metric_elasticity_absolute), ('v125', 'elasticity', 'relative', 'test1', 'e729_nd_v125_test1_el_rel', metric_elasticity_relative), ('v125', 'elasticity', 'relative', 'test2', 'e729_nd_v125_test2_el_rel', metric_elasticity_relative), ('v125', 'elasticity', 'relative', 'test3', 'e729_nd_v125_test3_el_rel', metric_elasticity_relative)]

def metric_fn_for_model_test(model, test):
    for m, _p, _k, t, _sub, fn in E_NEWDATA_ROWS:
        if m == model and t == test:
            return fn
    return None

def subdir_for_model_test(model, test):
    for m, _p, _k, t, sub, _fn in E_NEWDATA_ROWS:
        if m == model and t == test:
            return sub
    return None

def metric_display_name(kind):
    if kind == 'relative':
        return 'ROC-AUC'
    return 'Pearson r'

def score_results_folder(folder_path, model, test):
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
    rows = []
    try:
        for model, prop, kind, test, subdir, fn in E_NEWDATA_ROWS:
            path = os.path.join(RESULTS_ROOT, subdir)
            preds, gts = load_results(path)
            metric_name = metric_display_name(kind)
            if preds is None:
                rows.append((model, prop, kind, test, metric_name, 'MISSING', subdir))
                continue
            val = fn(preds, gts)
            rows.append((model, prop, kind, test, metric_name, f'{val:.4f}', subdir))
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
    print('| Model | Property | Task | Test | Metric | E newdata | Folder |')
    print('|---|---|---|---|---|---:|---|')
    for model, prop, kind, test, metric_name, val, subdir in rows:
        print(f'| {model} | {prop} | {kind} | {test} | {metric_name} | {val} | `{subdir}` |')
if __name__ == '__main__':
    main()
