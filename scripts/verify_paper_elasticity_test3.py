from __future__ import annotations
import argparse
import os
import sys
from typing import Optional
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    from calculate_e729_newdata_dc_scores import load_results, metric_elasticity_absolute, metric_elasticity_relative
    from svd_paths import get_results_newdata_root, get_svd_root
except ImportError as exc:
    raise SystemExit(f'Import failed: {exc}') from exc

def _score_folder(folder: str, kind: str) -> Optional[float]:
    try:
        preds, gts = load_results(folder)
        if preds is None:
            return None
        if kind == 'absolute':
            return float(metric_elasticity_absolute(preds, gts))
        if kind == 'relative':
            return float(metric_elasticity_relative(preds, gts))
        return None
    except (ValueError, TypeError, OSError):
        return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--svd-root', default=None)
    parser.add_argument('--atol', type=float, default=0.03, help='Absolute tolerance vs paper table')
    parser.add_argument('--e-abs-dir', default=None, help='Override E v124 absolute result directory')
    parser.add_argument('--e-rel-dir', default=None, help='Override E v125 relative result directory')
    parser.add_argument('--b-abs-dir', default=None, help='Override B v124 result directory')
    parser.add_argument('--b-rel-dir', default=None, help='Override B v125 result directory')
    args = parser.parse_args()
    root = get_svd_root(args.svd_root)
    os.environ['SVD_ROOT'] = root
    nd_root = get_results_newdata_root()
    e_abs = args.e_abs_dir or os.path.join(nd_root, 'e729_nd_v124_test3_el_abs')
    e_rel = args.e_rel_dir or os.path.join(nd_root, 'e729_nd_v125_test3_el_rel')
    b_abs = args.b_abs_dir or os.path.join(root, 'results', 'mirror_eval', 'v124_vjepa_test3')
    b_rel = args.b_rel_dir or os.path.join(root, 'results', 'mirror_eval', 'v125_vjepa_test3')
    expected = {'E DynamiCrafter abs (Pearson)': (0.1, _score_folder(e_abs, 'absolute')), 'E DynamiCrafter rel (ROC-AUC)': (0.61, _score_folder(e_rel, 'relative')), 'B V-JEPA-2 abs (Pearson)': (0.37, _score_folder(b_abs, 'absolute')), 'B V-JEPA-2 rel (ROC-AUC)': (0.99, _score_folder(b_rel, 'relative'))}
    failed = False
    for name, (paper_val, got) in expected.items():
        if got is None:
            print(f'[MISSING] {name}: no valid JSON (check that result directories exist; paths printed above)')
            failed = True
            continue
        ok = abs(got - paper_val) <= args.atol
        flag = 'OK' if ok else 'FAIL'
        print(f'[{flag}] {name}: paper={paper_val:.2f} got={got:.4f} (atol={args.atol})')
        if not ok:
            failed = True
    if failed:
        sys.exit(1)
    print('All checks passed (within tolerance). See docs/paper_table1_reference.md.')
if __name__ == '__main__':
    main()
