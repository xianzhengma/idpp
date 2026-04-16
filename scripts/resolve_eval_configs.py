from __future__ import annotations
import argparse
import os
import sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    from svd_paths import get_checkpoint_path, get_results_newdata_root, get_svd_root
except ImportError as exc:
    raise SystemExit(f'Cannot import svd_paths: {exc}') from exc

def _write_resolved(template_name: str, mapping: dict, out_dir: str) -> str:
    try:
        template_path = os.path.join(_REPO, 'configs', 'reproduce', 'templates', template_name)
        with open(template_path, 'r', encoding='utf-8') as handle:
            text = handle.read()
        for key, value in mapping.items():
            text = text.replace(f'__{key}__', value.replace('\\', '/'))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, template_name.replace('.template', ''))
        with open(out_path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return out_path
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f'Failed to process template {template_name}: {exc}') from exc

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate resolved eval YAMLs under SVD_ROOT.')
    parser.add_argument('--svd-root', default=None, help='Override SVD_ROOT (default from env or ./asset_root)')
    parser.add_argument(
        '--generated-out-dir',
        default=None,
        help='Optional directory for resolved YAMLs (default: configs/reproduce/generated). '
        'Use a per-experiment subdir when running multiple SVD_ROOT ablations so configs are not overwritten.',
    )
    args = parser.parse_args()
    try:
        root = get_svd_root(args.svd_root)
        os.environ['SVD_ROOT'] = root
        results_nd = get_results_newdata_root()
        ckpt_e124 = get_checkpoint_path('training_1024_v124_7.29/checkpoints/epoch=4-step=2625.ckpt')
        ckpt_e125 = get_checkpoint_path('training_1024_v125_7.29/checkpoints/epoch=3-step=500.ckpt')
        ckpt_b124 = get_checkpoint_path('training_1024_v124_vjepa_norm/checkpoints/epoch=21-step=8375.ckpt')
        ckpt_b125 = get_checkpoint_path('training_1024_v125_vjepa_norm/checkpoints/epoch=126-step=15875.ckpt')
        data_abs = os.path.join(root, 'data', 'elasticity_absolute_test3')
        data_rel = os.path.join(root, 'data', 'elasticity_relative_test3')
        lists_abs = os.path.join(root, 'data', 'lists', 'elasticity_absolute_test3.json')
        lists_rel = os.path.join(root, 'data', 'lists', 'elasticity_relative_test3.json')
        out_e124 = os.path.join(results_nd, 'e729_nd_v124_test3_el_abs')
        out_e125 = os.path.join(results_nd, 'e729_nd_v125_test3_el_rel')
        out_b124 = os.path.join(root, 'results', 'mirror_eval', 'v124_vjepa_test3')
        out_b125 = os.path.join(root, 'results', 'mirror_eval', 'v125_vjepa_test3')
        if args.generated_out_dir:
            out_dir = os.path.abspath(os.path.expanduser(args.generated_out_dir))
        else:
            out_dir = os.path.join(_REPO, 'configs', 'reproduce', 'generated')
        templates = [('config_e729_nd_v124_test3_el_abs.yaml.template', {'CKPT_E124': ckpt_e124, 'SAVE_E124': out_e124, 'DATA_ELASTICITY_ABS': data_abs, 'LIST_ELASTICITY_ABS_T3': lists_abs}), ('config_e729_nd_v125_test3_el_rel.yaml.template', {'CKPT_E125': ckpt_e125, 'SAVE_E125': out_e125, 'DATA_ELASTICITY_REL': data_rel, 'LIST_ELASTICITY_REL_T3': lists_rel}), ('config_v124_vjepa_test3.yaml.template', {'CKPT_B124': ckpt_b124, 'SAVE_B124': out_b124, 'DATA_ELASTICITY_ABS': data_abs, 'LIST_ELASTICITY_ABS_T3': lists_abs}), ('config_v125_vjepa_test3.yaml.template', {'CKPT_B125': ckpt_b125, 'SAVE_B125': out_b125, 'DATA_ELASTICITY_REL': data_rel, 'LIST_ELASTICITY_REL_T3': lists_rel})]
        written = []
        for name, mapping in templates:
            written.append(_write_resolved(name, mapping, out_dir))
        print('SVD_ROOT=', root)
        for path in written:
            print('Wrote', path)
    except (OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
if __name__ == '__main__':
    main()
