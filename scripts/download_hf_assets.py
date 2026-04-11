from __future__ import annotations
import argparse
import os
import sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

def _download_repo(repo_id: str, local_dir: str, repo_type: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install huggingface_hub: pip install huggingface_hub') from exc
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type=repo_type, local_dir_use_symlinks=False)
    except Exception as exc:
        raise RuntimeError(f'Download failed for {repo_id}: {exc}') from exc

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--svd-root', default=os.environ.get('SVD_ROOT', os.path.join(os.getcwd(), 'asset_root')), help='Root directory for checkpoints and data')
    parser.add_argument('--checkpoints-repo', default=os.environ.get('HF_SVD_CHECKPOINTS_REPO', 'YOUR_ORG/svd-physics-checkpoints'), help='Hugging Face model repo ID (weights)')
    parser.add_argument('--data-repo', default=os.environ.get('HF_SVD_DATA_REPO', 'YOUR_ORG/svd-physics-eval-data'), help='Hugging Face dataset repo ID (eval videos and list json)')
    args = parser.parse_args()
    root = os.path.abspath(args.svd_root)
    os.environ['SVD_ROOT'] = root
    ckpt_dir = os.path.join(root, 'checkpoints')
    data_dir = os.path.join(root, 'data_hf')
    try:
        os.makedirs(root, exist_ok=True)
        if 'YOUR_ORG' in args.checkpoints_repo:
            print('Skipping checkpoint download: set HF_SVD_CHECKPOINTS_REPO or --checkpoints-repo to a real Hub ID.', file=sys.stderr)
        else:
            _download_repo(args.checkpoints_repo, ckpt_dir, 'model')
        if 'YOUR_ORG' in args.data_repo:
            print('Skipping dataset download: set HF_SVD_DATA_REPO or --data-repo to a real Hub ID.', file=sys.stderr)
        else:
            _download_repo(args.data_repo, data_dir, 'dataset')
        print('SVD_ROOT=', root)
        print('If you placed files manually, adjust paths in configs/paths.example.yaml.')
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
if __name__ == '__main__':
    main()
