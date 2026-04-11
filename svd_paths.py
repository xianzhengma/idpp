from __future__ import annotations
import os
from typing import Optional

def get_svd_root(explicit_root: Optional[str]=None) -> str:
    try:
        if explicit_root is not None:
            return os.path.abspath(os.path.expanduser(explicit_root))
        env_root = os.environ.get('SVD_ROOT', '').strip()
        if env_root:
            return os.path.abspath(os.path.expanduser(env_root))
        return os.path.abspath(os.path.join(os.getcwd(), 'asset_root'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Failed to resolve SVD_ROOT: {exc}') from exc

def get_results_newdata_root(root: Optional[str]=None) -> str:
    try:
        base = get_svd_root(root)
        return os.path.join(base, 'results', 'e729_newdata_20260327')
    except (OSError, TypeError) as exc:
        raise RuntimeError(f'Failed to build results path: {exc}') from exc

def get_checkpoint_path(relative_under_checkpoints: str, root: Optional[str]=None) -> str:
    try:
        base = get_svd_root(root)
        return os.path.join(base, 'checkpoints', relative_under_checkpoints)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f'Failed to build checkpoint path: {exc}') from exc
