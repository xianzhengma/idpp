#!/usr/bin/env bash
# 作用：在已登录 HF 或已设置 HF_TOKEN 时，上传 IDPP-Data 的 video_zips 归档。
# 输入：可选参数透传给 upload_idpp_hub_zips.py（如 --dry-run、--limit N）。
# 输出：上传进度与结果。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
exec python3 "${SCRIPT_DIR}/upload_idpp_hub_zips.py" "$@"
