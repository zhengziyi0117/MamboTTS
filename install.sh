#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
    echo '请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
# Arch 的系统 Python 更新很快；独立 Python 3.10 避免旧推理依赖不兼容。
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.10 .venv
fi
exec .venv/bin/python install_linux.py "$@"
