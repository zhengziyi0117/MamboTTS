#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo '请先运行 ./install.sh --backend rocm' >&2
    exit 1
fi
if [[ "${1:-}" == '--desktop' ]]; then
    shift
    exec .venv/bin/python app.py "$@"
fi
if [[ "${1:-}" == '--api' ]]; then
    shift
    exec .venv/bin/python run_engine.py "$@"
fi
if [[ "${1:-}" == '--web' ]]; then
    shift
fi
exec .venv/bin/python web_app.py "$@"
