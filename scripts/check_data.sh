#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen problem manifest}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -m hilore.splits --manifest "$SPLIT_MANIFEST"
