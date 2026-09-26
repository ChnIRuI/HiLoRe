#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen problem manifest}"
: "${UPDATES_DIR:?Set UPDATES_DIR to development actor replays}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new initialization directory}"
: "${RISK_BUDGET:?Set RISK_BUDGET to the trial surrogate risk budget}"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
args=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4
      -m hilore.initialize --split-manifest "$SPLIT_MANIFEST" --updates "$UPDATES_DIR"
      --output "$OUTPUT_DIR" --risk-budget "$RISK_BUDGET"
      --config "${CONFIG:-$ROOT/configs/qwen25_3b_deepmath.json}")
if [[ -n "${MODEL_PATH:-}" ]]; then args+=(--model "$MODEL_PATH"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
"$PYTHON_BIN" -m hilore.splits --manifest "$SPLIT_MANIFEST"
exec "${args[@]}"
