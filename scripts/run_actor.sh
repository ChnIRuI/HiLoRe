#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
METHOD=${1:-}
if [[ $# != 1 ]] || [[ "$METHOD" != gc && "$METHOD" != hilore ]]; then
  echo "Usage: bash scripts/run_actor.sh gc|hilore" >&2
  exit 2
fi
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen problem manifest}"
: "${UPDATES_DIR:?Set UPDATES_DIR to the actor replay directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
args=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4
      -m hilore.train --method "$METHOD" --split-manifest "$SPLIT_MANIFEST"
      --updates "$UPDATES_DIR" --output "$OUTPUT_DIR"
      --config "${CONFIG:-$ROOT/configs/qwen25_3b_deepmath.json}")
if [[ -n "${MODEL_PATH:-}" ]]; then args+=(--model "$MODEL_PATH"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ "${DEVELOPMENT_PROFILE:-0}" == 1 ]]; then args+=(--development-profile); fi
if [[ "$METHOD" == hilore ]]; then
  : "${PROFILE_PATH:?HiLoRe requires PROFILE_PATH from measured joint validation}"
  : "${RISK_BUDGET:?Set RISK_BUDGET to the frozen surrogate budget}"
  args+=(--profile "$PROFILE_PATH" --risk-budget "$RISK_BUDGET")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi
"$PYTHON_BIN" -m hilore.splits --manifest "$SPLIT_MANIFEST"
exec "${args[@]}"
