#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-runs/$TAG}"
START_SECONDS=$SECONDS
START_TIME=$(date --iso-8601=seconds)

finish() {
    status=$?
    elapsed=$((SECONDS - START_SECONDS))
    printf '[exp2a-smoke] finished=%s elapsed=%02d:%02d:%02d status=%d\n' \
        "$(date --iso-8601=seconds)" "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" "$status"
    printf '[exp2a-smoke] output=%s\n' "$(realpath -m "$RUN_DIR")"
    exit "$status"
}
trap finish EXIT

printf '[exp2a-smoke] started=%s task=place_fan states=1 rollouts_per_model=2 total=4\n' "$START_TIME"
printf '[exp2a-smoke] model_gpu=%s sim_gpu=%s port=%s\n' \
    "${EXP2A_MODEL_GPU:-4}" "${EXP2A_SIM_GPU:-0}" "${EXP2A_PORT:-8860}"

python collect.py \
    --states 1 \
    --rollouts 2 \
    --tasks place_fan \
    --output "$RUN_DIR" \
    --resume

/root/data/miniconda3/envs/robotwin-openwam/bin/python \
    analyze_modes.py "$RUN_DIR" --clusters 2

printf '[exp2a-smoke] summary=%s\n' "$(realpath -m "$RUN_DIR/analysis/summary.md")"
