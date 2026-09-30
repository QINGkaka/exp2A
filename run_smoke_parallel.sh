#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_CUDA_INDEX="${MODEL_CUDA_INDEX:-6}"
MODEL_PHYSICAL_GPU="${MODEL_PHYSICAL_GPU:-7}"
SIM_GPU="${SIM_GPU:-0}"
PORT="${PORT:-8861}"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
    echo "[exp2a-smoke] port $PORT is already in use" >&2
    exit 1
fi

model_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$MODEL_PHYSICAL_GPU")"
sim_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$SIM_GPU")"
if (( model_used > 1024 )); then
    echo "[exp2a-smoke] physical model GPU $MODEL_PHYSICAL_GPU is busy (${model_used} MiB)" >&2
    exit 1
fi
if (( sim_used > 1024 )); then
    echo "[exp2a-smoke] simulator GPU $SIM_GPU is busy (${sim_used} MiB)" >&2
    exit 1
fi

echo "[exp2a-smoke] CUDA model index=$MODEL_CUDA_INDEX -> physical GPU $MODEL_PHYSICAL_GPU"
echo "[exp2a-smoke] simulator GPU=$SIM_GPU port=$PORT"

EXP2A_MODEL_GPU="$MODEL_CUDA_INDEX" \
EXP2A_SIM_GPU="$SIM_GPU" \
EXP2A_PORT="$PORT" \
./run_smoke.sh
