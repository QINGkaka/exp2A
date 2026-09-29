#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TAG="full_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-runs/$TAG}"
python collect.py --output "$RUN_DIR" --resume
/root/data/miniconda3/envs/robotwin-openwam/bin/python analyze_modes.py "$RUN_DIR" --clusters 6
