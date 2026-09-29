#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENWAM_DIR="${OPENWAM_DIR:-/root/data/robot/OpenWAM}"
PATCH_FILE="$SCRIPT_DIR/patches/openwam-exp2a.patch"

if [[ ! -d "$OPENWAM_DIR/.git" ]]; then
    printf 'OpenWAM repository not found: %s\n' "$OPENWAM_DIR" >&2
    exit 1
fi

if git -C "$OPENWAM_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    printf 'Experiment 2A OpenWAM patch is already applied.\n'
    exit 0
fi

git -C "$OPENWAM_DIR" apply --check "$PATCH_FILE"
git -C "$OPENWAM_DIR" apply "$PATCH_FILE"
printf 'Applied Experiment 2A integration patch to %s\n' "$OPENWAM_DIR"
