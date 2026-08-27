#!/usr/bin/env bash
# Stage a self-contained Docker build context for ml-core + noble-ml-core.
# Required because az acr build and the Discovery deploy skill do not support
# Docker BuildKit --build-context.
#
# Usage:
#   # Staged temp dir (manual az acr build):
#   STAGE=$(./stage-build-context.sh /path/to/noble-ml-core)
#   az acr build --registry <acr> --image ml-core:latest "$STAGE"
#
#   # In-place (Discovery deploy skill builds from tools/ml-core/):
#   ./stage-build-context.sh /path/to/noble-ml-core in-place
#   pwsh ... deploy-discovery-agent.ps1 formulation-modeling-agent -BuildMode remote

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOBLE_ML_CORE_PATH="${1:-${NOBLE_ML_CORE_PATH:-}}"
MODE="${2:-}"

if [[ -z "$NOBLE_ML_CORE_PATH" ]]; then
  echo "Usage: $0 /path/to/noble-ml-core [in-place|stage-dir]" >&2
  echo "  or:  NOBLE_ML_CORE_PATH=/path/to/noble-ml-core $0 in-place" >&2
  exit 1
fi

if [[ ! -f "$NOBLE_ML_CORE_PATH/pyproject.toml" ]]; then
  echo "ERROR: noble-ml-core not found at $NOBLE_ML_CORE_PATH" >&2
  exit 1
fi

if [[ "$MODE" == "in-place" ]] || [[ "${STAGE_IN_PLACE:-0}" == "1" ]]; then
  STAGE_DIR="$SCRIPT_DIR"
  NOBLE_DEST="$SCRIPT_DIR/noble-ml-core"
  echo "Staging noble-ml-core in-place for Discovery deploy skill: $NOBLE_DEST" >&2
else
  STAGE_DIR="${MODE:-$(mktemp -d)}"
  NOBLE_DEST="$STAGE_DIR/noble-ml-core"
  mkdir -p "$NOBLE_DEST"
  cp "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/ml_core_utils.py" "$SCRIPT_DIR/entrypoint.py" "$STAGE_DIR/"
  if [[ -f "$SCRIPT_DIR/.dockerignore" ]]; then
    cp "$SCRIPT_DIR/.dockerignore" "$STAGE_DIR/"
  fi
fi

mkdir -p "$NOBLE_DEST"
rsync -a \
  --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='examples/' \
  --exclude='tests/' \
  --exclude='notebooks/' \
  --exclude='wandb/' \
  --exclude='htmlcov/' \
  --exclude='build/' \
  --exclude='cache/' \
  --exclude='ray_tune_results/' \
  --exclude='*.egg-info/' \
  "$NOBLE_ML_CORE_PATH/" "$NOBLE_DEST/"

echo "Staged build context: $STAGE_DIR" >&2
du -sh "$STAGE_DIR" "$NOBLE_DEST" >&2
echo "$STAGE_DIR"
