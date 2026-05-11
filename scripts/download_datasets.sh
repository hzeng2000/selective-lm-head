#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/repos}"
mkdir -p "$ROOT"

clone_or_update() {
  local url="$1"
  local dest="$2"
  shift 2

  if [ -d "$dest/.git" ]; then
    echo "Updating $dest"
    git -C "$dest" fetch --depth 1 origin
    git -C "$dest" pull --ff-only
  else
    echo "Cloning $url -> $dest"
    git clone "$@" "$url" "$dest"
  fi
}

clone_or_update \
  https://github.com/ShishirPatil/gorilla.git \
  "$ROOT/gorilla" \
  --depth 1 --filter=blob:none --sparse

git -C "$ROOT/gorilla" sparse-checkout set berkeley-function-call-leaderboard

clone_or_update \
  https://github.com/guidance-ai/jsonschemabench.git \
  "$ROOT/jsonschemabench" \
  --depth 1 --filter=blob:none

python scripts/prepare_dataset_manifest.py \
  --bfcl-dir "$ROOT/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data" \
  --jsonschema-dir "$ROOT/jsonschemabench/data" \
  --output data/dataset_manifest.json

