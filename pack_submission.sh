#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${HOME}/Desktop/2023CS50284.zip"
STAGE="$(mktemp -d)"
DEST="${STAGE}/2023CS50284"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$DEST"

rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'data/train/' \
  --exclude 'data/dev/' \
  --exclude 'data/ltr_data/' \
  --exclude 'pack_submission.sh' \
  --exclude 'submission/model_*.txt' \
  --exclude 'experiments/' \
  --exclude 'runs/' \
  --include 'scripts/cv_lambdamart.py' \
  --include 'scripts/cv_sweep.py' \
  --include 'scripts/data_cache.py' \
  --include 'scripts/fusion_eval.py' \
  --include 'scripts/lambdamart_utils.py' \
  --include 'scripts/make_toy_data.py' \
  --include 'scripts/prepare_mslr.py' \
  --include 'scripts/run_recipe.py' \
  --include 'scripts/smoke_test.sh' \
  --include 'scripts/train_model.py' \
  --include 'scripts/train_neural.py' \
  --exclude 'scripts/*' \
  "${ROOT}/" "${DEST}/"

rm -f "$OUT"
(
  cd "$STAGE"
  zip -r -q -X "$OUT" 2023CS50284
)

echo "Wrote ${OUT}"
unzip -l "$OUT" | tail -n 3
