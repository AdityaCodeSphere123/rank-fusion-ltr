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
  --exclude 'a3-rankfusion-LTR.pdf' \
  --exclude 'a3-rankfusion-LTR.tex' \
  --exclude 'pack_submission.sh' \
  --exclude 'submission/model_*.txt' \
  "${ROOT}/" "${DEST}/"

rm -f "$OUT"
(
  cd "$STAGE"
  zip -r -q -X "$OUT" 2023CS50284
)

echo "Wrote ${OUT}"
unzip -l "$OUT" | tail -n 3
