#!/usr/bin/env bash
# Package the receipt photos + image audit for upload as a Kaggle dataset -> kaggle/tarang-receipts.zip
set -euo pipefail
cd "$(dirname "$0")/.."
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/images"
cp data/raw/images/*.jpg "$stage/images/"
cp reports/image_audit.csv "$stage/"
rm -f kaggle/tarang-receipts.zip
(cd "$stage" && zip -q -r -X "$OLDPWD/kaggle/tarang-receipts.zip" images image_audit.csv)
echo "kaggle/tarang-receipts.zip: $(unzip -l kaggle/tarang-receipts.zip | tail -1)"
