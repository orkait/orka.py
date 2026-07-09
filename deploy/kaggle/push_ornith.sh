#!/usr/bin/env bash
# Push the orka package as the core dataset, then push the Ornith-9B pack kernel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_SLUG="orka-compiler-core-v2"
TMP_DS="/tmp/orka_dataset_bundle"

echo "--- Step 1: versioning orka package dataset ---"
rm -rf "$TMP_DS"; mkdir -p "$TMP_DS"
(cd "$ROOT_DIR" && zip -rq "$TMP_DS/orka_core.zip" orka -x "*.pyc" "__pycache__/*")
cd "$TMP_DS"
kaggle datasets init -p . >/dev/null 2>&1 || true
sed -i "s/INSERT_TITLE_HERE/$DATASET_SLUG/g; s/INSERT_SLUG_HERE/$DATASET_SLUG/g" dataset-metadata.json
if ! kaggle datasets status "superkaiii/$DATASET_SLUG" >/dev/null 2>&1; then
    kaggle datasets create -p .
else
    kaggle datasets version -p . -m "ornith: 9b pack kernel"
fi

echo "--- Step 2: pushing Ornith pack kernel ---"
TMP_DEPLOY="/tmp/orka_ornith_deploy"
rm -rf "$TMP_DEPLOY"; mkdir -p "$TMP_DEPLOY"
cp "$SCRIPT_DIR/orka_ornith9b_kaggle.py" "$TMP_DEPLOY/"
cp "$SCRIPT_DIR/kernel-metadata-ornith.json" "$TMP_DEPLOY/kernel-metadata.json"
cd "$TMP_DEPLOY"
kaggle kernels push --path .
echo "Pushed: superkaiii/orka-ornith-1-0-9b-pack"
echo "Monitor: kaggle kernels status superkaiii/orka-ornith-1-0-9b-pack"
