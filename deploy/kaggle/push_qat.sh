#!/usr/bin/env bash
# Push the orka package (with qat) as the core dataset, then push the QAT kernel.
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
    kaggle datasets version -p . -m "qat: VQ-QAT module"
fi

echo "--- Step 2: pushing QAT kernel ---"
TMP_DEPLOY="/tmp/orka_qat_deploy"
rm -rf "$TMP_DEPLOY"; mkdir -p "$TMP_DEPLOY"
cp "$SCRIPT_DIR/orka_qat_kaggle.py" "$TMP_DEPLOY/"
cp "$SCRIPT_DIR/kernel-qat-metadata.json" "$TMP_DEPLOY/kernel-metadata.json"
cd "$TMP_DEPLOY"
kaggle kernels push --path .
echo "Pushed: superkaiii/orka-qat-2bpw"
echo "Monitor: kaggle kernels status superkaiii/orka-qat-2bpw"
