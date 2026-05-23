#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_SLUG="orka-compiler-core-v2"
TMP_DS="/tmp/orka_dataset_bundle"

echo "--- Step 1: Versioning Code in Kaggle Dataset ---"
rm -rf "$TMP_DS"
mkdir -p "$TMP_DS"

# Create a zip of the package
(cd "$ROOT_DIR" && zip -r "$TMP_DS/orka_core.zip" orka -x "*.pyc" "__pycache__/*")

# Always init metadata to ensure it exists
cd "$TMP_DS"
kaggle datasets init -p .
sed -i "s/INSERT_TITLE_HERE/$DATASET_SLUG/g" dataset-metadata.json
sed -i "s/INSERT_SLUG_HERE/$DATASET_SLUG/g" dataset-metadata.json

if ! kaggle datasets status "superkaiii/$DATASET_SLUG" > /dev/null 2>&1; then
    echo "Creating new private dataset..."
    kaggle datasets create -p .
else
    echo "Updating existing dataset version..."
    kaggle datasets version -p . -m "Update Orka Core"
fi

echo "--- Step 2: Pushing Kernel ---"
TMP_DEPLOY="/tmp/orka_kernel_deploy"
rm -rf "$TMP_DEPLOY"
mkdir -p "$TMP_DEPLOY"

cp "$SCRIPT_DIR/orka_smol_kaggle.py" "$TMP_DEPLOY/orka.py"
cp "$SCRIPT_DIR/kernel-metadata.json" "$TMP_DEPLOY/"

# Fix metadata to point to local orka.py
sed -i 's|../../orka.py|orka.py|g' "$TMP_DEPLOY/kernel-metadata.json"

cd "$TMP_DEPLOY"
kaggle kernels push --path .

KERNEL_ID=$(python3 -c "import json; print(json.load(open('kernel-metadata.json'))['id'])")
echo "Pushed: $KERNEL_ID"
echo "Monitor: kaggle kernels status $KERNEL_ID"
