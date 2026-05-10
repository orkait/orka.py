#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 1. Zip the orka package
TMP_ZIP="/tmp/orka_src.zip"
rm -f "$TMP_ZIP"
(cd "$ROOT_DIR" && zip -r "$TMP_ZIP" orka -x "*.pyc" "__pycache__/*")

# 2. Base64 encode the zip
B64_DATA=$(base64 -w 0 "$TMP_ZIP")

# 3. Inject into template
TMP_DEPLOY="/tmp/orka_kaggle_deploy"
rm -rf "$TMP_DEPLOY"
mkdir -p "$TMP_DEPLOY"

cp "$ROOT_DIR/orka_entry_template.py" "$TMP_DEPLOY/orka.py"
sed -i "s|{{ORKA_SOURCE_B64}}|$B64_DATA|g" "$TMP_DEPLOY/orka.py"
cp "$SCRIPT_DIR/kernel-metadata.json" "$TMP_DEPLOY/"

# Fix metadata to point to local orka.py
sed -i 's|../../orka.py|orka.py|g' "$TMP_DEPLOY/kernel-metadata.json"

echo "Pushing self-extracting bundle from $TMP_DEPLOY..."
kaggle kernels push --path "$TMP_DEPLOY"

KERNEL_ID=$(python3 -c "import json; print(json.load(open('$TMP_DEPLOY/kernel-metadata.json'))['id'])")
echo "Pushed: $KERNEL_ID"
echo "Monitor: kaggle kernels status $KERNEL_ID"
