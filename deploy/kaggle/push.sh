#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create a bundle directory
TMP_DEPLOY="/tmp/orka_kaggle_deploy"
rm -rf "$TMP_DEPLOY"
mkdir -p "$TMP_DEPLOY"

# Copy package and entry
cp -r "$ROOT_DIR/orka" "$TMP_DEPLOY/"
cp "$ROOT_DIR/orka.py" "$TMP_DEPLOY/"
cp "$SCRIPT_DIR/kernel-metadata.json" "$TMP_DEPLOY/"

# Fix metadata to point to local orka.py
sed -i 's|../../orka.py|orka.py|g' "$TMP_DEPLOY/kernel-metadata.json"

echo "Pushing bundle from $TMP_DEPLOY..."
kaggle kernels push --path "$TMP_DEPLOY"

KERNEL_ID=$(python3 -c "import json; print(json.load(open('$TMP_DEPLOY/kernel-metadata.json'))['id'])")
echo "Pushed: $KERNEL_ID"
echo "Monitor: kaggle kernels status $KERNEL_ID"
