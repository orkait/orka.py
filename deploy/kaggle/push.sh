#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kaggle kernels push --path "$SCRIPT_DIR"

KERNEL_ID=$(python3 -c "import json; print(json.load(open('$SCRIPT_DIR/kernel-metadata.json'))['id'])")
echo "Pushed: $KERNEL_ID"
echo "Monitor: kaggle kernels status $KERNEL_ID"
