#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "   ORKA END-TO-END PIPELINE INTEGRATION TEST"
echo "=========================================================="

# Setup paths
ORKA_DIR="/tmp/e2e_smol.orka"
GGUF_NORMAL="/tmp/e2e_smol.gguf"
GGUF_OBFUSCATED="/tmp/e2e_smol_obfuscated.gguf"
SOURCE_MODEL="/home/kai/ai-models/misc/orka-smollm2-135m"

# Cleanup from any previous runs
rm -rf "$ORKA_DIR" "$GGUF_NORMAL" "$GGUF_OBFUSCATED"

echo -e "\n--- Step 1: Pack Checkpoint (First 5 Tensors) ---"
.venv/bin/python3 -m orka pack \
  "$SOURCE_MODEL/model.safetensors" \
  --out "$ORKA_DIR" \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --em-aq-passes 1 \
  --sample-vectors 50000 \
  --max-tensors 5 \
  --backend torch \
  --device cuda

echo -e "\n--- Step 2: Convert to Normal GGUF ---"
.venv/bin/python3 tools/orka_to_gguf.py \
  "$ORKA_DIR" \
  -o "$GGUF_NORMAL"

echo -e "\n--- Step 3: Convert to Obfuscated GGUF ---"
.venv/bin/python3 tools/orka_to_gguf.py \
  "$ORKA_DIR" \
  -o "$GGUF_OBFUSCATED" \
  --obfuscate

echo -e "\n--- Step 4: Verify Normal GGUF Correctness ---"
.venv/bin/python3 tools/verify_gguf.py \
  "$ORKA_DIR" \
  "$GGUF_NORMAL"

echo -e "\n--- Step 5: Verify Obfuscated GGUF Correctness ---"
.venv/bin/python3 tools/verify_gguf.py \
  "$ORKA_DIR" \
  "$GGUF_OBFUSCATED" \
  --obfuscate

# Clean up
echo -e "\n--- Step 6: Cleanup Temp Outputs ---"
rm -rf "$ORKA_DIR" "$GGUF_NORMAL" "$GGUF_OBFUSCATED"

echo "=========================================================="
echo "   SUCCESS: End-to-End Pipeline Completed Flawlessly!"
echo "=========================================================="
