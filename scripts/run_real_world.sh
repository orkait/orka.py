#!/bin/bash
set -e

MODEL_PATH="/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m/model.safetensors"
OUT_DIR="results/smollm2-135m-optimized.orka"

echo "Starting FULL Model Quantization: SmollM2-135M"
echo "Strategy: RVQ-16-8, AWQ-Block-Max, Orthogonal Rotation, Joint EM-AQ"

# Run the optimized pack
time .venv/bin/python3 -m orka pack \
  "$MODEL_PATH" \
  --out "$OUT_DIR" \
  --sensitivity-map results/sensitivity_map_wikitext.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir "/mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m" \
  --quant-mode rvq-16-8 \
  --codebook-mode family \
  --rotation orthogonal \
  --group-size 8 \
  --backend torch \
  --device cuda \
  --iterations 4

echo "Quantization Complete. Running Verify..."

time .venv/bin/python3 -m orka verify "$OUT_DIR"
