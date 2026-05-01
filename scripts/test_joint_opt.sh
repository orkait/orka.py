#!/bin/bash
set -e

echo "Starting Small Test Run (smollm2-135m, per-tensor, EM-AQ)..."

.venv/bin/python3 -m orka pack \
  /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m/model.safetensors \
  --out results/test-joint-opt.orka \
  --sensitivity-map results/sensitivity_map_wikitext.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m \
  --quant-mode rvq-16-8 \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --max-tensors 10 \
  --backend torch \
  --device cuda \
  --iterations 2

echo "Pack complete. Verifying artifact (Testing Vectorized Decode)..."

.venv/bin/python3 -m orka verify results/test-joint-opt.orka

echo "Test complete successfully."
