#!/bin/bash
set -e

echo "Starting Baseline Test (smollm2-135m, block-max, vq-4)..."

.venv/bin/python3 -m orka pack \
  /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m/model.safetensors \
  --out results/test-baseline.orka \
  --normalization block-max \
  --block-scale-size 16 \
  --quant-mode vq-4 \
  --codebook-mode per-tensor \
  --group-size 1 \
  --outlier-frac 0.01 \
  --max-tensors 5 \
  --backend torch \
  --device cuda \
  --iterations 2

echo "Pack complete. Verifying artifact..."
.venv/bin/python3 -m orka verify results/test-baseline.orka

echo "Evaluating artifact (fast eval)..."
.venv/bin/python3 -m orka eval \
  results/test-baseline.orka \
  --prompts wiki_prompts.txt \
  --out results/eval-baseline.json \
  --model-dir /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo "Baseline Test complete successfully."