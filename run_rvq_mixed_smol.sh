#!/bin/bash
set -e

echo "Starting rvq-mixed Smol Run (20 Tensors)..."
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/rvq-mixed-smol.orka \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --em-aq-passes 1 \
  --sample-vectors 250000 \
  --max-tensors 20 \
  --backend torch \
  --device cuda

echo -e "\n=== EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/rvq-mixed-smol.orka \
  --prompts wiki_prompts.txt \
  --out results/rvq-mixed-smol.eval.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo -e "\n=== SIZE CHECK ==="
du -sh results/rvq-mixed-smol.orka
