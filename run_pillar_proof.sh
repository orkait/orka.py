#!/bin/bash
set -e

echo "=== 1. PACKING (Pillar Protection Proof) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/smollm-pillars.orka \
  --sensitivity-map results/sensitivity_smollm_pillars.json \
  --normalization slrq-block \
  --quant-mode vq-8 \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --em-aq-passes 1 \
  --max-tensors 15 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/smollm-pillars.orka \
  --prompts wiki_prompts.txt \
  --out results/smollm-pillars.eval.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 10 \
  --max-length 64

echo -e "\n=== 3. SIZE COMPARISON ==="
du -sh results/smollm-pillars.orka
