#!/bin/bash
set -e

echo "=== 1. PACKING (Qwen3-0.6B Intelligence First) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-max.orka \
  --sensitivity-map results/sensitivity_qwen.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --outlier-frac 0.01 \
  --em-aq-passes 3 \
  --sample-vectors 250000 \
  --max-tensors 25 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/qwen-max.orka \
  --prompts wiki_prompts.txt \
  --out results/qwen-max.eval.json \
  --model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --device cuda \
  --max-prompts 5 \
  --max-length 128

cat results/qwen-max.eval.json
