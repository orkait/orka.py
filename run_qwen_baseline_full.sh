#!/bin/bash
set -e

echo "=== 1. FULL PACKING (Qwen3 Best Local: Passthrough Vocab) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-baseline-full.orka \
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
  --backend torch \
  --device cuda

echo -e "\n=== 2. FULL EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/qwen-baseline-full.orka \
  --prompts wiki_prompts.txt \
  --out results/qwen-baseline-full.eval.json \
  --model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --device cuda \
  --max-prompts 20 \
  --max-length 128

echo -e "\n=== FINAL STATS ==="
du -sh results/qwen-baseline-full.orka
cat results/qwen-baseline-full.eval.json | grep -E "loss_delta|perplexity_ratio"
