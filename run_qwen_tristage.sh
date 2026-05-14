#!/bin/bash
set -e

echo "=== 1. PACKING (Qwen3 Tri-Stage with FP16 Pillars) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-tristage.orka \
  --sensitivity-map results/qwen_pillars_10pct.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --outlier-frac 0.01 \
  --em-aq-passes 1 \
  --sample-vectors 250000 \
  --max-tensors 30 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/qwen-tristage.orka \
  --prompts wiki_prompts.txt \
  --out results/qwen-tristage.eval.json \
  --model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --device cuda \
  --max-prompts 10 \
  --max-length 128

echo -e "\n=== 3. FINAL COMPRESSION CHECK ==="
du -sh results/qwen-tristage.orka
cat results/qwen-tristage.eval.json | grep -E "loss_delta|perplexity_ratio"
