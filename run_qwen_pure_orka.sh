#!/bin/bash
set -e

echo "=== 1. FULL-MODEL PACKING (Pure Orka: SLRQ + EM-AQ) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-pure-slrq.orka \
  --normalization slrq-block \
  --block-scale-size 16 \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 1024 \
  --em-aq-passes 3 \
  --sample-vectors 500000 \
  --codebook-cache /tmp/orka-cache-pure \
  --max-cpu-threads 8 \
  --max-system-ram-gb 16.0 \
  --workload-budget-gb 9.0 \
  --max-gpu-mem-gb 10.0 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATION (Wikitext Accuracy) ==="
.venv/bin/python3 -m orka eval \
  results/qwen-pure-slrq.orka \
  --prompts wiki_prompts.txt \
  --out results/qwen-pure-slrq.eval.json \
  --model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --device cuda \
  --max-prompts 50 \
  --max-length 128

echo -e "\n=== PURE SLRQ SUMMARY ==="
du -sh results/qwen-pure-slrq.orka
cat results/qwen-pure-slrq.eval.json | grep -E "loss_delta|perplexity_ratio"
