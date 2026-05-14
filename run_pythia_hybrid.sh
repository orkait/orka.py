#!/bin/bash
set -e

echo "=== 1. PACKING (Pythia-160m Scalar-Vector Hybrid) ==="
# We use rvq-mixed which I updated to [12, s4, s4] for embeddings.
# This is the "True" fix for vocabulary.
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/pythia-160m/model.safetensors \
  --out results/pythia-hybrid.orka \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --group-size 1024 \
  --em-aq-passes 1 \
  --max-tensors 10 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/pythia-hybrid.orka \
  --prompts wiki_prompts.txt \
  --out results/pythia-hybrid.eval.json \
  --model-dir /home/kai/ai-models/misc/pythia-160m \
  --device cuda \
  --max-prompts 10 \
  --max-length 64

echo -e "\n=== FINAL STATS ==="
cat results/pythia-hybrid.eval.json | grep -E "loss_delta|perplexity_ratio"
du -sh results/pythia-hybrid.orka
