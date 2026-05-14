#!/bin/bash
set -e

echo "=== 1. PACKING (Pythia-160m Granular Vocab Fix) ==="
# We use group-size 1024 for logic layers, but Orka will now 
# automatically drop to group-size 8 for the vocabulary.
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/pythia-160m/model.safetensors \
  --out results/pythia-granular.orka \
  --quant-mode vq-8 \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --group-size 1024 \
  --em-aq-passes 1 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/pythia-granular.orka \
  --prompts wiki_prompts.txt \
  --out results/pythia-granular.eval.json \
  --model-dir /home/kai/ai-models/misc/pythia-160m \
  --device cuda \
  --max-prompts 20 \
  --max-length 128

echo -e "\n=== FINAL STATS ==="
du -sh results/pythia-granular.orka
cat results/pythia-granular.eval.json | grep -E "loss_delta|perplexity_ratio"
