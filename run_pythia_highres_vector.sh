#!/bin/bash
set -e

echo "=== 1. PACKING (Pythia-160m rvq-16-8 Vector Only) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/pythia-160m/model.safetensors \
  --out results/pythia-highres-vector.orka \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --group-size 1024 \
  --em-aq-passes 1 \
  --sample-vectors 250000 \
  --max-tensors 10 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/pythia-highres-vector.orka \
  --prompts wiki_prompts.txt \
  --out results/pythia-highres-vector.eval.json \
  --model-dir /home/kai/ai-models/misc/pythia-160m \
  --device cuda \
  --max-prompts 10 \
  --max-length 64

echo -e "\n=== FINAL STATS ==="
cat results/pythia-highres-vector.eval.json | grep -E "loss_delta|perplexity_ratio"
du -sh results/pythia-highres-vector.orka
