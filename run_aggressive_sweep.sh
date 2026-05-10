#!/bin/bash
set -e

echo "Starting Aggressive Sweep (MSE/Cosine generation)..."
.venv/bin/python3 -m orka sweep \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/aggressive-sweep.json \
  --group-sizes 8 \
  --quant-modes vq-8 vq-4 \
  --codebook-modes per-tensor \
  --normalizations none slrq-block block-max \
  --outlier-frac 0.01 \
  --rotation orthogonal \
  --backend torch \
  --device cuda \
  --max-tensors 10 \
  --iterations 3

echo "Sweep complete. Starting Eval-Sweep (Perplexity/Loss generation)..."
.venv/bin/python3 -m orka eval-sweep \
  results/aggressive-sweep.json \
  --prompts wiki_prompts.txt \
  --out results/aggressive-eval-sweep.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo "Aggressive Eval-Sweep complete."
