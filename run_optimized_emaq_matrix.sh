#!/bin/bash
set -e

echo "Starting Optimized EM-AQ GPU Matrix Sweep..."
.venv/bin/python3 -m orka sweep \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/emaq-sweep.json \
  --group-sizes 8 \
  --quant-modes vq-8 rvq-8-8 \
  --codebook-modes per-tensor \
  --normalizations none slrq-block \
  --outlier-frac 0.01 \
  --rotation orthogonal \
  --em-aq-passes 3 \
  --sample-vectors 250000 \
  --codebook-cache /tmp/orka-cache \
  --backend torch \
  --device cuda \
  --max-tensors 5 \
  --iterations 3

echo "Sweep complete. Starting EM-AQ Eval-Sweep..."
.venv/bin/python3 -m orka eval-sweep \
  results/emaq-sweep.json \
  --prompts wiki_prompts.txt \
  --out results/emaq-eval-sweep.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo "Optimized EM-AQ Matrix complete."
