#!/bin/bash
set -e

echo "=== 1. PACKING (Pythia-160m Intelligence First) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/pythia-160m/model.safetensors \
  --out results/pythia-max.orka \
  --sensitivity-map results/sensitivity_pythia.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /home/kai/ai-models/misc/pythia-160m \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --outlier-frac 0.01 \
  --em-aq-passes 3 \
  --sample-vectors 250000 \
  --max-tensors 15 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING ==="
.venv/bin/python3 -m orka eval \
  results/pythia-max.orka \
  --prompts wiki_prompts.txt \
  --out results/pythia-max.eval.json \
  --model-dir /home/kai/ai-models/misc/pythia-160m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

cat results/pythia-max.eval.json
