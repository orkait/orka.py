#!/bin/bash
set -e

echo "=== 1. PACKING (Intelligence First) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/intelligence-max.orka \
  --sensitivity-map results/sensitivity_map_wikitext.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --outlier-frac 0.01 \
  --em-aq-passes 3 \
  --sample-vectors 250000 \
  --max-tensors 10 \
  --backend torch \
  --device cuda

echo -e "\n=== 2. EVALUATING (Cognitive Retention) ==="
.venv/bin/python3 -m orka eval \
  results/intelligence-max.orka \
  --prompts wiki_prompts.txt \
  --out results/intelligence-max.eval.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

cat results/intelligence-max.eval.json
