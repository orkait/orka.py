#!/bin/bash
set -e

# Use 30% Pillar Protection (14,745 tokens)
PCT=30
echo -e "\n=== RUNNING TRI-STAGE PILLAR PROOF: ${PCT}% PROTECTION (FP16) ==="

# 1. Create specific sensitivity map
.venv/bin/python3 -c "
import json
with open('results/smollm_pillar_variants.json') as f:
    vars = json.load(f)
with open('results/sensitivity_tristage.json', 'w') as f:
    # We pass the top_tokens so the new FP16 pillar logic picks them up
    json.dump({'top_tokens': vars['30pct'], 'layers': []}, f)
"

# 2. Pack (Logic layers to 3 bpw, Vocab to 30% FP16 + 70% 8-bit)
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/smollm-tristage.orka \
  --sensitivity-map results/sensitivity_tristage.json \
  --normalization slrq-block \
  --quant-mode vq-8 \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 8 \
  --em-aq-passes 1 \
  --max-tensors 20 \
  --backend torch \
  --device cuda

# 3. Eval
.venv/bin/python3 -m orka eval \
  results/smollm-tristage.orka \
  --prompts wiki_prompts.txt \
  --out results/smollm-tristage.eval.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 10 \
  --max-length 64

echo -e "\n=== TRI-STAGE RESULTS ==="
cat results/smollm-tristage.eval.json | grep -E "loss_delta|perplexity_ratio"
echo -e "Final Artifact Size:"
du -sh results/smollm-tristage.orka
echo -e "Original Model Size (Theoretical 20 tensors): ~25 MB"
