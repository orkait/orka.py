#!/bin/bash
set -e

for PCT in 10 20 30
# for PCT in 10 20 30 40
do
  echo -e "\n\n=== RUNNING PILLAR SWEEP: ${PCT}% PROTECTION ==="
  
  # 1. Create specific sensitivity map for this percentage
  .venv/bin/python3 -c "
import json
with open('results/smollm_pillar_variants.json') as f:
    vars = json.load(f)
with open('results/sensitivity_sweep_${PCT}.json', 'w') as f:
    json.dump({'top_tokens': vars['${PCT}pct'], 'layers': []}, f)
"

  # 2. Pack
  .venv/bin/python3 -m orka pack \
    /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
    --out results/smollm-pillars-${PCT}pct.orka \
    --sensitivity-map results/sensitivity_sweep_${PCT}.json \
    --normalization slrq-block \
    --quant-mode vq-8 \
    --codebook-mode per-tensor \
    --rotation orthogonal \
    --group-size 8 \
    --em-aq-passes 1 \
    --max-tensors 15 \
    --backend torch \
    --device cuda

  # 3. Eval
  .venv/bin/python3 -m orka eval \
    results/smollm-pillars-${PCT}pct.orka \
    --prompts wiki_prompts.txt \
    --out results/smollm-pillars-${PCT}pct.eval.json \
    --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
    --device cuda \
    --max-prompts 10 \
    --max-length 128

  echo -e "--- ${PCT}% Result ---"
  cat results/smollm-pillars-${PCT}pct.eval.json | grep -E "loss_delta|perplexity_ratio"
  du -sh results/smollm-pillars-${PCT}pct.orka
done
