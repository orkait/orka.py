#!/bin/bash
set -e

echo "=== 1. PACKING (Compression) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/misc/orka-smollm2-135m/model.safetensors \
  --out results/test-metrics.orka \
  --quant-mode rvq-8-8 \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --outlier-frac 0.01 \
  --rotation orthogonal \
  --em-aq-passes 3 \
  --sample-vectors 250000 \
  --codebook-cache /tmp/orka-cache \
  --backend torch \
  --device cuda \
  --max-tensors 5

echo -e "\n=== 2. VERIFYING (Mathematical Fidelity) ==="
.venv/bin/python3 -m orka verify results/test-metrics.orka

echo -e "\n=== 3. EVALUATING (Cognitive Retention) ==="
.venv/bin/python3 -m orka eval \
  results/test-metrics.orka \
  --prompts wiki_prompts.txt \
  --out results/test-metrics.eval.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo -e "\n=== 4. PULSE CHECK (Behavioral Drift) ==="
.venv/bin/python3 -m orka pulse-check \
  results/test-metrics.orka \
  --prompts wiki_prompts.txt \
  --out results/test-metrics.pulse.json \
  --model-dir /home/kai/ai-models/misc/orka-smollm2-135m \
  --device cuda \
  --max-prompts 5 \
  --max-length 64

echo -e "\n=== RESULTS ==="
echo "--- Eval (Loss/Perplexity) ---"
cat results/test-metrics.eval.json
echo -e "\n--- Pulse (KL Divergence/Top-1) ---"
cat results/test-metrics.pulse.json
