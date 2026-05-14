#!/bin/bash
set -e

echo "=== 1. PRE-FLIGHT: CPU-ONLY CALIBRATION (VRAM Safe) ==="
CUDA_VISIBLE_DEVICES="" .venv/bin/python3 -m orka sem-calc \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-final-boss.calib.json \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --max-cpu-threads 8 \
  --device cpu

echo -e "\n=== 2. PACKING (GPU-Accelerated) ==="
.venv/bin/python3 -m orka pack \
  /home/kai/ai-models/qwen/Qwen3-0.6B/model.safetensors \
  --out results/qwen-final-boss.orka \
  --sensitivity-map results/qwen_pillars_10pct.json \
  --normalization awq-block-max \
  --awq-activations-file results/qwen-final-boss.calib.json \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --rotation orthogonal \
  --group-size 1024 \
  --em-aq-passes 1 \
  --sample-vectors 250000 \
  --codebook-cache /tmp/orka-cache \
  --max-cpu-threads 8 \
  --max-system-ram-gb 16.0 \
  --workload-budget-gb 9.0 \
  --max-gpu-mem-gb 8.0 \
  --backend torch \
  --device cuda

echo -e "\n=== 3. RECONSTRUCTION & EVAL ==="
.venv/bin/python3 -m orka reconstruct \
  results/qwen-final-boss.orka \
  --out results/qwen-final-boss-model/model.safetensors \
  --format safetensors \
  --device cuda

cp /home/kai/ai-models/qwen/Qwen3-0.6B/*.json results/qwen-final-boss-model/ 2>/dev/null || true
cp /home/kai/ai-models/qwen/Qwen3-0.6B/*.txt results/qwen-final-boss-model/ 2>/dev/null || true

.venv/bin/python3 -m orka eval \
  results/qwen-final-boss.orka \
  --prompts wiki_prompts.txt \
  --out results/qwen-final-boss.eval.json \
  --model-dir /home/kai/ai-models/qwen/Qwen3-0.6B \
  --device cuda \
  --max-prompts 50 \
  --max-length 128

echo -e "\n=== BENCHMARK COMPLETE ==="
du -sh results/qwen-final-boss.orka
cat results/qwen-final-boss.eval.json | grep -E "loss_delta|perplexity_ratio"
