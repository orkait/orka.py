#!/bin/bash
set -e

echo "=== 1. ANALYZING OPENMOE-BASE ==="
.venv/bin/python3 -m orka sem-analyze hpcai-tech/openmoe-base \
  --out results/openmoe_analysis.json \
  --save-sensitivity-map results/openmoe_pillars.json

echo -e "\n=== 2. PACKING (MoE Crusher: Best Config) ==="
.venv/bin/python3 -m orka pack \
  hpcai-tech/openmoe-base \
  --out results/openmoe-crushed.orka \
  --sensitivity-map results/openmoe_pillars.json \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --rotation orthogonal \
  --group-size 1024 \
  --em-aq-passes 1 \
  --sample-vectors 500000 \
  --max-gpu-mem-gb 10.0 \
  --backend torch \
  --device cuda

echo -e "\n=== 3. EVALUATING ==="
# Safe path extraction
MODEL_DIR=$(.venv/bin/python3 -c "import json; print(json.load(open('results/openmoe_analysis.json'))['model_dir'])")

.venv/bin/python3 -m orka eval \
  results/openmoe-crushed.orka \
  --prompts wiki_prompts.txt \
  --out results/openmoe-crushed.eval.json \
  --model-dir "$MODEL_DIR" \
  --device cuda \
  --max-prompts 20 \
  --max-length 128

echo -e "\n=== FINAL MOE BENCHMARK SUMMARY ==="
du -sh results/openmoe-crushed.orka
cat results/openmoe-crushed.eval.json | grep -E "loss_delta|perplexity_ratio"
