#!/bin/bash
set -e

# 1. GENERATE PILLAR MAP
echo "=== 1. GENERATING SEMANTIC MAP ==="
.venv/bin/python3 -m orka sem-analyze HuggingFaceTB/nanowhale-100m \
  --out results/nanowhale_analysis.json \
  --save-sensitivity-map results/nanowhale_pillars.json

# 2. PACK (MoE Crusher)
echo -e "\n=== 2. PACKING (MoE Crusher Pipeline) ==="
.venv/bin/python3 -m orka pack \
  HuggingFaceTB/nanowhale-100m \
  --out results/nanowhale-crushed.orka \
  --sensitivity-map results/nanowhale_pillars.json \
  --quant-mode rvq-mixed \
  --codebook-mode per-tensor \
  --normalization slrq-block \
  --rotation orthogonal \
  --group-size 1024 \
  --em-aq-passes 1 \
  --sample-vectors 250000 \
  --max-gpu-mem-gb 8.0 \
  --backend torch \
  --device cuda

# 3. EVALUATE
echo -e "\n=== 3. EVALUATING ==="
# Safe path extraction from the analysis JSON
MODEL_DIR=$(.venv/bin/python3 -c "import json; print(json.load(open('results/nanowhale_analysis.json'))['model_dir'])")

.venv/bin/python3 -m orka eval \
  results/nanowhale-crushed.orka \
  --prompts wiki_prompts.txt \
  --out results/nanowhale-crushed.eval.json \
  --model-dir "$MODEL_DIR" \
  --device cuda \
  --max-prompts 20 \
  --max-length 128

echo -e "\n=== FINAL MOE CRUSHER SUMMARY ==="
du -sh results/nanowhale-crushed.orka
cat results/nanowhale-crushed.eval.json | grep -E "loss_delta|perplexity_ratio"
