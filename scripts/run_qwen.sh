#!/bin/bash
echo "Starting Qwen 0.5B Compress Pipeline..." > /tmp/orka-qwen.log


echo "2. Packing Model (AWQ-Block-Max, RVQ-16-8, K-Means||, EM-AQ)..." >> /tmp/orka-qwen.log
.venv/bin/python3 -m orka pack \
  /tmp/qwen-0.5b/model.safetensors \
  --out results/qwen-0.5b-ultimate.orka \
  --sensitivity-map results/qwen_sensitivity_map.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /tmp/qwen-0.5b \
  --quant-mode rvq-16-8 \
  --codebook-mode family \
  --rotation orthogonal \
  --group-size 8 \
  --progress-file .orka_progress_qwen \
  --backend torch \
  --device cuda \
  --iterations 4 >> /tmp/orka-qwen.log 2>&1

echo "3. Evaluating Model..." >> /tmp/orka-qwen.log
.venv/bin/python3 -m orka eval \
  results/qwen-0.5b-ultimate.orka \
  --prompts wiki_prompts.txt \
  --out results/eval-qwen-ultimate.json \
  --model-dir /tmp/qwen-0.5b \
  --device cuda \
  --max-prompts 10 \
  --max-length 128 >> /tmp/orka-qwen.log 2>&1

echo "Done!" >> /tmp/orka-qwen.log
