#!/bin/bash
echo "Starting SOTA Pack..." > /tmp/orka-ultimate.log

.venv/bin/python3 -m orka pack \
  /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m/model.safetensors \
  --out results/smollm2-135m-ultimate.orka \
  --sensitivity-map results/sensitivity_map_wikitext.json \
  --normalization awq-block-max \
  --awq-calibration wiki_prompts.txt \
  --awq-model-dir /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m \
  --quant-mode rvq-16-8 \
  --codebook-mode family \
  --rotation orthogonal \
  --group-size 8 \
  --max-tensors 50 \
  --progress-file .orka_progress \
  --backend torch \
  --device cuda \
  --iterations 4 >> /tmp/orka-ultimate.log 2>&1

echo "Pack complete. Starting Eval..." >> /tmp/orka-ultimate.log

.venv/bin/python3 -m orka eval \
  results/smollm2-135m-ultimate.orka \
  --prompts wiki_prompts.txt \
  --out results/eval-ultimate.json \
  --model-dir /mnt/storage/codespace/code/orkait/graphstore/graphstore/models/orka-smollm2-135m \
  --device cuda \
  --max-prompts 10 \
  --max-length 128 >> /tmp/orka-ultimate.log 2>&1

echo "Eval complete." >> /tmp/orka-ultimate.log
