#!/usr/bin/env bash
# End-to-end proof: decode real Orka tensors inside a ggml custom op and check
# the matmul against the numpy reference. Builds against a llama.cpp libggml.
#
# Usage: kernel/run_ggml_proof.sh <artifact.orka> [llama.cpp dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ARTIFACT="${1:-$ROOT/.local_runs/dist/SmolLM2-135M-4bpw.orka}"
LLAMA="${2:-$ROOT/llama.cpp}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

make -C "$HERE" ggml-op LLAMA="$LLAMA" >/dev/null

# Representative tensor types: attention, both mlp projections, the big
# embedding (49152 rows, absolute outlier positions past 2^24).
TENSORS=(
  model.embed_tokens.weight
  model.layers.0.self_attn.q_proj.weight
  model.layers.0.self_attn.o_proj.weight
  model.layers.5.mlp.gate_proj.weight
  model.layers.5.mlp.down_proj.weight
)
fail=0
for t in "${TENSORS[@]}"; do
  if ! "$PY" "$HERE/dump_tensor.py" "$ARTIFACT" "$t" "$TMP/t" >/dev/null 2>&1; then
    printf '%-44s SKIP (not a quantized tensor)\n' "$t"
    continue
  fi
  line="$("$HERE/ggml_orka_op" "$TMP/t" 2>/dev/null | grep -o 'rel=.* -> .*' || echo 'ERR')"
  printf '%-44s %s\n' "$t" "$line"
  [[ "$line" == *MATCH* ]] || fail=1
done
exit $fail
