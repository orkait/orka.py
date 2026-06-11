# ORKA_VQ kernel - compressed-inference proof of concept

This directory holds the CPU dequantization kernel for a future
`GGML_TYPE_ORKA_VQ` quantization type: the path that lets Orka's compression
survive into RAM at inference time, instead of being inflated back to dense
weights at load (what `orka export-vllm` does today).

## What is proven

`orka_vq.c` decodes one Orka-packed tensor through the **full** decode chain -
N-stage residual VQ, slrq block scale, outlier escape, salient escape, low-rank
correction - in dependency-free C99. It is verified **byte-exact** against the
reference Python decoder (`orka.pipeline.decode._decode_tensor`):

```
make                      # builds + runs the self-contained C unit test
python verify_kernel.py .local_runs/dist/SmolLM2-135M-4bpw.orka 211
# -> worst relative diff across 211 tensors: 0.000e+00
```

Zero diff on all 211 tensors of SmolLM2-135M-4bpw (and the 360M artifact). The
genuinely novel and risky part of a GGML type - the numerical decode and its
operation ordering - is correct and tested. The kernel receives already-unpacked
`int32` indices and `f32` codebooks/scales: bit-unpacking and stream
entropy-decode are the loader's job, not the kernel's.

## The architectural problem this is built around

llama.cpp's IQ types (IQ2_XXS, IQ3_S, ...) are codebook-based, but the codebook
is a **single universal grid compiled into the binary** - every IQ3_S tensor in
every model shares one 2 KB table (`iq3s_grid` in `ggml-common.h`). Orka's
codebooks are **per-tensor and learned**. There is no way to express an Orka
codebook as an IQ type; a genuinely new type is required, one whose kernel reads
the codebook from a **side tensor** rather than a constant.

That is the design here: `orka_vq_tensor` carries per-stage codebook pointers.
The GGML integration must thread those side tensors to the kernel.

## The crux is solved: codebook decode runs inside ggml

The hard unknown was whether a *per-tensor learned codebook* could be threaded
into ggml's compute backend at all, given that every standard GGML kernel
signature (`dequantize_row_*(const block*, float*, k)`) has no slot for a
codebook. The answer, demonstrated here, is **a custom op carrying the
compressed weight as a real ggml tensor**:

```
kernel/ggml_orka_op.c   - ggml_map_custom2(blob_tensor, activations):
                          decodes the Orka weight from the blob inside the
                          op and does the GEMM, dispatched by ggml's CPU backend.
kernel/run_ggml_proof.sh - builds against libggml and checks the matmul of real
                          135M tensors vs the numpy reference.
```

Result - the full weight (all stage codebooks + indices + slrq scales +
outlier/salient/low-rank sidecars) packed into one ggml tensor, decoded inside
the backend, GEMM verified against numpy:

```
model.embed_tokens.weight              rel=9.3e-07  MATCH   (49152 rows)
model.layers.0.self_attn.q_proj.weight rel=5.1e-07  MATCH
model.layers.0.self_attn.o_proj.weight rel=4.2e-07  MATCH
model.layers.5.mlp.gate_proj.weight    rel=5.9e-07  MATCH
model.layers.5.mlp.down_proj.weight    rel=9.3e-07  MATCH
```

(rel ~1e-6 is f32 GEMM accumulation noise, not algorithmic error.) A real
debugging find along the way: absolute outlier positions exceed 2^24 on the
embedding and cannot round-trip as float32 - they are stored as int32 bit
patterns in the blob, the kind of issue only a real end-to-end run surfaces.

This proves the load-bearing mechanism. The `ggml_map_custom` op is the
deployable form for an out-of-tree build (no llama.cpp fork needed); a fully
upstreamed `GGML_TYPE_ORKA_VQ` (native enum + block struct + type traits) is a
cleaner long-term packaging, but it would run the *same* decode this op already
runs correctly.

## Remaining work to ship a runnable GGUF model

| Step | File | Effort | Status |
|---|---|---|---|
| Codebook decode inside ggml backend | `ggml_orka_op.c` | - | **done, verified** |
| GGUF writer emits compressed weight blobs + a custom-op tag | `tools/orka_to_gguf.py` | medium | the XOR mode must be dropped first |
| Wire the custom op per linear in llama's graph | `src/llama-model.cpp` / `src/models/` | medium | architecture boilerplate |
| Multi-threaded + SIMD GEMM in the op | `ggml_orka_op.c` | medium | POC is single-threaded |
| Optional: native `GGML_TYPE_ORKA_VQ` upstream | `ggml.h`, `ggml-common.h`, `ggml-cpu.c` | high | packaging, not capability |
| CUDA op | `ggml-cuda/` | high | post-CPU |

## Files

- `orka_vq.h` / `orka_vq.c` - the kernel and its struct contract
- `test_orka_vq.c` - synthetic self-test (no artifact, no Python; for CI)
- `verify_kernel.py` - byte-exact comparison vs the Python decoder on real artifacts
- `Makefile` - `make` builds + runs the self-test and the shared lib
