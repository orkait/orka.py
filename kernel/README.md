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

## Remaining work to run inside llama.cpp (mechanical, scoped)

| Step | File | Effort |
|---|---|---|
| Add `GGML_TYPE_ORKA_VQ` enum | `ggml/include/ggml.h` | trivial |
| Block struct + `GGML_QUANT_SIZES` | `ggml-common.h`, `gguf-py/gguf/constants.py` | low |
| CPU type-traits entry (`to_float`, `vec_dot`, `vec_dot_type=Q8_K`) | `ggml-cpu/ggml-cpu.c` | low |
| Wrap this kernel as `dequantize_row_orka_vq` | `ggml-cpu/quants.c` | done here (adapt signature) |
| **Pass per-tensor codebook/sidecars to the kernel** | `ggml.h` compute params + graph build | **HIGH - the crux** |
| GGUF writer emits the type + side tensors | `tools/orka_to_gguf.py` | medium |
| CUDA kernel + support matrix | `ggml-cuda/` | high (post-CPU) |

The crux remains the codebook-passing plumbing: every existing GGML kernel
assumes "type fully determines decode", so the codebook is implicit. ORKA_VQ
needs the kernel to receive an extra per-tensor tensor. Standard GGML signatures
(`dequantize_row_*(const block*, float*, k)`) have no slot for it, so the side
tensors must travel through `ggml_compute_params` / op context. That is C
plumbing in llama.cpp internals - well understood, but it is the part that
must be done inside a full llama.cpp build, not here.

## Files

- `orka_vq.h` / `orka_vq.c` - the kernel and its struct contract
- `test_orka_vq.c` - synthetic self-test (no artifact, no Python; for CI)
- `verify_kernel.py` - byte-exact comparison vs the Python decoder on real artifacts
- `Makefile` - `make` builds + runs the self-test and the shared lib
