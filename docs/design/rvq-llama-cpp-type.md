# RVQ as a llama.cpp/GGML quant type — design

Make orka's RVQ a first-class llama.cpp quant type so compressed models run on a
production C++ engine instead of as a transformers add-on. This is the only path to
matching unsloth on **runtime** (it already matches on compression).

## Why — the measured gap

LFM2-230M, RTX 3060, orka rvq-12-12 group-4 (planed) vs unsloth Q5_K_M, all measured:

| | orka (transformers + Triton VQ) | unsloth (llama.cpp) |
|---|---|---|
| disk | 148 MB | 161 MB |
| weight VRAM density | 2.89x | ~4x |
| N=1 decode | 97 tok/s | 681 tok/s |
| prefill | 489 tok/s | ~10,700 tok/s |
| perplexity (wikitext) | 56.9 | 36.2 (own fp16 baseline; +3.6% vs +3.4%) |

The gap decomposes (measured by isolating each layer):

```
unsloth llama.cpp C++        681 tok/s   (1x)
transformers dense fp16      291 tok/s   (2.3x slower  = transformers eager runtime)
orka planed VQ (Triton)       97 tok/s   (3x slower again = VQ kernel vs dense matvec)
```

Two costs, both runtime-side, neither fixable inside transformers:
1. **transformers eager** is 2.3x llama.cpp even on dense fp16.
2. The **VQ kernel** is 3x a dense matvec. The custom-op experiment (#88) compiled the
   model to **0 graph breaks / 1 graph** and CUDA graphs gave only +12% — proving the
   bottleneck is kernel efficiency, not Python/launch overhead.

Hand-optimizing orka's own CUDA VQ kernel was attempted and **regressed** (naive CUDA 88,
group-major+float2 69 tok/s) — beating the Triton path needs gemv_f4-class tuning per
shape. The leverage is in the engine, not another bespoke kernel.

## Goal / non-goals

- **Goal:** `GGML_TYPE_ORKA_RVQ` + CPU/CUDA dequant-mat-vec kernels + a `.orka -> .gguf`
  converter, so an orka model loads via stock llama.cpp and runs at k-quant-class speed.
- **Non-goal:** changing the codec. RVQ (arbitrary-width learned codebooks, multi-stage
  residual) is the moat. The type must carry RVQ as-is, not approximate it as int4.

## The core wrinkle: per-tensor codebook vs GGML's self-contained blocks

GGML quant types are **self-describing blocks** (e.g. `block_q4_K` = 256 weights + their
scales, contiguous). The dequant kernel receives only the weight block pointer. orka RVQ
stores **indices per group** + a **per-tensor codebook** (4096 x group_size halves ~ 64KB
/stage) shared by every block — the indices alone are not self-describing.

Three resolutions (matrix):

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. codebook ref per block** | each block header carries an offset to the tensor's codebook (stored once at tensor end) | fits GGML's "block ptr only" kernel signature; codebook stays single-copy | block struct is non-standard (a pointer/offset, not pure data) - needs care with GGML's contiguity assumptions |
| **B. extend GGML aux** | add a per-tensor side buffer passed to the dequant kernel | cleanest semantically | invasive to ggml core + every backend; large upstream surface |
| **C. per-block codebook** | re-learn codebooks per 256-weight block | pure standard GGML | changes the codec, more codebook overhead, likely hurts quality - **rejected (moat)** |

**Decision: Option A.** Codebooks are tiny (~64 KB/stage) → stage them in **shared memory**
once per threadblock; the index stream (the bulk) streams from HBM coalesced. This is the
exact pattern that makes the kernel memory-bound on indices (< int16 traffic) and fast -
the same insight behind the existing `gemv_f4` reaching dense parity for group-8.

## Block layout (Option A, sketch)

```
tensor data = [ index blocks ... ][ codebooks (per stage) ][ block scales ]
block_orka_rvq (one per GROUP_SIZE weights):
  uint8/uint12  idx[n_stages]     // packed indices (the bit-plane work, #84/#85)
  (codebook + scale referenced by tensor-level offsets in the type's extra header)
```
Reuse the bit-plane packing (`_pack_index_planes`, #84) for the index stream so VRAM
density carries over (10/12-bit, not int16).

## Kernel sketch (CUDA dequant mat-vec, N=1 decode)

```
load codebooks[stage] -> __shared__   (once per block; ~64KB fits)
for each group g handled by this warp/thread:
    idx_s = unpack(index_stream)             // lo | hi<<8, coalesced
    w = sum_s shared_codebook_s[idx_s]       // gather from SHARED mem (fast)
    acc += scale[g] * dot(w, x_g)
```
Memory-bound on the index stream (< int16). The codebook gather hits shared memory (the
thing my Triton/global-memory attempts got wrong). GGML's mat_vec harness handles tiling,
warp reduction, and dtype.

## Converter

`.orka -> .gguf`: emit each quantized tensor as `GGML_TYPE_ORKA_RVQ` (index blocks +
codebooks + scales per Option A); passthrough tensors as f16; reuse the existing
`export-vllm` tensor walk for naming/shape. The bit-plane packers already produce the
index bytes.

## Integration points (llama.cpp fork)

1. `ggml.h` / `ggml.c`: `GGML_TYPE_ORKA_RVQ` enum + `type_traits` (block size, to/from float).
2. `ggml-cuda/`: dequant + `mul_mat_vec` / `mul_mat` for the type (shared-mem codebook).
3. `ggml-cpu/`: scalar dequant (correctness reference + CPU inference).
4. `llama.cpp` model load: accept the type, no other model changes.

## Verification plan

- **Correctness:** CPU dequant of an orka tensor == `reconstruct_weight()` (bit-exact on
  indices; fp tolerance on the dot). Reuse the structural oracle for the pack side.
- **Speed:** `llama-bench` on the converted model vs the unsloth k-quant of the same model
  - target k-quant-class decode/prefill (the 681 / 10,700 tok/s regime).
- **Quality:** `llama-perplexity` on the converted model == orka's HF-eval ppl (same
  artifact, different runtime) within noise.

## Phasing

1. **PoC:** single-tensor `GGML_TYPE_ORKA_RVQ` + CUDA mat-vec (Option A, shared codebook),
   prove decode reaches k-quant speed on one shape. Gates the whole bet.
2. **Converter + CPU path:** full `.orka -> .gguf`, CPU dequant correctness.
3. **CUDA mat_mul (prefill)** + llama.cpp loader wiring + end-to-end `llama-bench` /
   `llama-perplexity` vs unsloth.

PoC (phase 1) is the go/no-go: if a shared-codebook CUDA RVQ mat-vec hits k-quant speed on
one tensor, the rest is engineering. If it can't (RVQ gather inherently slower than k-quant
arithmetic), the runtime gap is structural and orka stays a compression/research tool.
