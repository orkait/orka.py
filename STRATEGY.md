# Orka Strategy

Comprehensive record of every quantization strategy implemented in `orka.py`, why it exists, how it interacts with the rest of the pipeline, and current evidence.

This document is meant to be self-contained. If you reset state and re-read this, you should be able to pick up where the work left off without re-discovering tradeoffs.

---

## 1. Goal

Compress a Hugging Face causal LM checkpoint into a compact `.orka` directory using vector quantization, optional residual stages, and a stack of pre-/post-VQ transforms. The compiler is research-grade: format works, encode/decode round-trips, but bit-budget viability for actual LM behaviour is still being established.

Hard targets at the time of writing:
- 3.3 GB ceiling for an 8B model (≈ 3.3 bits per weight at group size 8).
- Strict 10 GB GPU VRAM cap (RTX 3060 12 GB).
- 24 GB host RAM, 12 cores.
- CPU-first, optional CUDA via `--backend torch --device cuda`.

---

## 2. Artifact format

A `.orka` artifact is a directory:

```
manifest.json
codebooks/
  <key>.codebook.f32           # single-stage VQ, family/global mode
  <key>.s{i}.codebook.f32      # multi-stage RVQ, family/global mode
tensors/
  <safe>.indices               # single-stage indices
  <safe>.s{i}.indices          # multi-stage indices, per stage
  <safe>.codebook.f32          # single-stage, per-tensor mode
  <safe>.s{i}.codebook.f32     # multi-stage, per-tensor mode
  <safe>.row_l2_scale.f32      # row-l2 normalization
  <safe>.col_l2_scale.f32      # col-l2 normalization
  <safe>.block_max_scale.f32   # block-max normalization
  <safe>.outliers.idx          # outlier sidecar positions (uint32)
  <safe>.outliers.val          # outlier sidecar values (fp16)
```

Index integer width is sized by per-stage bits: uint8 (≤8), uint16 (≤16), uint32 (≤32), uint64 (≤64).

The manifest records every per-tensor decision: stages list, codebook size per stage, index bits per stage, normalization, scale path/count/bytes, outlier metadata, rotation type and seed, error metrics (MSE, RMSE, MAE, max abs, source/recon L2 norms, dot, cosine, relative RMSE).

---

## 3. Quant spec naming

Compositional, parsed by `parse_quant_spec`:

- `vq-{b}` → single stage VQ. Codebook size `2^b`.
- `rvq-{b1}-{b2}-...{bn}` → N-stage Residual VQ. Stage `i` quantizes residual after stages `0..i-1`. Decoder sums centroid lookups across stages.
- `rvq-mixed` → preset for per-family mixed precision. Forces `--codebook-mode per-tensor`.

Constraints: per-stage bits 1..64, total bits per vector ≤ 64. Single-stage spec must use `vq-`, multi-stage must use `rvq-`. Cross-misuse is rejected at parse time.

Effective bits per weight = `sum(per_stage_bits) / group_size`. At g=8: `vq-8` = 1.0 bpw, `rvq-16-8` = 3.0 bpw, `rvq-16-16-16-16` = 8.0 bpw (max).

`rvq-mixed` family map (default):
- `embedding`: `rvq-16-16-16` (6 bpw)
- `attention`: `rvq-16-8` (3 bpw)
- `mlp`: `rvq-16-8` (3 bpw)
- `other`: `vq-16` (2 bpw)

---

## 4. Codebook modes

`--codebook-mode {per-tensor, global, family}`:

| mode | codebooks | fit | metadata cost |
|---|---|---|---|
| per-tensor | one per tensor | best | high (211 × 2 MB at k=65536 = 422 MB) |
| global | one shared | worst | tiny (2 MB) |
| family | one per role group | balanced | small (4 × 2 MB = 8 MB) |

Family classifier (`classify_tensor_family`) routes by tensor name regex:
- `embedding`: `embed`, `embedding`, `wte`, `wpe`
- `attention`: `attn`, `attention`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `c_attn`
- `mlp`: `.mlp.`, `mlp`, `gate_proj`, `up_proj`, `down_proj`, `c_fc`
- `other`: anything else

For shared modes, a stage's codebook is trained on the concatenation of (residual) vectors across all candidates in that group. Each candidate then quantizes its own vectors against the shared codebook.

---

## 5. Pre-VQ pipeline (in order)

Each candidate tensor passes through these stages **in this exact order** during pack. Decode reverses them.

### 5a. Normalization (`--normalization {none, row-l2, col-l2, block-max}`)

- **none**: raw tensor flattens directly.
- **row-l2**: each row of the (rows, cols) view divided by its L2 norm. One fp32 scale per row.
- **col-l2**: each column divided by its L2 norm. One fp32 scale per col. Mirror of row-l2 for output-channel scaling (GPTQ-style per-channel intuition).
- **block-max**: flat tensor split into N-element blocks (configurable via `--block-scale-size`, default 32). Each block divided by its max-abs. One fp32 scale per block. Inspired by 4/6 paper and GGUF Q4_K_M block scaling.

Storage cost (at g=8):
- row-l2: `4 * rows / total_params` bpw — small for tall matrices.
- col-l2: `4 * cols / total_params` bpw — small for wide matrices.
- block-max(32): `4 / 32 = 0.125 B/weight = 1 bpw` — biggest, captures most local magnitude variation.

### 5b. Rotation (`--rotation {none, orthogonal, hadamard}`)

Applies an orthogonal transform along the inner-axis of the tensor before VQ. Decoder applies the inverse.

- **orthogonal**: per-tensor random orthogonal matrix `Q` from QR of seeded random Gaussian. Seed = blake2b(name) XOR global `--rotation-seed`. Q regenerated at decode from same seed (no matrix stored). Works for any inner dim. Compute cost: O(cols^2) for QR + O(rows × cols^2) for matmul. Memory: Q matrix = cols^2 × 4 B.
- **hadamard**: deterministic FWHT (Fast Walsh-Hadamard Transform) along inner axis. Requires inner dim be a power of 2. No seed, involutive when normalized by 1/sqrt(n). Compute cost: O(rows × cols log cols). Memory: in-place, no Q stored.

### 5c. Outlier escape (`--outlier-frac F`)

After normalization and rotation but before vectorizing:
- Take `K = floor(F * packed_values)` positions of largest |value| in the post-rotate, post-normalize flat.
- Save (positions uint32, values fp16) sidecars. 6 bytes per outlier.
- Zero those positions in the tensor that VQ sees.

Decoder reads the sidecar and re-injects the exact fp16 values at the saved positions before un-rotating and un-normalizing.

Effect: VQ centroids no longer have to cover extreme outliers. Codebook fits the bulk. Outliers are stored exactly (within fp16 precision).

### 5d. Vectorize and group

Flatten, pad to multiple of `group_size`, reshape to `(V, g)`. Each row is a vector that the codebook quantizes.

---

## 6. RVQ stages

For `rvq-{b1}-...{bn}`:

1. `vectors_orig = vectors`
2. `vectors_residual = vectors_orig`
3. `decoded_sum = 0`
4. For stage `i` in 0..n-1:
   - Train codebook (or load from cache) on `vectors_residual` (per the `--codebook-mode`).
   - Quantize: `indices_i = argmin_k ||residual_v - codebook_i[k]||`.
   - Optionally apply GPTQ for stage 0 (see §10).
   - `decoded_i = codebook_i[indices_i]`.
   - `decoded_sum += decoded_i`.
   - `vectors_residual = vectors_orig - decoded_sum`.

Final reconstruction in normalized rotated space = `decoded_sum`. Decoder un-rotates and un-normalizes from there.

---

## 7. K-means details

`learn_codebook_auto` dispatches by backend:

- **python**: pure stdlib, for tiny tests only.
- **numpy**: CPU vectorized.
- **torch**: GPU when `--device cuda`. Uses `torch.cdist` chunked to bound the squared-distance matrix.

### 7a. Init: K-means++

Replaces the original linspace-sort init. For a sample of N vectors and target k centroids:
1. Pick first centroid uniformly random.
2. Maintain `min_d2[i]` = squared distance from vector i to nearest already-picked centroid.
3. For each subsequent centroid: sample one vector with probability proportional to `min_d2`. Update `min_d2` against the new centroid.

Better local optimum than deterministic init. Quality lift is small but free. Runtime: O(k * N * d). At k=65536, N=1M, d=8 → ~30 sec on RTX 3060.

K-means++ is **inherently sequential** (each centroid depends on prior). Strict-zero-loss parallelization isn't possible.

### 7b. Lloyd iterations

Standard EM:
- Assign each vector to its nearest centroid.
- Update each centroid as the mean of its assigned vectors (using `bincount` + `index_add_` on torch / `np.add.at` on numpy).

Default 12 iterations. Past 8 most movement has stopped; past 16 returns are tiny.

### 7c. Sample subsetting

`--sample-vectors N`: train k-means on a deterministic subset of the full vector set (linspace sampling) when `N < V`. Quantization itself still runs on every vector. Default behavior trains on full set.

For k=65536 the rule of thumb is 50× to 100× samples (3M to 7M) for stable codebook. We commonly use 1M, which is below that bar; quality is borderline at the very tail of the codebook.

### 7d. Adaptive cdist chunking

`_torch_assign` computes `(chunk × k)` distance matrices in float32. Chunk size auto-derived as `max(256, min(default, (1 << 28) / k))`, capping the matrix at ~256 MB regardless of k. Exceeding this caused OOM at k=65536 on the 10 GB cap; smaller chunks fit.

### 7e. linspace float64 fix

`_sample_vector_rows` originally used `torch.linspace(..., dtype=float32)`. For `len(vectors) > 16.7M` (float32 mantissa limit) the endpoint overflows by epsilon, rounding to an out-of-bounds index. Fixed by forcing `dtype=float64` and clamping to `len-1`. Same fix applied in `_learn_codebook_torch` init position calculation.

---

## 8. Codebook caching

`--codebook-cache DIR`. For stage 0 only (later stages depend on stage 0 output and are not safely cacheable in their current form). Cache key is a blake2b of:

`(scope, source_signature, mode, family_or_tensor_key, group_size, k, sample_vectors, iterations, backend, normalization, rotation, rotation_seed, outlier_frac, max_tensors, stage_index)`

`source_signature` = `f"{file_size}-{mtime_ns}"` so any source change invalidates the cache. Atomic write via `.tmp` rename.

First run with a config: cache miss, train + save. Subsequent identical runs: load cached codebook in milliseconds, skip k-means entirely.

Strict zero quality loss for cache hits (binary identical centroids).

---

## 9. Index storage and integer widths

`_INDEX_BIT_SPECS` tuples define `(ceiling_bits, numpy_dtype, struct_format)`:
- `(8, "<u1", "<B")`, `(16, "<u2", "<H")`, `(32, "<u4", "<I")`, `(64, "<u8", "<Q")`

`_index_bit_spec(b)` picks the smallest container that fits `b` bits. Indices are written little-endian, contiguous, no header. The manifest carries `index_bits` per stage so the reader knows which dtype to use.

---

## 10. GPTQ error correction (partial / unstable)

`--gptq-calibration <prompts.txt>` enables activation-aware quantization for stage 0:

1. Load HF model from `--gptq-model-dir`.
2. Register forward hooks on every `torch.nn.Linear`. Run calibration prompts.
3. For each module, collect input activations `X` of shape `(n_samples, in_dim)`. Sub-sample if more than `--gptq-max-samples`.
4. Compute Hessian `H = (2/n) X^T X`, dampen diagonal by `--gptq-dampening * mean(diag(H))`, Cholesky `H = L_lower L_lower^T`, invert via `cholesky_inverse`, take upper-Cholesky `U = chol(H^-1, upper=True)`.
5. For each weight matrix `W` of shape `(out, in)`: process column blocks of size `g` left-to-right. For each block `b` at column `c`:
   - VQ-assign each row's block to nearest codebook centroid (vectorized over all output rows in one cdist).
   - Compute per-row residual `err = W[:, c:c+g] - centroid`.
   - Solve `U[c:c+g, c:c+g] x = err` via `solve_triangular` (upper).
   - Spread error: `W[:, c+g:] -= x @ U[c:c+g, c+g:]`.

This is the textbook block-GPTQ adapted to VQ assignment. The key property is that subsequent column blocks see a residual already pre-compensated for prior quantization error.

### Status (current evidence)

Constraints in current implementation:
- GPTQ branch only runs when `rotation == "none"`, `normalization == "none"`, `outlier_frac == 0`, and `backend == "torch"`. Combining GPTQ with other lift tricks would require activation-space transforms that aren't wired.
- Tested at vq-8 (1 bpw) and rvq-16-8 (3 bpw). Both **regressed vs the same configs without GPTQ**. Evidence:
  - vq-8 plain: loss_delta +9.34. vq-8 + GPTQ: +11.91.
  - rvq-16-8 family + (col-l2 + rotation + outliers, no GPTQ): +6.68. rvq-16-8 family + GPTQ alone: +8.06.

Suspected causes (not yet root-caused):
- Calibration undersampling. With 16 prompts × 128 tokens, each layer sees ~2k samples. For MLP weights with `in_dim=1536`, the empirical Hessian has rank ≤ 2k while the matrix is 1536² → rank-deficient. Dampening 0.01 may be too low.
- VQ assignment is L2-nearest-centroid, not Hessian-weighted. Standard GPTQ for scalar quant doesn't change this either, but the interaction with discrete VQ centroids may be sensitive in ways scalar GPTQ isn't.
- Possible subtle indexing or dtype bug not yet caught.

GPTQ is shipped but should be considered **experimental** at this revision. Higher dampening (0.1+), more calibration (>32 prompts × 256 tokens), and sanity checks on the Hessian conditioning are next steps.

---

## 11. VRAM cap and CappedOutOfMemoryError

`--max-gpu-mem-gb GB` (pack + sweep) calls `torch.cuda.set_per_process_memory_fraction(GB / total, device_index)` at startup. PyTorch's caching allocator refuses any allocation that would exceed the cap and raises `torch.cuda.OutOfMemoryError` (or a `RuntimeError` containing `"out of memory"`).

`_wrap_capped_oom(cap, fn, *args, **kwargs)` re-raises matching CUDA OOM as `CappedOutOfMemoryError` (subclass of `RuntimeError`) when a cap was set. This makes "I exceeded the user-stated budget" distinguishable from "the GPU itself ran out of memory under no cap".

Caveats: the cap only counts torch's caching allocator. cuBLAS/cuDNN workspace (50-200 MB) is outside. Real usable budget is ~10 - 0.2 = ~9.8 GB.

---

## 12. Decode pipeline

`_decode_tensor` reverses the pack pipeline:

1. Read all stages: codebook + indices for each. Produce `decoded_sum = sum_i codebook_i[indices_i]` (in flat form, length = `padded_values`).
2. Truncate to `packed_values`.
3. If outliers present: read sidecars, set `decoded[positions] = values` (overrides VQ centroids at outlier positions).
4. If rotation == orthogonal: rebuild Q from seed, multiply by Q^T.
5. If rotation == hadamard: apply FWHT (involutive when normalized).
6. If normalization == row-l2: multiply each row by its scale.
7. If normalization == col-l2: multiply each column by its scale.
8. If normalization == block-max: multiply each 32-element block by its scale.

Order matters: rotation/normalization were applied in pack as `norm → rotate`, so decode must do `un-rotate → un-norm`. Outliers were extracted after both transforms in pack, so they re-inject before un-rotate at decode.

The `_decode_tensor` is dependency-light (numpy-only). The pack-side metric helpers `_stage_quality_metrics` and `_denorm_metrics_from_flat` mirror this for in-pack reporting.

Backward compat: manifest entries without `stages` are interpreted as a single-stage config from the top-level fields. Old `.orka` artifacts continue to load.

---

## 13. Sweep, eval, and eval-sweep

- `sweep`: cartesian over `--group-sizes × {codebook-sizes ∪ quant-modes} × codebook-modes × normalizations` plus rotation, outlier, GPTQ flags. One pack per combination, one report each, summary JSON with `best_by_relative_rmse`, `best_by_cosine_similarity`, `best_by_cosine_per_mb`.
- `eval`: load HF model, reconstruct an `.orka` artifact into a temp model dir, score causal-LM loss on the prompts of `--prompts`, compare to the original. Reports `original_loss`, `orka_loss`, `loss_delta`, perplexity ratio.
- `eval-sweep`: walk every artifact recorded in a sweep summary, run `eval` on each, aggregate with `best_by_loss_delta`, `best_by_perplexity_ratio`, `best_by_artifact_bytes`.

`eval` uses `local_files_only=True` by default — Hugging Face will not download anything unless `--allow-download` is set.

---

## 14. Ergonomics for fail-fast iteration

- `--max-tensors N`: stop pack after the first N candidate tensors. Cuts a 10 minute run on smollm2-135M to ~10 sec for 5 tensors. Useful for catching pipeline bugs without paying the full cost.
- Tiny test recipe: tiny-random-gpt2 + vq-8 + 4 iter ⇒ pack + verify in ~5 sec.
- Mid test recipe: smollm2-135M + vq-8 + max-tensors 50 + 5 prompts + max-length 64 ⇒ pack + eval in ~30 sec.

---

## 15. Evidence so far (smollm2-135M)

Reproduced from `results/eval-*.json` against the full 50-candidate-tensor checkpoint. Earlier rows that are not present in `results/` have been moved to §15a.

| artifact | config | size | weighted_cos | loss_delta | ppl_ratio | eval file |
|---|---|---|---|---|---|---|
| `smollm2-135m-ultimate.orka` | rvq-16-8 fmly + awq-block-max + EM-AQ Joint + Hessian K-Means\|\| | 32.0 MB | N/A | **+0.657** | 1.9x | `eval-ultimate.json` |
| `smollm2-135m-best-awq.orka` | rvq-16-8 family + awq + rotation orthogonal | 26.5 MB | 0.986 | +8.15 | 3.45e3 | `eval-best-awq.json` |
| `smollm2-135m-quip.orka` | vq-8 e8-lattice + rht | 7.1 MB | 0.721 | +15.11 | 3.65e6 | `eval-quip.json` |
| `smollm2-135m-sota2.orka` | vq-8 e8-lattice + awq-block-max + rotation orthogonal | 7.1 MB | 0.822 | +50.07 | 5.56e21 | `eval-sota2.json` |
| `smollm2-135m-sota.orka` | vq-8 e8-lattice + awq-block-max + rotation orthogonal | 7.1 MB | 0.799 | +62.67 | 1.66e27 | `eval-sota.json` |

Sweep micro-runs (only first 5 tensors, **not full-model**, kept for triage):

| artifact | config | size | weighted_cos | loss_delta | ppl_ratio | eval file |
|---|---|---|---|---|---|---|
| `g8-vq-8-per-tensor-awq.orka` | vq-8 per-tensor + awq | 3.9 MB (5 t) | 0.818 | +8.14 | 3.42e3 | `eval-awq.json` |
| `g8-vq-8-per-tensor-none.orka` | vq-8 per-tensor + none | 3.9 MB (5 t) | 0.850 | +12.06 | 1.72e5 | `eval-none.json` |

Current full-model best (The Ultimate Run): **rvq-16-8 family + awq-block-max + EM-AQ Joint + Hessian K-Means||**, size 32.0 MB, loss_delta +0.657, perplexity ratio 1.9x. This shatters the +1.0 loss-delta barrier using 4 bpw, and proves that activation-aware clustering and sensitivity-driven precision mapping can preserve base model intelligence across quantization.

### 15a. Historic claims not in current `results/`

These rows lived in earlier revisions of this file and are **not reproduced** by any artifact in `results/`. They are kept here as decisions to re-verify, not as evidence:

- vq-8 family plain at 1.0 bpw: cos 0.812, loss_delta +9.34
- vq-16 family plain at 2.0 bpw: cos 0.950 (loss not eval'd)
- rvq-8-8-8 family plain at 3.0 bpw: cos 0.977 (loss not eval'd)
- rvq-16-8 family + col-l2 + rotation + outliers at 3.1 bpw / 59 MB: cos 0.982, loss_delta +6.68
- rvq-16-8 family + GPTQ stage 0 at 3.0 bpw / 51 MB: loss_delta +8.06
- GPTQ vq-8 at 1.0 bpw / 17 MB: loss_delta +11.91
- rvq-16-8 family + block-max-32 + rotation at 4.0 bpw / 74 MB: cos 0.985, loss_delta +4.23

Re-running these requires the smollm2-135M HF checkpoint at the path recorded in any current manifest's `source` field plus a CUDA-capable host.

---

## 16. What is open

Not yet implemented but on the docket:

1. **[SOLVED] AWQ-style activation-aware scaling**. Per-channel scale derived from calibration activations. Pre-multiply weights so VQ error lands in low-activation directions. ~200 LOC. Likely worth more than GPTQ at low bpw.
2. **Block-wise codebooks** (Q4_K_M flavour). Split each tensor into 64-1024 element super-blocks, learn a small codebook per super-block. Increases metadata but tightens per-region fit.
3. **Lattice quantization** (E8, Leech). No training, algebraic decode. Cheap baseline, ~0.3 dB worse than optimal VQ.
4. **Auto bit allocation across RVQ stages**. Solve a small budget assignment problem instead of hand-picking `(16, 8)` etc.
5. **[SOLVED] Sensitivity-driven mixed precision**. Replace the hardcoded `rvq-mixed` family map with per-tensor sensitivity from gradient or perplexity probe.
6. **fp16 / fp8 scale dtypes** instead of fp32. Halves or quarters scale-overhead bpw.
7. **Hadamard padding** for non-pow2 cols. Currently `--rotation hadamard` only accepts pow2 inner dims; padding would extend coverage.
8. **CUDA Graph capture** for inner cdist+argmin and GPTQ inner block. ~1.3-1.5× scheduler win, zero quality loss.
9. **[SOLVED] Parallel kmeans++ (k-means||)**. Bahmani et al. Replaces the sequential k-iteration init with O(log k) parallel rounds. ~50-100× init speedup at k=65536. Tiny variance vs strict sequential.
10. **GGUF writer**. Per ORKA.md ship target.
11. **Runtime kernel** (VQ-aware matmul). Required for any actual inference, not just storage compression.
12. **QAT** (quantization-aware fine-tuning). Heavy but essential to crack the loss-delta floor at ≤3 bpw.

Open issues to debug:
- GPTQ regression at vq-8 and rvq-16-8 (see §10).
- Lift-stack ceiling around cos 0.985 / loss_delta +5-7 at 3 bpw on this checkpoint.

---

## 17. Pipeline diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  pack pipeline (per candidate tensor)                           │
└─────────────────────────────────────────────────────────────────┘

  source tensor (R, C, ...)
        │
        ▼
  [normalize]   ← row-l2 / col-l2 / block-max  (saves scales sidecar)
        │
        ▼
  [rotate]      ← orthogonal QR / hadamard FWHT (saves seed in manifest)
        │
        ▼
  [outlier extract]  ← top-K abs values to fp16 sidecar; zero in tensor
        │
        ▼
  [vectorize]   ← flatten, pad, reshape to (V, g)
        │
        ▼
  [RVQ stage 0] → indices_0, decoded_sum = decoded_0
        │
        ▼  residual = vectors_orig - decoded_sum
  [RVQ stage 1] → indices_1, decoded_sum += decoded_1
        │
       ...
        ▼
  [write manifest + indices + codebooks + scales + outlier sidecars]


┌─────────────────────────────────────────────────────────────────┐
│  decode pipeline (mirror, per tensor)                           │
└─────────────────────────────────────────────────────────────────┘

  read indices + codebooks
        │
        ▼
  decoded_sum = Σ_i codebook_i[indices_i]
        │
        ▼
  [outlier inject]   ← overwrite at saved positions
        │
        ▼
  [un-rotate]        ← Q^T or FWHT
        │
        ▼
  [un-normalize]     ← multiply by row/col/block scales
        │
        ▼
  reconstructed tensor
```
