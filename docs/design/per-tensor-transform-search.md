# Per-tensor transform search (allocate increment 2)

Increment 1 (PR #116) shipped the write path: the allocation map can carry
`{name: {normalization?, rotation?}}` and `pack` applies it per tensor (per-tensor
codebook mode only). Nothing yet *chooses* those values. This doc designs the
search that populates the map - making `allocate` pick normalization/rotation per
tensor instead of taking one global flag.

## Problem

`build_allocation` (`orka/quant/allocate.py:75`) currently, per tensor:
1. samples group-vectors from the **raw** tensor,
2. runs a full greedy-RVQ k-means probe (`_probe_spec_distortion`, `allocate.py:46`) for each spec in `DEFAULT_CANDIDATE_SPECS` (5 specs),
3. Lagrangian water-fills `(distortion, bits)` across tensors to a global bpw budget.

It varies only the **spec**, on the **raw** distribution. We want to also vary
`normalization` and `rotation` per tensor. The naive way - full k-means for every
grid point - is the blocker:

```
grid = specs(5) × norms(3) × rotations(2) = 30 full k-means probes/tensor
     vs 5 today  →  ~6x slower allocate
```

A cheap proxy avoids that.

## Design: two-stage search per tensor

```
Stage 1  cheap transform ranking   (no k-means, all grid points)
Stage 2  full RVQ probe            (existing k-means, survivors only)
```

```
for each tensor (full weights, not a sample):
    # Stage 1 - rank transforms with a scalar-quant SQNR proxy
    for (norm, rot) in TRANSFORM_GRID:                 # ~6 configs, O(N) / O(N log N)
        Wt        = apply_transform(W, norm, rot)       # block-max scale / Hadamard
        proxy_mse = scalar_quant_proxy(Wt, norm, rot)   # measured in ORIGINAL space
    survivors = argmin_k proxy_mse                       # k = 2 (not 1; see Risks)

    # Stage 2 - full RVQ probe only on survivor transforms × spec menu
    for (norm, rot) in survivors:
        sample = sample_vectors(apply_transform(W, norm, rot))
        for spec in specs:
            dist = _probe_spec_distortion(sample, spec, ...)   # existing, ORIGINAL space
            bits = index_bits(spec) + transform_overhead_bits(norm, rot)
            candidates.append((dist * numel, bits, spec, norm, rot))
    rows.append({name, numel, candidates})

# Lagrangian (existing solver, generalized to arbitrary candidates):
#   each tensor independently picks argmin_c (distortion_c + λ·bits_c); binary-search λ.
# emit per-tensor {stages, normalization, rotation, bpw}
```

Cost: `6 cheap proxies + 2 survivors × 5 specs = ~10 full probes/tensor` vs 5 today
→ **~2x**, not 6x. Stage 1 proxies are O(N) (or O(N log N) for Hadamard), dominated
by the k-means in Stage 2.

## The scalar-quant proxy

Per (norm, rot) config, after transforming the full tensor:

```
scalar_quant_proxy(Wt, bits=4):
    s   = max_abs_per_block(Wt) / (2^(bits-1) - 1)      # symmetric uniform, per block
    q   = clip(round(Wt / s), -lim, lim) * s            # b-bit scalar reconstruction
    return distortion_in_original_space(W, q, norm, rot)
```

Why it predicts VQ distortion: scalar-quant MSE tracks how "quantizable" a
distribution is - a rotation that suppresses outliers (Hadamard → Gaussianizes)
lowers scalar MSE *and* VQ MSE; block-max that equalizes inter-block scale lowers
both. It is a rank proxy, not an absolute predictor - Stage 2 supplies the real
numbers for the survivors.

### Correctness: distortion must be in ORIGINAL weight space

Comparing distortion across transforms is only valid in the original units:
- **rotation is orthogonal (isometric)** → MSE in rotated space == MSE in original
  space, so no un-rotate needed.
- **normalization is NOT isometric** (it scales) → the reconstruction must be
  **denormalized** before the MSE, else block-max looks artificially good (smaller
  numbers → smaller raw MSE). With block scale `s_b`: `MSE_orig = Σ_b s_b² · MSE_norm,b`.

### Correctness: transform on full tensor, sample after

Block-max normalization is block-structured over the **flattened tensor**
(`block_scale_size=128`). A random vector sample breaks block contiguity, so the
transform is computed on the **full** tensor first (block-max = reshape + per-block
max, O(N); Hadamard = O(N log N)), *then* sampled for the proxy / probe. Transforms
are not the bottleneck; k-means is.

## Rate accounting

`bits` per candidate must include each transform's storage overhead, or the
Lagrangian mis-spends:

| transform | extra bytes |
|---|---|
| `none` | 0 |
| `block-max` | `n_blocks × scale_bytes` |
| `slrq-block` | salient fp16 sidecar |
| rotation (`hadamard`/`orthogonal`) | 0 on disk (seed only); decode compute only |
| `outlier_frac` (future axis) | fp16 outlier sidecar |

The existing Lagrangian (`allocate.py:180`) already consumes arbitrary
`(distortion, bits)` candidates - it just needs the generalized candidate list and
a correct `_bits` that adds `transform_overhead_bits`.

## Grid scope (first version)

| axis | values | note |
|---|---|---|
| normalization | `none`, `block-max`, `slrq-block` | the high-value three |
| rotation | `none`, `hadamard` | **skip orthogonal QR in the search** - O(N²)/tensor is too costly to probe; allow it only as a global flag |
| spec | existing `DEFAULT_CANDIDATE_SPECS` | unchanged |

`group_size` and `outlier_frac` are deferred: `group_size` conflicts with shared
codebooks (`pack.py:246-250`) and outlier extraction lives in a different scope
(`pack.py:619`).

## Emit / wire

Extend the allocation JSON per tensor with `normalization` + `rotation` (the
reader `allocation_tensor_transforms`, `allocate.py:227`, already consumes them;
`cmd_pack` already loads and passes the map). So Stage-2 output just needs to write
those two fields next to `stages`.

## Validation plan (needs RAM)

1. **Proxy fidelity gate (the make-or-break):** on ~20 real tensors, compute
   Spearman correlation between the proxy's transform ranking and the full-k-means
   ranking. Require the proxy's top-2 to contain the true best ≥ 90% of the time.
   If it fails, the proxy is wrong - widen survivors or switch proxy before trusting it.
2. **End-to-end:** pack a model with searched per-tensor transforms vs one global
   transform at matched bpw; expect ≥ matched ppl at ≤ size.
3. **Structural oracle** unchanged for the default path (search is opt-in via a new
   `allocate` flag; bare `pack` is untouched).
4. **Cost:** `allocate` wall-time ≤ ~2x current.

## Risks

| risk | mitigation |
|---|---|
| Proxy mis-ranks (VQ captures group correlations scalar quant can't) | top-**2** survivors into Stage 2, not top-1; fidelity gate before trusting |
| Orthogonal QR too expensive to probe | excluded from the search grid; Hadamard only |
| Transform overhead bits wrong → Lagrangian over/under-spends | unit-test `_bits` against known sidecar sizes |
| Search-cost regression on huge models | Stage-1 proxy is O(N); cap survivors at 2; specs unchanged |

## Build order

1. `scalar_quant_proxy` + `distortion_in_original_space` helpers (pure, unit-testable, no pack).
2. `transform_overhead_bits(norm, rot, tensor)` (pure, unit-testable).
3. Generalize the candidate list + `_bits` in `build_allocation`; emit transform fields.
4. New `--search-transforms` flag on `allocate` (opt-in; default = today's spec-only search).
5. Proxy-fidelity test + end-to-end pack comparison (RAM).
