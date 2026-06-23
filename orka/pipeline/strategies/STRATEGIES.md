# Compression strategies

Every trick `pack_checkpoint` composes onto the base RVQ codec, what enables it, where the code lives, and where it is wired into the pipeline. This is the human-readable rendering of `STRATEGY_REGISTRY` in `__init__.py` - keep both in sync.

## Catalog

| Strategy | Enabled by | Code | Wired at | Effect |
|---|---|---|---|---|
| **rvq** | always (`codebook_size` / `codebook_sizes`) | `orka.codebook` | per-tensor stage loop | base bpw = `ceil(log2 k)` per stage / `group_size` |
| **normalization** | `normalization=` | `orka.transforms.normalize` | pre-stage (`_apply_normalization`) | tighter fit; adds a scale sidecar |
| **rotation** | `rotation=` | `orka.transforms.rotate` | pre-vectorize (`_rotate_tensor_to_2d`) | spreads outliers; stores a seed, no extra bpw |
| **outliers** | `outlier_frac>0` | `orka.transforms.outliers` (`_extract_outliers`) | pre-stage | exact-stores worst weights; sparse sidecar |
| **salient** | `slrq_salient` + `normalization=slrq-block` | `orka.transforms.normalize` (slrq-block emits salient w/i) | pre-stage (inside `_apply_normalization`) | one exact weight per block; small sidecar |
| **hessian_weighting** | `awq_activations` provided | `orka.pipeline.pack` (digest in `pack_helpers`) | before stage learning | weighted k-means; no extra bpw |
| **error_compensation** | `error_compensation=True` (torch, `rotation=none`, activations) | `strategies.error_compensation` | post-assignment; skips EM-AQ when applied | lower output error, rewrites index stream, no extra bpw |
| **em_aq** | `em_aq_passes>0` | `strategies.refinement` | after greedy stage loop | tightens multi-stage fit, no extra bpw |
| **mse_scale** | `mse_scale=True` (`rotation=none`, block-max family) | `strategies.refinement` | after refinement | LS-optimal block scale, strictly lower error, no extra bpw |

## Pipeline wiring order (per tensor)

```
load -> rotation -> normalization -> outliers/salient extract -> hessian weights
     -> RVQ stage loop (learn + quantize)
     -> error_compensation  (if applied, skip em_aq)
     -> em_aq               (else)
     -> mse_scale
     -> persist (codebooks, indices, scale + sparse sidecars, manifest)
```

## Pluggable post-assignment strategies (Strategy pattern)

The post-assignment steps (error_compensation, em_aq, mse_scale) are **plugins**, not
hardcoded calls. Each is a `PostAssignmentStrategy` (`base.py`) with:

- `applies(ctx, c) -> bool` - does this strategy run for this candidate + config
- `apply(ctx, c) -> None` - run it, mutating the candidate in place

`pack_pipeline` applies `POST_ASSIGNMENT_STRATEGIES` (ordered list in `__init__.py`) with a
single generic loop:

```python
for strategy in POST_ASSIGNMENT_STRATEGIES:
    if strategy.applies(ctx, c):
        strategy.apply(ctx, c)
```

Order is load-bearing: error_compensation first (sets `c["_compensated"]`), then em_aq
(its `applies` returns False when compensated), then mse_scale. The dependency lives in the
gates, not the loop - so the loop never changes.

## Adding a post-assignment strategy

1. Subclass `PostAssignmentStrategy` (in `strategies/`), implement `applies` + `apply`
   (read config from `ctx`, duck-typed - do not import `PackCtx`).
2. Append an instance to `POST_ASSIGNMENT_STRATEGIES` in `__init__.py` at the right
   position. **No edit to `pack_pipeline` or `pack_checkpoint`.**
3. Add a row to the catalog above + an entry to `STRATEGY_REGISTRY`.
4. Add an oracle config / unit test that exercises it, so the wiring is regression-covered.

Base transforms (normalization / rotation / outliers / RVQ) are not post-assignment
strategies - they run pre-stage and stay in `transforms` / `codebook`, catalogued by
reference above.

The **normalization** axis is itself pluggable: `orka.transforms.normalize` keeps a
`NORMALIZATION_REGISTRY` (mode string -> handler returning a `NormalizationResult`).
`_apply_normalization` is a registry lookup, so a new mode plugs in via
`register_normalization("my-mode", handler)` - no edit to the dispatcher (open/closed).
Unknown / `none` falls through to a passthrough handler.

The **rotation** axis is pluggable the same way: `orka.transforms.rotate` keeps a
`ROTATION_REGISTRY` (mode -> `RotationStrategy(name, rotate, unrotate)`), so a new
invertible rotation registers via `register_rotation(RotationStrategy(...))` - the
`_rotate_tensor_to_2d` / `_unrotate_flat` dispatchers do not change. `none` is identity.
