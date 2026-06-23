# Compression strategies

Every trick `pack_checkpoint` composes onto the base RVQ codec, what enables it, where the code lives, and where it is wired into the pipeline. This is the human-readable rendering of `STRATEGY_REGISTRY` in `__init__.py` - keep both in sync.

## Catalog

| Strategy | Enabled by | Code | Wired at | Effect |
|---|---|---|---|---|
| **rvq** | always (`codebook_size` / `codebook_sizes`) | `orka.codebook` | per-tensor stage loop | base bpw = `ceil(log2 k)` per stage / `group_size` |
| **normalization** | `normalization=` | `orka.transforms.normalize` | pre-stage (`_apply_normalization`) | tighter fit; adds a scale sidecar |
| **rotation** | `rotation=` | `orka.transforms.rotate` | pre-vectorize (`_rotate_tensor_to_2d`) | spreads outliers; stores a seed, no extra bpw |
| **outliers** | `outlier_frac>0` | `orka.transforms` (`_extract_outliers`) | pre-stage | exact-stores worst weights; sparse sidecar |
| **salient** | `slrq_salient` + `normalization=slrq-block` | `orka.pipeline.pack` | pre-stage | one exact weight per block; small sidecar |
| **hessian_weighting** | `awq_activations` provided | `orka.pipeline.pack` | before stage learning | weighted k-means; no extra bpw |
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

## Adding a strategy

1. Implement it (in `strategies/` if it is a pack-time step; in `transforms`/`codebook` if it is a base transform).
2. Wire the call into `pack_checkpoint` at the right stage.
3. Add an entry to `STRATEGY_REGISTRY` and a row here.
4. Add an oracle config in the golden test or a unit test that exercises it, so the wiring is regression-covered.
