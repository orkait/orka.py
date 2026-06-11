# ORKA: Core Truth

## Purpose

Orka is Orkait's CPU-first compressed language model project. The goal is to turn an existing Hugging Face causal language model into a compact, runtime-aware artifact that can later be carried inside GGUF and executed by custom Orka kernels.

The project is not a claim of solved intelligence, speed, or compression. Every number must come from measurement or from an explicit theoretical formula.

## Product Target

Orka targets three related artifacts:

1. A compiler that inspects and packs model weights into vector-codebook indices.
2. A packed research artifact that records indices, codebooks, scales, tensor layout, and error metrics.
3. A future GGUF quantization type that lets llama.cpp-compatible tools run Orka-packed models after runtime support exists.

## Baseline Compression Geometry

Let:

- `N` = number of weights.
- `G` = weights represented by one codebook index.
- `K` = number of codebook entries.
- `B = ceil(log2(K))` = bits per index.
- `V = ceil(N / G)` = number of vectors.

Then:

```text
index_bytes = ceil(V * B / 8)
bits_per_weight = B / G
```

Baseline Orka VQ8:

```text
G = 8
K = 256
B = 8
bits_per_weight = 1.0
```

For an 8.03B parameter model:

```text
V = ceil(8.03B / 8) = 1.00375B vectors
index_bytes = 1.00375 GB before scales, codebooks, and metadata
```

## Format Direction

The first implementation writes an internal `.orka` directory, not a production GGUF file. The internal artifact is intentionally simple:

```text
manifest.json
codebooks/
  global.codebook.f32                # global mode, single-stage
  global.s{i}.codebook.f32           # global mode, multi-stage (RVQ)
  <family>.codebook.f32              # family mode, single-stage
  <family>.s{i}.codebook.f32         # family mode, multi-stage
tensors/
  <safe tensor name>.indices         # single-stage indices
  <safe tensor name>.s{i}.indices    # multi-stage indices, per stage
  <safe tensor name>.codebook.f32    # per-tensor mode, single-stage
  <safe tensor name>.s{i}.codebook.f32  # per-tensor mode, multi-stage
  <safe tensor name>.block_max_scale.f32 # block-max / channel-block-max / slrq-block / awq-block-max scales
  <safe tensor name>.awq_col_scale.f32   # awq-block-max column scales, when calibration exists
  <safe tensor name>.outliers.idx       # outlier escape: uint32 positions
  <safe tensor name>.outliers.val       # outlier escape: float16 values
  <safe tensor name>.salient.idx        # slrq-block salient local indices
  <safe tensor name>.salient.val        # slrq-block salient weights
  <safe tensor name>.pillars.idx        # frequency-aware protected positions
  <safe tensor name>.pillars.f2         # frequency-aware protected values
passthrough.safetensors              # skipped tensors such as norms and biases
```

Stage-suffixed files (`.s0`, `.s1`, ...) appear only when N-stage RVQ is used (more than one codebook per tensor). Single-stage VQ keeps the legacy unsuffixed names.

`codebooks/<key>.codebook.f32` exists only when global or family codebook mode is used.
Per-tensor mode stores each tensor codebook next to its index file.
`block_max_scale.f32` exists for `block-max`, `channel-block-max`, `slrq-block`, and `awq-block-max`.
`awq_col_scale.f32` exists for `awq-block-max` when AWQ calibration data exists.
`salient.idx` + `salient.val` exist when `slrq-block` salient extraction is enabled.
`pillars.idx` + `pillars.f2` exist when a sensitivity map provides protected top tokens.
`outliers.idx` + `outliers.val` exist only when `--outlier-frac > 0`.

Index files are bit-packed when the stage bit width is not byte-aligned. Byte-aligned stages are stored as little-endian unsigned integers sized by stage bits: uint8 (<=8 bits), uint16 (<=16), uint32 (<=32), uint64 (<=64).

The manifest must include:

- Orka format version.
- Source path.
- Group size.
- Codebook sizes (one or more, per stage).
- N stages (1 = pure VQ, ≥2 = RVQ).
- Codebook mode (per-tensor / global / family).
- Family stages map (only when `rvq-mixed` is used).
- Normalization (`none`, `block-max`, `channel-block-max`, `awq`, `awq-block-max`, or `slrq-block`).
- Outlier fraction.
- Rotation type (`none`, `orthogonal`, or `hadamard`) + global seed.
- Per-tensor entries: shape, packed/padded values, vector count, training vector count, group size, codebook size (stage 0), index bits (stage 0), index bytes (sum across stages), `n_stages`, `stages: [...]` list with per-stage codebook + index paths and bits, `total_bits_per_vector`, error metrics (MSE/RMSE/MAE/relative_rmse/cosine_similarity/source_l2_sq/reconstructed_l2_sq), normalization, scale path + count + bytes, outlier metadata, salient metadata, pillar metadata, and rotation seed.

Codebooks are written as raw little-endian float32 values in row-major order.
The manifest owns the dimensions and meaning of the file.

GGUF remains the public adoption target, but runtime compatibility requires a future Orka quantization type such as `Q_ORKA_VQ8` or `Q_ORKA_VQ16`.

## Input Tensor Dtypes

Safetensors inputs are loaded through PyTorch when it is available. This allows Orka to inspect and pack common model dtypes such as FP64, FP32, FP16, BF16, signed integers, unsigned 8-bit integers, and bool tensors. Orka normalizes tensor values to float32 for codebook learning, quantization, verification, and reconstruction.

## GPU Support

Orka has two GPU paths:

- `pack` and `sweep` can use PyTorch for codebook learning and nearest-codebook assignment with `--backend torch --device cuda`.
- `eval` and `eval-sweep` can use CUDA for Hugging Face model scoring with `--device cuda`.

The default compiler path is still CPU-first:

- `--backend auto` and `--backend numpy` use CPU NumPy for packing.
- `--backend torch --device cpu` uses PyTorch on CPU.
- `--backend torch --device auto` uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU.

Example GPU pack:

```bash
python3 orka.py pack model.safetensors \
  --out model.orka \
  --group-size 8 \
  --codebook-size 256 \
  --codebook-mode family \
  --backend torch \
  --device cuda \
  --sample-vectors 65536
```

Example GPU sweep:

```bash
python3 orka.py sweep model.safetensors \
  --out /tmp/orka-sweep.json \
  --group-sizes 8 16 \
  --codebook-sizes 256 \
  --codebook-modes global family \
  --normalizations none \
  --backend torch \
  --device cuda \
  --sample-vectors 65536 \
  --iterations 8
```

CUDA support requires a working PyTorch CUDA install. If CUDA is requested and unavailable, Orka fails instead of silently falling back to CPU.

## Compiler Scope v0

`orka.py` owns the research compiler prototype:

- `calc`: estimate payload size from parameter count and quantization geometry.
- `inspect`: report candidate tensors in a local safetensors or PyTorch checkpoint.
- `pack`: learn a small vector codebook per tensor and write packed indices plus metadata.
- `report`: summarize a packed `.orka` artifact, including index size, codebook size, weighted MSE, and worst tensors.
- `verify`: decode a packed `.orka` artifact against its source checkpoint and recompute MSE from stored indices and codebooks.
- `reconstruct`: decode a packed `.orka` artifact to JSON or safetensors tensors for inspection and downstream experiments.
- `sweep`: run a matrix of pack/report experiments and write one JSON comparison file.
- `eval`: reconstruct an `.orka` artifact into a temporary Hugging Face model directory and compare prompt loss against the original model.
- `eval-sweep`: run `eval` across artifacts recorded in a sweep JSON and rank candidates by loss delta, perplexity ratio, and artifact bytes.

The compiler does not claim model IQ or tokens per second. `report`, `verify`, and `sweep` measure reconstruction error. `eval` and `eval-sweep` can measure prompt loss/perplexity when optional Hugging Face dependencies and a causal language model directory are available.

## Normalization Modes

`pack` supports these normalization modes:

- `none`: quantizes raw flattened tensor values.
- `block-max`: divides each block by its max absolute value, stores one float32 block scale, and multiplies by that scale during decode.
- `channel-block-max`: aligns block-max scaling to the tensor's inner channel dimension when possible, with flat block-max fallback for unsupported shapes.
- `slrq-block`: keeps a power-of-two block anchor, optionally extracts one salient value per block, and restores salient values after decode scaling.
- `awq`: activation-aware column scaling. It is feature-gated behind `ORKA_ENABLE_AWQ=1`.
- `awq-block-max`: torch-only AWQ column scaling followed by block-max scaling. If a tensor has no AWQ activation entry, it falls back to block-max for that tensor.

`block-max`, `channel-block-max`, and `slrq-block` add `4 * block_count` bytes per tensor for block scales. `slrq-block` also stores salient sidecars unless disabled with `--no-slrq-salient`.
Use `awq` and `awq-block-max` only when calibration activations are available and AWQ is explicitly enabled.

Example:

```bash
python3 orka.py pack model.safetensors \
  --out model-slrq.orka \
  --group-size 8 \
  --codebook-size 256 \
  --codebook-mode per-tensor \
  --backend numpy \
  --sample-vectors 65536 \
  --normalization slrq-block
```

## Quality Metrics

`report` and `verify` expose tensor and aggregate reconstruction metrics:

- `weighted_mse`: mean squared reconstruction error across packed values.
- `rmse`: square root of MSE, in the original tensor value scale.
- `relative_rmse`: RMSE normalized by source tensor norm. This is usually more useful than raw MSE across tensors with different scales.
- `mae`: mean absolute reconstruction error.
- `max_abs_error`: largest absolute reconstruction error.
- `cosine_similarity`: full-vector cosine similarity between source and reconstructed values.

Raw MSE is not enough for embeddings or tensors with large value scale. Treat relative RMSE and cosine similarity as the first-pass quality signals until behavior-level evaluation exists.

## Measurement Loop

The next engineering loop is:

```text
inspect checkpoint -> pack checkpoint -> report artifact -> verify artifact -> reconstruct tensors -> compare settings
```

Example:

```bash
python3 orka.py calc --params 8.03b --group-size 8 --codebook-size 256
python3 orka.py inspect model.safetensors
python3 orka.py pack model.safetensors --out model.orka --group-size 8 --codebook-size 256
python3 orka.py report model.orka
python3 orka.py verify model.orka
python3 orka.py reconstruct model.orka --out reconstructed.json
python3 orka.py reconstruct model.orka --out reconstructed.safetensors --format safetensors
```

Use `--max-values-per-tensor` during early experiments to test the compiler on tensor samples before packing full checkpoints.
JSON reconstruction is dependency-free. Safetensors reconstruction requires optional `numpy` and `safetensors` packages.

For larger local safetensors checkpoints, use sampled codebook training. Use the torch backend when a CUDA GPU is available:

```bash
python3 orka.py pack model.safetensors \
  --out model.orka \
  --group-size 8 \
  --codebook-size 256 \
  --codebook-mode global \
  --backend torch \
  --device cuda \
  --sample-vectors 65536
```

`--sample-vectors` limits only the codebook training set. Orka still quantizes every packed vector when writing indices and reporting MSE.

Use `sweep` when comparing settings:

```bash
python3 orka.py sweep model.safetensors \
  --out /tmp/orka-sweep.json \
  --group-sizes 4 8 16 \
  --codebook-sizes 256 \
  --codebook-modes global family per-tensor \
  --normalizations none block-max channel-block-max slrq-block \
  --backend torch \
  --device cuda \
  --sample-vectors 65536 \
  --iterations 8
```

The sweep summary records every artifact path plus artifact size, index bytes, codebook bytes, scale bytes, weighted MSE, relative RMSE, cosine similarity, and `cosine_per_mb`. It also records best candidates by cosine similarity, relative RMSE, and cosine per MB.

Use `eval` after sweep has identified one candidate artifact:

```bash
python3 orka.py eval /tmp/orka-sweep.artifacts/g8-k256-global-block-max.orka \
  --prompts prompts.txt \
  --out /tmp/orka-eval.json \
  --model-dir /path/to/hf-model-dir \
  --max-length 512 \
  --device cpu
```

`eval` expects one prompt per non-empty line. It loads the original model and reconstructed Orka model with `transformers`, computes causal language model loss on the same prompts, and writes original loss, Orka loss, loss delta, original perplexity, Orka perplexity, and perplexity ratio. The reconstructed eval checkpoint is complete: packed tensors are decoded from Orka, while skipped tensors such as biases and normalization weights are copied unchanged from the source checkpoint. It is intentionally local-first: by default it passes `local_files_only=True` to Hugging Face. Use `--allow-download` only when downloading missing model files is intended.

Use `eval-sweep` when the sweep has multiple candidates and behavior-level ranking matters:

```bash
python3 orka.py eval-sweep /tmp/orka-sweep.json \
  --prompts prompts.txt \
  --out /tmp/orka-eval-sweep.json \
  --model-dir /path/to/hf-model-dir \
  --max-length 512 \
  --device cpu
```

`eval-sweep` writes individual eval JSON files under `<out stem>.evals/` and one aggregate JSON with `best_by_loss_delta`, `best_by_perplexity_ratio`, and `best_by_artifact_bytes`. Use `--max-runs N` for a quick subset. By default reconstructed Hugging Face model directories are temporary; use `--reconstructed-model-root` only when those directories need to be inspected after the run.

## Codebook Modes

Orka v0 supports three codebook modes:

- `per-tensor`: learns one codebook per tensor. This usually lowers tensor-level reconstruction error but increases metadata/codebook overhead.
- `global`: learns one shared codebook across all packed tensors. This reduces codebook overhead and is easier to reason about as a runtime format, but can raise reconstruction error.
- `family`: learns one shared codebook per tensor family. This is the middle-ground mode for testing whether a few runtime-friendly codebooks can retain most of the quality of per-tensor mode.

Family mode uses deterministic name-based routing:

- `embedding`: tensor names containing `embed`, `embedding`, `wte`, or `wpe`.
- `attention`: tensor names containing `attn`, `attention`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, or `c_attn`.
- `mlp`: tensor names containing `.mlp.`, `mlp`, `gate_proj`, `up_proj`, `down_proj`, or `c_fc`.
- `other`: fallback for everything else.

Example:

```bash
python3 orka.py pack model.safetensors --out model-global.orka --group-size 8 --codebook-size 256 --codebook-mode global
python3 orka.py pack model.safetensors --out model-family.orka --group-size 8 --codebook-size 256 --codebook-mode family
```

## Quant Spec Naming

Quantization layouts use compositional spec strings:

- `vq-{bits}` - single-stage VQ. Codebook size = `2^bits`. E.g., `vq-8` (k=256), `vq-16` (k=65536).
- `rvq-{b1}-{b2}-...{bn}` - N-stage Residual VQ. Each stage learns a codebook on the residual of previous stages. Decode = sum of stage centroid lookups. E.g., `rvq-16-8` (stage 1 k=65536, stage 2 k=256), `rvq-16-16-16-16` (4 stages of k=65536).
- `rvq-mixed` - preset for mixed precision per family: embedding gets `rvq-16-16-16` (6 bpw), attention `rvq-16-8` (3 bpw), mlp `rvq-16-8` (3 bpw), other `vq-16` (2 bpw). Forces `--codebook-mode per-tensor`.

Constraints: per-stage bits 1..64, total bits/vector ≤ 64. Stages > 1 require the `rvq-` prefix; single stage requires the `vq-` prefix. Practical sweet spot remains 8-16 bits per stage; beyond 16, k-means training cost grows fast for marginal quality gain (extra stages cheaper).

`bits_per_weight = sum(stages) / group_size`. At g=8: `vq-8` is 1.0 bpw, `rvq-16-8` is 3.0 bpw, `rvq-16-16-16-16` is 8.0 bpw.

## Outlier Escape

`--outlier-frac F` extracts the top-F-fraction of weights by magnitude per tensor and stores them as a fp16 sidecar with compact integer positions. The corresponding positions are zeroed in the source before VQ training, so the codebook is not pulled by extreme values. At decode time outliers are re-injected at their saved positions before normalization is undone.

Recommended values: 0.001 (0.1%) to 0.005 (0.5%). Storage overhead depends on the selected position dtype plus the value dtype. Critical for low-bpw modes - a few outliers per tensor dominate matmul output and stretch VQ centroids if not escaped.

## Rotation

`--rotation orthogonal` applies a per-tensor random orthogonal rotation along the inner axis (cols) before VQ. The rotation is generated deterministically from `--rotation-seed` xor blake2b hash of the tensor name, so no rotation matrices are stored - the manifest only carries the global seed and per-tensor seed for reproducibility.

Effect: rotation smears outliers across all dimensions, making post-rotation distributions closer to gaussian. VQ centroids fit a near-isotropic distribution more tightly. Storage overhead: 8 bytes per tensor (seed). Compute overhead: one QR factorization per tensor at pack time + one rotation matmul; a second matmul at decode time. Inspired by SpinQuant / QuaRot.

`--rotation hadamard` applies a block-diagonal FWHT along the inner axis. It uses the full axis when the inner dimension is a power of two, otherwise the largest usable power-of-two divisor.

Pipeline applied during `pack`:
1. Optional normalization (`none`, `block-max`, `channel-block-max`, `awq`, `awq-block-max`, or `slrq-block`).
2. Optional rotation (`none`, `orthogonal`, or `hadamard`).
3. Optional outlier escape (`--outlier-frac F`).
4. RVQ stages (one or more codebooks).

Decode reverses the pipeline: VQ stages -> re-inject outliers -> un-rotate -> un-normalize.

## Memory Cap

`--max-gpu-mem-gb GB` (pack and sweep) calls `torch.cuda.set_per_process_memory_fraction` to cap PyTorch's per-process allocator. Exceeding the cap raises `CappedOutOfMemoryError` (subclass of `RuntimeError`) instead of crashing the kernel or exceeding the user's budget. The `--backend torch` cdist chunking is adaptive: `chunk = max(256, min(default, (1 << 28) // k))` so the squared-distance matrix never exceeds ~256 MB. This keeps single allocations safely under the cap regardless of `k`.

## Non-Goals v0

- No runtime inference engine.
- No production GGUF runtime quantization type yet.
- No quantization-aware training loop.
- No public performance claims.

Previously listed non-goals that are now implemented:

- Orthogonal rotation (`--rotation orthogonal`) - replaces "no SpinQuant rotation" item.

## Open Research Questions

- Which model size is best under a 2.1 GB ceiling: 8B, 14B, or 27B with larger vector groups?
- Are per-tensor codebooks enough, or do sensitive layers need per-channel or escape-codebook support?
- Does VQ8 preserve enough quality after quantization-aware fine-tuning?
- Is VQ16 the better first public target because `uint16` indices are simpler and higher quality?
- What GGUF metadata and tensor layout are needed for fast Orka runtime kernels?
