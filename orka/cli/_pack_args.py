"""Argument spec for the `pack` subcommand (extracted from cli.parser).

A large, self-contained block of argparse options; kept separate so parser.py
stays scannable."""
from __future__ import annotations

def _add_pack_args(p):
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=None,
        help="explicit per-stage codebook sizes (overrides --codebook-size and --quant-mode)",
    )
    p.add_argument(
        "--quant-mode",
        default=None,
        help="compositional spec like vq-8 or rvq-16-8 (per-stage bits 1..64, total <= 64; multi-stage requires the rvq- prefix)",
    )
    p.add_argument(
        "--codebook-mode",
        choices=["per-tensor", "global", "family"],
        default="per-tensor",
    )
    p.add_argument(
        "--backend", choices=["auto", "numpy", "torch"], default="auto"
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch backend device, for example cpu, cuda, cuda:0, or auto",
    )
    p.add_argument(
        "--normalization",
        choices=["none", "block-max", "channel-block-max", "awq", "awq-block-max", "slrq-block"],
        default="none",
    )
    p.add_argument(
        "--block-scale-size",
        type=int,
        default=32,
        help="elements per block when --normalization block-max (typical 16 or 32)",
    )
    p.add_argument(
        "--rotation",
        choices=["none", "orthogonal", "hadamard"],
        default="none",
        help="rotation along inner axis before VQ. orthogonal: per-tensor seeded random orthogonal (any size). hadamard: block-diagonal FWHT (uses largest pow2 divisor of last dim; full FWHT if last dim is pow2).",
    )
    p.add_argument(
        "--em-aq-passes",
        type=int,
        default=3,
        help="number of EM-AQ joint refinement passes after greedy RVQ. 0 disables.",
    )
    p.add_argument(
        "--no-slrq-salient",
        dest="slrq_salient",
        action="store_false",
        default=True,
        help="disable salient-weight extraction inside slrq-block (keeps power-of-2 anchor only).",
    )
    p.add_argument(
        "--tensor-partition-count",
        type=int,
        default=None,
        help="run only this partition of quantizable tensors for multi-GPU/CPU partitioning",
    )
    p.add_argument(
        "--tensor-partition-index",
        type=int,
        default=None,
        help="zero-based partition index, only used with --tensor-partition-count",
    )
    p.add_argument(
        "--partition-worker-count",
        type=int,
        default=1,
        help="max concurrent partition workers when orchestrating multi-GPU Kaggle packing",
    )
    p.add_argument(
        "--rotation-seed",
        type=int,
        default=None,
        help="seed for orthogonal rotation (deterministic)",
    )
    p.add_argument("--sample-vectors", type=int, default=None)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--max-values-per-tensor", type=int, default=None)
    p.add_argument(
        "--max-gpu-mem-gb",
        type=float,
        default=None,
        help="strict cap on per-process GPU memory (GB)",
    )
    p.add_argument(
        "--max-system-ram-gb",
        type=float,
        default=None,
        help="strict cap on total system RAM (GB), enforced by the 100ms RSS poll-monitor. Hard ceiling 25GB.",
    )
    p.add_argument(
        "--workload-budget-gb",
        type=float,
        default=None,
        help="estimated process RAM budget (GB) used by preflight check. "
             "Typical: SmolLM2=5, Pythia=5, Bloom=7, Qwen3-0.6B=9. "
             "Required when --max-system-ram-gb is set.",
    )
    p.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="cap CPU threads (torch + OMP/MKL/OPENBLAS env + sched_setaffinity).",
    )
    p.add_argument(
        "--outlier-frac",
        type=float,
        default=0.0,
        help="fraction of top-magnitude weights kept as fp16 sidecar (e.g. 0.001 = 0.1%%)",
    )
    p.add_argument(
        "--awq-calibration",
        default=None,
        help="prompts file for AWQ calibration; enables activation-aware VQ",
    )
    p.add_argument(
        "--awq-model-dir",
        default=None,
        help="HF model dir for AWQ activation collection",
    )
    p.add_argument(
        "--awq-activations-file",
        default=None,
        help="JSON file containing pre-calculated AWQ activations to reuse",
    )
    p.add_argument(
        "--awq-alpha",
        type=float,
        default=0.5,
        help="activation magnitude scaling power (default 0.5)",
    )
    p.add_argument("--calibration-max-prompts", type=int, default=32)
    p.add_argument("--calibration-max-length", type=int, default=256)
    p.add_argument(
        "--calibration-max-samples",
        type=int,
        default=4096,
        help="max activation samples retained per layer for AWQ calibration",
    )
    p.add_argument(
        "--no-hessian",
        action="store_true",
        help="disable default Hessian-weighting (skip auto activation collection "
        "from the bundled calibration corpus); packs unweighted and faster",
    )
    p.add_argument("--progress-file", help="file to write real-time progress status")
    p.add_argument(
        "--sensitivity-map",
        help="JSON file from sensitivity.py to enable mixed-precision",
    )
    p.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="limit pack to first N tensors (for fail-fast iteration)",
    )
    p.add_argument(
        "--only-tensors",
        nargs="+",
        default=None,
        help="list of exact tensor names to process; skip all others",
    )
    p.add_argument(
        "--codebook-cache",
        default=None,
        help="dir to cache stage-0 codebooks (zero-loss reuse on identical configs)",
    )
    p.add_argument(
        "--codebook-dtype",
        choices=["float16", "int8", "float32"],
        default="float16",
        help="on-disk codebook precision. int8 (per-column symmetric) halves "
             "codebook bytes - biggest win on small models where codebooks "
             "are ~25%% of the artifact",
    )
    p.add_argument(
        "--error-compensation",
        action="store_true",
        help="GPTQ-style block-OBS error compensation: column groups are "
             "re-assigned left-to-right with committed error propagated into "
             "remaining columns via the calibration Hessian inverse. Needs "
             "--awq-calibration activations, torch backend, rotation none; "
             "replaces EM-AQ on compensated tensors.",
    )
    p.add_argument(
        "--mse-scale",
        action="store_true",
        help="MSE-optimal block scales: after quantization, replace each block's "
             "max scale with the least-squares-optimal scale for its assigned "
             "codewords (excludes salient/outlier positions). Free quality gain "
             "(no extra bits, no inference cost). rotation none + block-max-family "
             "normalization only.",
    )
    p.add_argument(
        "--allocation-map",
        default=None,
        help="JSON from 'orka allocate': per-tensor measured stage specs "
             "(requires --codebook-mode per-tensor; disables family group sizing)",
    )
