"""argparse parser construction. Each subcommand registered with its cmd_*.

SLRQ subcommand removed (parallel path replaced by --normalization slrq-block in pack).
"""

from __future__ import annotations

import argparse

from orka.cli.commands import (
    cmd_calc,
    cmd_eval,
    cmd_eval_sweep,
    cmd_inspect,
    cmd_kaggle_pack,
    cmd_pack,
    cmd_pulse_check,
    cmd_reconstruct,
    cmd_report,
    cmd_sem_analyze,
    cmd_sweep,
    cmd_verify,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orka model compiler prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    calc = sub.add_parser("calc", help="estimate Orka payload size")
    calc.add_argument(
        "--params", required=True, help="parameter count, for example 8.03b"
    )
    calc.add_argument("--group-size", type=int, default=8)
    calc.add_argument("--codebook-size", type=int, default=256)
    calc.add_argument("--scale-block-vectors", type=int, default=64)
    calc.add_argument("--scale-bits", type=int, default=16)
    calc.set_defaults(func=cmd_calc)

    inspect = sub.add_parser(
        "inspect", help="inspect a safetensors or PyTorch checkpoint"
    )
    inspect.add_argument("source")
    inspect.set_defaults(func=cmd_inspect)

    def add_pack_args(p):
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
            help="compositional spec like vq-8 or vq-16-8 (per-stage bits, 1..16, total ≤ 64)",
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
            choices=["none", "block-max", "awq", "awq-block-max", "slrq-block"],
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
            help="strict cap on total system RAM (GB). RLIMIT_AS-enforced. Hard ceiling 25GB.",
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
            "--codebook-cache",
            default=None,
            help="dir to cache stage-0 codebooks (zero-loss reuse on identical configs)",
        )

    pack = sub.add_parser(
        "pack", help="pack candidate weight tensors into an .orka directory"
    )
    pack.add_argument("source")
    pack.add_argument("--out", required=True)
    add_pack_args(pack)
    pack.set_defaults(func=cmd_pack)

    kp = sub.add_parser(
        "kaggle-pack", help="Download from HF, pack on Kaggle, and upload back to HF"
    )
    kp.add_argument("--repo-id", required=True, help="HF model repo to download")
    kp.add_argument(
        "--out",
        default=None,
        help="output .orka directory (default on Kaggle: /kaggle/working/<slug>.orka)",
    )
    kp.add_argument("--upload-repo", help="HF repo to upload the result to")
    add_pack_args(kp)
    kp.add_argument("--run-eval", action="store_true",
                    help="run perplexity eval after packing")
    kp.add_argument("--eval-prompts", default=None,
                    help="prompts file for perplexity eval (defaults to AWQ calibration file)")
    kp.add_argument("--eval-max-prompts", type=int, default=16)
    kp.add_argument("--eval-max-length", type=int, default=128)
    kp.set_defaults(func=cmd_kaggle_pack)

    report = sub.add_parser("report", help="summarize an .orka artifact")
    report.add_argument("artifact")
    report.set_defaults(func=cmd_report)

    verify = sub.add_parser(
        "verify", help="decode an .orka artifact and recompute source MSE"
    )
    verify.add_argument("artifact")
    verify.set_defaults(func=cmd_verify)

    reconstruct = sub.add_parser(
        "reconstruct", help="decode an .orka artifact into a standard format"
    )
    reconstruct.add_argument("artifact")
    reconstruct.add_argument("--out", required=True)
    reconstruct.add_argument(
        "--format", choices=["json", "safetensors"], default="json"
    )
    reconstruct.add_argument(
        "--device", default=None, help="device for decoding (cpu/cuda)"
    )
    reconstruct.set_defaults(func=cmd_reconstruct)

    sweep = sub.add_parser(
        "sweep", help="run a pack/report matrix and write a JSON summary"
    )
    sweep.add_argument("source")
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--group-sizes", type=int, nargs="+", default=[8])
    sweep.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        default=None,
        help="single-stage codebook sizes to sweep",
    )
    sweep.add_argument(
        "--quant-modes",
        nargs="+",
        default=None,
        help="compositional specs (e.g. vq-8 vq-16 vq-16-8 vq-16-16-16-16)",
    )
    sweep.add_argument(
        "--codebook-modes",
        choices=["per-tensor", "global", "family"],
        nargs="+",
        default=["global"],
    )
    sweep.add_argument(
        "--normalizations",
        choices=["none", "block-max", "awq", "awq-block-max", "slrq-block"],
        nargs="+",
        default=["none"],
    )
    sweep.add_argument(
        "--rotation", choices=["none", "orthogonal", "hadamard"], default="none"
    )
    sweep.add_argument("--rotation-seed", type=int, default=None)
    sweep.add_argument(
        "--backend", choices=["auto", "numpy", "torch"], default="auto"
    )
    sweep.add_argument(
        "--device",
        default="cpu",
        help="torch backend device, for example cpu, cuda, cuda:0, or auto",
    )
    sweep.add_argument("--sample-vectors", type=int, default=None)
    sweep.add_argument("--iterations", type=int, default=12)
    sweep.add_argument("--max-values-per-tensor", type=int, default=None)
    sweep.add_argument(
        "--verify",
        action="store_true",
        help="verify every sweep artifact after packing",
    )
    sweep.add_argument(
        "--max-gpu-mem-gb",
        type=float,
        default=None,
        help="strict cap on per-process GPU memory (GB)",
    )
    sweep.add_argument(
        "--max-system-ram-gb",
        type=float,
        default=None,
        help="strict cap on total system RAM (GB). RLIMIT_AS-enforced. Hard ceiling 25GB.",
    )
    sweep.add_argument(
        "--workload-budget-gb",
        type=float,
        default=None,
        help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.",
    )
    sweep.add_argument(
        "--max-cpu-threads",
        type=int,
        default=None,
        help="cap CPU threads (torch + OMP/MKL/OPENBLAS env + sched_setaffinity).",
    )
    sweep.add_argument(
        "--progress-file", help="file to write real-time progress status"
    )
    sweep.add_argument(
        "--max-tensors", type=int, default=None, help="limit sweep to first N tensors"
    )
    sweep.add_argument(
        "--outlier-frac",
        type=float,
        default=0.0,
        help="fraction of top-magnitude weights kept as fp16 sidecar",
    )
    sweep.add_argument(
        "--awq-calibration",
        default=None,
        help="prompts file for AWQ calibration; enables activation-aware VQ",
    )
    sweep.add_argument(
        "--awq-model-dir",
        default=None,
        help="HF model dir for AWQ activation collection",
    )
    sweep.add_argument(
        "--awq-alpha",
        type=float,
        default=0.5,
        help="activation magnitude scaling power (default 0.5)",
    )
    sweep.add_argument(
        "--awq-alphas",
        type=float,
        nargs="+",
        default=None,
        help="sweep multiple AWQ alphas in one run; overrides --awq-alpha when set",
    )
    sweep.add_argument("--calibration-max-prompts", type=int, default=32)
    sweep.add_argument("--calibration-max-length", type=int, default=256)
    sweep.add_argument("--calibration-max-samples", type=int, default=4096)
    sweep.add_argument(
        "--em-aq-passes",
        type=int,
        default=3,
        help="number of EM-AQ joint refinement passes after greedy RVQ. 0 disables.",
    )
    sweep.add_argument(
        "--sensitivity-map",
        help="JSON file from sensitivity.py to enable mixed-precision",
    )
    sweep.add_argument(
        "--codebook-cache",
        default=None,
        help="dir to cache stage-0 codebooks (zero-loss reuse on identical configs)",
    )
    sweep.set_defaults(func=cmd_sweep)

    eval_cmd = sub.add_parser(
        "eval", help="evaluate an .orka artifact with Hugging Face prompt loss"
    )
    eval_cmd.add_argument("artifact")
    eval_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_cmd.add_argument("--out", required=True)
    eval_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_cmd.add_argument("--max-length", type=int, default=512)
    eval_cmd.add_argument("--device", default="cpu")
    eval_cmd.add_argument("--reconstructed-model-dir", default=None)
    eval_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_cmd.add_argument("--max-system-ram-gb", type=float, default=None,
                          help="strict cap on total system RAM (GB). RLIMIT_AS-enforced.")
    eval_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                          help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    eval_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                          help="cap CPU threads (torch + OMP/MKL + affinity).")
    eval_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                          help="strict cap on per-process GPU memory (GB).")
    eval_cmd.set_defaults(func=cmd_eval)

    pulse_check_cmd = sub.add_parser(
        "pulse-check", help="fast logit-based eval (KL Divergence & Top-1 Agreement)"
    )
    pulse_check_cmd.add_argument("artifact")
    pulse_check_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    pulse_check_cmd.add_argument("--out", required=True)
    pulse_check_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    pulse_check_cmd.add_argument("--max-prompts", type=int, default=None)
    pulse_check_cmd.add_argument("--max-length", type=int, default=512)
    pulse_check_cmd.add_argument("--device", default="cpu")
    pulse_check_cmd.add_argument("--reconstructed-model-dir", default=None)
    pulse_check_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    pulse_check_cmd.add_argument("--max-system-ram-gb", type=float, default=None,
                                 help="strict cap on total system RAM (GB). RLIMIT_AS-enforced.")
    pulse_check_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                                 help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    pulse_check_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                                 help="cap CPU threads (torch + OMP/MKL + affinity).")
    pulse_check_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                                 help="strict cap on per-process GPU memory (GB).")
    pulse_check_cmd.set_defaults(func=cmd_pulse_check)

    eval_sweep_cmd = sub.add_parser(
        "eval-sweep", help="evaluate every artifact recorded in a sweep JSON"
    )
    eval_sweep_cmd.add_argument("sweep")
    eval_sweep_cmd.add_argument(
        "--prompts", required=True, help="text file with one prompt per non-empty line"
    )
    eval_sweep_cmd.add_argument("--out", required=True)
    eval_sweep_cmd.add_argument(
        "--model-dir", default=None, help="override Hugging Face model directory"
    )
    eval_sweep_cmd.add_argument("--max-prompts", type=int, default=None)
    eval_sweep_cmd.add_argument("--max-length", type=int, default=512)
    eval_sweep_cmd.add_argument("--device", default="cpu")
    eval_sweep_cmd.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="evaluate only the first N sweep runs",
    )
    eval_sweep_cmd.add_argument(
        "--reconstructed-model-root",
        default=None,
        help="keep reconstructed model directories under this root",
    )
    eval_sweep_cmd.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to download missing files",
    )
    eval_sweep_cmd.add_argument("--max-system-ram-gb", type=float, default=None,
                                help="strict cap on total system RAM (GB). RLIMIT_AS-enforced.")
    eval_sweep_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                                help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    eval_sweep_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                                help="cap CPU threads (torch + OMP/MKL + affinity).")
    eval_sweep_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                                help="strict cap on per-process GPU memory (GB).")
    eval_sweep_cmd.set_defaults(func=cmd_eval_sweep)

    sem_analyze = sub.add_parser(
        "sem-analyze", help="Phase 1-3: Ingest vocabulary and discover linguistic roots/clusters"
    )
    sem_analyze.add_argument("model_dir", help="Hugging Face model directory")
    sem_analyze.add_argument(
        "--out", required=True, help="output JSON for the semantic analysis"
    )
    sem_analyze.add_argument(
        "--device", default="cpu", help="device for embedding analysis (cpu/cuda)"
    )
    sem_analyze.set_defaults(func=cmd_sem_analyze)

    def _run_tests(_args):
        import unittest

        loader = unittest.defaultTestLoader
        suite = loader.discover("tests", top_level_dir=".")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


    selftest = sub.add_parser("selftest", help="run built-in tests")
    selftest.set_defaults(func=_run_tests)
    return parser
