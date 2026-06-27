"""argparse parser construction. Each subcommand registered with its cmd_*.

SLRQ subcommand removed (parallel path replaced by --normalization slrq-block in pack).
"""

from __future__ import annotations

import argparse

from orka.cli._pack_args import _add_pack_args

from orka.quant.allocate import DEFAULT_CANDIDATE_SPECS, cmd_allocate
from orka.artifact.export import cmd_export_vllm
from orka.cli.commands import (
    cmd_autoquant,
    cmd_calc,
    cmd_correct,
    cmd_distill,
    cmd_eval,
    cmd_eval_sweep,
    cmd_inspect,
    cmd_kaggle_pack,
    cmd_pack,
    cmd_merge_orka,
    cmd_pulse_check,
    cmd_reconstruct,
    cmd_report,
    cmd_sem_analyze,
    cmd_sem_map,
    cmd_sem_calc,
    cmd_sweep,
    cmd_verify,
)



def _add_calc_parser(sub):
    calc = sub.add_parser("calc", help="estimate Orka payload size")
    calc.add_argument(
        "--params", required=True, help="parameter count, for example 8.03b"
    )
    calc.add_argument("--group-size", type=int, default=8)
    calc.add_argument("--codebook-size", type=int, default=256)
    calc.add_argument("--scale-block-vectors", type=int, default=64)
    calc.add_argument("--scale-bits", type=int, default=16)
    calc.set_defaults(func=cmd_calc)

    aq = sub.add_parser("autoquant", help="auto-derive a per-tensor quant config for any model")
    aq.add_argument("model", help="HF model dir (safetensors)")
    aq.add_argument("--objective", choices=["min-bits", "max-quality", "knee"], default="knee")
    aq.add_argument("--out", default="allocation_map.json")
    aq.add_argument("--target", default=None, help="KL/bpw/MB target for min-bits/max-quality")
    aq.add_argument("--prompts", default=None, help="pulse-check prompts file")
    aq.add_argument("--no-llm", action="store_true", help="pure deterministic policy (no LLM)")
    aq.set_defaults(func=cmd_autoquant)


def _add_inspect_parser(sub):
    inspect = sub.add_parser(
        "inspect", help="inspect a safetensors or PyTorch checkpoint"
    )
    inspect.add_argument("source")
    inspect.set_defaults(func=cmd_inspect)


def _add_pack_parser(sub):
    pack = sub.add_parser(
        "pack", help="pack candidate weight tensors into an .orka directory"
    )
    pack.add_argument("source")
    pack.add_argument("--out", required=True)
    _add_pack_args(pack)
    pack.add_argument(
        "--sequential-calibration",
        action="store_true",
        help="pack blocks in forward order, recapturing calibration activations "
             "from the partially quantized model (GPTQ-style error propagation). "
             "Requires --awq-model-dir and --awq-calibration; per-tensor mode only.",
    )
    pack.set_defaults(func=cmd_pack)


def _add_kaggle_pack_parser(sub):
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
    _add_pack_args(kp)
    kp.add_argument("--run-eval", action="store_true",
                    help="run perplexity eval after packing")
    kp.add_argument("--eval-prompts", default=None,
                    help="prompts file for perplexity eval (defaults to AWQ calibration file)")
    kp.add_argument("--eval-max-prompts", type=int, default=16)
    kp.add_argument("--eval-max-length", type=int, default=128)
    kp.set_defaults(func=cmd_kaggle_pack)


def _add_report_parser(sub):
    report = sub.add_parser("report", help="summarize an .orka artifact")
    report.add_argument("artifact")
    report.set_defaults(func=cmd_report)


def _add_allocate_parser(sub):
    allocate = sub.add_parser(
        "allocate",
        help="measure per-tensor rate-distortion and solve a bit allocation "
             "(discrete water-filling) for a target bits-per-weight budget",
    )
    allocate.add_argument("source")
    allocate.add_argument("--out", required=True)
    allocate.add_argument("--target-bpw", type=float, required=True)
    allocate.add_argument(
        "--candidates",
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SPECS),
        help="candidate quant specs to probe per tensor",
    )
    allocate.add_argument("--group-size", type=int, default=8)
    allocate.add_argument("--sample-vectors", type=int, default=4096)
    allocate.add_argument("--iterations", type=int, default=4)
    allocate.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    allocate.add_argument("--device", default="cpu")
    allocate.add_argument("--max-tensors", type=int, default=None)
    allocate.add_argument("--progress-file", default=None)
    allocate.set_defaults(func=cmd_allocate)


def _add_export_vllm_parser(sub):
    export_cmd = sub.add_parser(
        "export-vllm",
        help="export to a Hugging Face model directory loadable by vLLM / "
             "transformers; low-rank corrections become a PEFT LoRA adapter",
    )
    export_cmd.add_argument("artifact")
    export_cmd.add_argument("--out", required=True)
    export_cmd.add_argument(
        "--model-dir", default=None,
        help="HF dir for config/tokenizer sidecars (default: source's directory)",
    )
    export_cmd.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    export_cmd.add_argument(
        "--merge-correction",
        action="store_true",
        help="merge low-rank corrections into the dense weights instead of "
             "emitting a PEFT adapter",
    )
    export_cmd.add_argument("--device", default="cpu")
    export_cmd.set_defaults(func=cmd_export_vllm)


def _add_correct_parser(sub):
    correct = sub.add_parser(
        "correct",
        help="add low-rank correction sidecars: W ~ decode(W) + A@B^T (fp16, "
             "rank r) fitted to the post-pack residual",
    )
    correct.add_argument("artifact")
    correct.add_argument("--rank", type=int, default=8)
    correct.add_argument("--device", default="cpu")
    correct.add_argument("--max-tensors", type=int, default=None)
    correct.set_defaults(func=cmd_correct)


def _add_distill_parser(sub):
    distill = sub.add_parser(
        "distill",
        help="post-pack codebook distillation: indices frozen, codebooks optimized "
             "against the source weights (optionally activation-weighted)",
    )
    distill.add_argument("artifact")
    distill.add_argument("--steps", type=int, default=200)
    distill.add_argument("--lr", type=float, default=1e-3)
    distill.add_argument("--device", default="cpu")
    distill.add_argument("--max-tensors", type=int, default=None)
    distill.add_argument(
        "--activations-file",
        default=None,
        help="JSON/pt activations for column-importance weighting (E[x^2])",
    )
    distill.add_argument(
        "--model-dir",
        default=None,
        help="HF model dir to collect fresh calibration activations",
    )
    distill.add_argument(
        "--prompts", default=None, help="prompts file for fresh calibration"
    )
    distill.add_argument("--calibration-max-prompts", type=int, default=32)
    distill.add_argument("--calibration-max-length", type=int, default=256)
    distill.add_argument("--calibration-max-samples", type=int, default=4096)
    distill.set_defaults(func=cmd_distill)


def _add_merge_orka_parser(sub):
    merge = sub.add_parser(
        "merge-orka",
        help="merge partitioned .orka artifacts into one complete artifact",
    )
    merge.add_argument(
        "artifacts",
        nargs="+",
        help="one or more partitioned .orka paths",
    )
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge_orka)


def _add_verify_parser(sub):
    verify = sub.add_parser(
        "verify", help="decode an .orka artifact and recompute source MSE"
    )
    verify.add_argument("artifact")
    verify.set_defaults(func=cmd_verify)


def _add_reconstruct_parser(sub):
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


def _add_sweep_parser(sub):
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
        choices=["none", "block-max", "channel-block-max", "awq", "awq-block-max", "slrq-block"],
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
        help="strict cap on total system RAM (GB), enforced by the 100ms RSS poll-monitor. Hard ceiling 25GB.",
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


def _add_eval_parser(sub):
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
                          help="strict cap on total system RAM (GB), enforced by the 100ms RSS poll-monitor.")
    eval_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                          help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    eval_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                          help="cap CPU threads (torch + OMP/MKL + affinity).")
    eval_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                          help="strict cap on per-process GPU memory (GB).")
    eval_cmd.set_defaults(func=cmd_eval)


def _add_pulse_check_parser(sub):
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
                                 help="strict cap on total system RAM (GB), enforced by the 100ms RSS poll-monitor.")
    pulse_check_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                                 help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    pulse_check_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                                 help="cap CPU threads (torch + OMP/MKL + affinity).")
    pulse_check_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                                 help="strict cap on per-process GPU memory (GB).")
    pulse_check_cmd.set_defaults(func=cmd_pulse_check)


def _add_eval_sweep_parser(sub):
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
                                help="strict cap on total system RAM (GB), enforced by the 100ms RSS poll-monitor.")
    eval_sweep_cmd.add_argument("--workload-budget-gb", type=float, default=None,
                                help="estimated process RAM budget (GB) for preflight. Required with --max-system-ram-gb.")
    eval_sweep_cmd.add_argument("--max-cpu-threads", type=int, default=None,
                                help="cap CPU threads (torch + OMP/MKL + affinity).")
    eval_sweep_cmd.add_argument("--max-gpu-mem-gb", type=float, default=None,
                                help="strict cap on per-process GPU memory (GB).")
    eval_sweep_cmd.set_defaults(func=cmd_eval_sweep)


def _add_sem_analyze_parser(sub):
    sem_analyze = sub.add_parser(
        "sem-analyze", help="Phase 1-3: Ingest vocabulary and discover linguistic roots/clusters"
    )
    sem_analyze.add_argument("model_dir", help="Hugging Face model directory")
    sem_analyze.add_argument(
        "--out", required=True, help="output JSON for the semantic analysis"
    )
    sem_analyze.add_argument(
        "--save-sensitivity-map", help="generate a .json file for orka pack --sensitivity-map"
    )
    sem_analyze.add_argument(
        "--device", default="cpu", help="device for embedding analysis (cpu/cuda)"
    )
    sem_analyze.set_defaults(func=cmd_sem_analyze)


def _add_sem_map_parser(sub):
    sem_map = sub.add_parser(
        "sem-map", help="Phase 4: Link character roots to geometric concept hubs"
    )
    sem_map.add_argument("analysis_json", help="output from orka sem-analyze")
    sem_map.add_argument(
        "--out", required=True, help="output JSON for the concept mapping table"
    )
    sem_map.set_defaults(func=cmd_sem_map)


def _add_sem_calc_parser(sub):
    sem_calc = sub.add_parser(
        "sem-calc", help="Pre-calculate AWQ activations and linguistic pillars"
    )
    sem_calc.add_argument("source", help="source checkpoint (.safetensors / .pt / .bin)")
    _add_pack_args(sem_calc)
    sem_calc.add_argument("--out", required=True, help="output JSON for the calculated data")
    sem_calc.set_defaults(func=cmd_sem_calc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orka model compiler prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_calc_parser(sub)
    _add_inspect_parser(sub)
    _add_pack_parser(sub)
    _add_kaggle_pack_parser(sub)
    _add_report_parser(sub)
    _add_allocate_parser(sub)
    _add_export_vllm_parser(sub)
    _add_correct_parser(sub)
    _add_distill_parser(sub)
    _add_merge_orka_parser(sub)
    _add_verify_parser(sub)
    _add_reconstruct_parser(sub)
    _add_sweep_parser(sub)
    _add_eval_parser(sub)
    _add_pulse_check_parser(sub)
    _add_eval_sweep_parser(sub)
    _add_sem_analyze_parser(sub)
    _add_sem_map_parser(sub)
    _add_sem_calc_parser(sub)
    return parser
