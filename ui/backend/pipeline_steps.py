"""The static pipeline-stage list (the journey stepper) and the trick catalogue (the Trick
Lab). Trick applicability/warnings are arch-gated from the Architecture flags - never
hardcoded per model."""
from __future__ import annotations

from .schema import Architecture, Stage, Trick

_STAGES = [
    ("load", "Load checkpoint", "Read source weights (safetensors)."),
    ("transform", "Normalize / rotate", "Per-tensor scale + optional Hadamard rotation."),
    ("allocate", "Bit allocation", "Uniform bpw across tensors (uniform beats per-tensor <1.5B)."),
    ("codebook", "Learn RVQ codebooks", "k-means codebooks per stage (rvq-12-12 = 2x K=4096)."),
    ("quantize", "Assign indices + residual", "Nearest-codeword assignment, residual to next stage."),
    ("strategies", "Post-assignment", "error-comp / EM-AQ / mse-scale (arch-gated)."),
    ("pack", "Write artifact", "Index planes + codebooks + manifest."),
]


def build_pipeline(arch: Architecture) -> list[Stage]:
    return [Stage(id=i, title=t, summary=s) for i, t, s in _STAGES]


def build_tricks(arch: Architecture) -> list[Trick]:
    f = arch.flags
    tricks = [
        Trick(id="bpw", label="Bits per weight", kind="scalar", default=3.0, applies=True,
              why="uniform bpw is the <1.5B sweet spot"),
        Trick(id="rvq_stages", label="RVQ stages", kind="scalar", default=2, applies=True,
              why="residual stages stack codebooks (12-12 = 3bpw)"),
        Trick(id="em_aq", label="EM-AQ refine", kind="toggle", default=True, applies=True,
              why="joint codebook refinement, free quality"),
        Trick(id="hessian", label="Hessian weighting", kind="toggle", default=True, applies=True,
              why="biggest free quality lever (1.80->1.35 at 3bpw)"),
        Trick(id="mse_scale", label="MSE-optimal scales", kind="toggle", default=False, applies=True,
              why="least-squares block scales, free quality"),
        Trick(id="keep_head_fp16", label="Keep head fp16", kind="toggle",
              default=bool(f.get("tied_head")), applies=True, gated_by="tied_head",
              why="tied head IS the logit projection; quantizing it explodes ppl"),
        Trick(id="error_comp", label="Error compensation (LDLQ)", kind="toggle", default=False,
              applies=True,
              why="block-OBS; auto-skipped on output head + recurrent/SSM tensors"),
        Trick(id="lattice", label="E8 lattice", kind="toggle", default=False, applies=True,
              warn="Pareto-loses to VQ on hybrid archs" if f.get("has_ssm") else None,
              why="codebook-free QuIP#; wins only on standard transformers at high bpw"),
        Trick(id="outliers", label="Outlier extraction", kind="toggle", default=False, applies=True,
              why="keep top-magnitude weights fp16"),
        Trick(id="rotation", label="Transform search", kind="toggle", default=False, applies=True,
              why="per-tensor normalize/rotate pick"),
    ]
    return tricks
