"""Pydantic models = the journey contract. Single source of truth for what every UI
layer renders. Versioned via schema_version."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Source = Literal["estimated", "measured"]
Treatment = Literal["quantize", "keep_fp16", "skip_error_comp"]
ArchClass = Literal["dense", "moe", "mamba_hybrid", "conv_hybrid", "hybrid"]


class ModelMeta(BaseModel):
    name: str
    params_total: int
    dtype: str
    vocab_size: int | None = None
    tie_word_embeddings: bool = False
    fp16_bytes: int


class FamilyBreakdown(BaseModel):
    family: str
    params: int
    pct: float
    role: str = ""


class ModuleEntry(BaseModel):
    name: str
    shape: list[int]
    family: str
    treatment: Treatment


class LayerBlock(BaseModel):
    index: int
    modules: list[ModuleEntry]


class Architecture(BaseModel):
    arch_class: ArchClass
    flags: dict[str, bool]
    param_breakdown: list[FamilyBreakdown]
    layers: list[LayerBlock]
    partial: bool = False


class Stage(BaseModel):
    id: str
    title: str
    summary: str


class Trick(BaseModel):
    id: str
    label: str
    kind: Literal["scalar", "toggle"]
    default: float | bool
    applies: bool
    why: str = ""
    warn: str | None = None
    gated_by: str | None = None


class Result(BaseModel):
    source: Source
    bpw: float
    ratio: float
    fp16_mb: float
    orka_mb: float
    ppl_base: float | None = None
    ppl_orka: float | None = None
    ppl_ratio: float | None = None
    trusted: bool | None = None
    trust_reason: str | None = None
    notes: list[str] = []


class Journey(BaseModel):
    schema_version: int
    model: ModelMeta
    architecture: Architecture
    pipeline: list[Stage]
    tricks: list[Trick]
    result: Result


class RDPoint(BaseModel):
    bpw: float
    sqnr: float


class TensorProbe(BaseModel):
    """Deep-probe of one tensor, computed on a range-fetched weight sample (CPU, no GPU)."""
    name: str
    shape: list[int]
    dtype: str
    sampled_elems: int
    std: float
    mean: float
    vmin: float
    vmax: float
    outlier_pct: float
    distribution: list[float]
    dist_range: list[float]
    codebook_values: list[float]
    rd: list[RDPoint]
    utilization: list[float]
    entropy_bits: float
    entropy_max: float
    weights_block: list[list[float]]
    error_block: list[list[float]]
    sqnr_3bpw: float
    error_pct: float
    vectors3d: list[list[float]]
    centroids3d: list[list[float]]
