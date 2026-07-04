"""Assemble the static (estimated) Journey from a model name: fetch config+shapes ->
ModelMeta + Architecture -> estimate -> pipeline + tricks."""
from __future__ import annotations

from .arch import build_architecture
from .estimator import estimate
from .fetch import fetch_config, fetch_shapes
from .pipeline_steps import build_pipeline, build_tricks
from .schema import Journey, ModelMeta
from .settings import HF_TOKEN, SCHEMA_VERSION

_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def build_static_journey(model: str, bpw: float = 3.0, keep_head: bool = True,
                         lattice: bool = False, token: str | None = None) -> Journey:
    token = token or HF_TOKEN
    config = fetch_config(model, token=token)
    shapes = fetch_shapes(model, token=token)

    params_total = sum(_numel(s) for s in shapes.values())
    dtype = config.get("torch_dtype", "bfloat16")
    nbytes = _DTYPE_BYTES.get(dtype, 2)
    meta = ModelMeta(
        name=model, params_total=params_total, dtype=dtype,
        vocab_size=config.get("vocab_size"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        fp16_bytes=params_total * nbytes,
    )
    arch = build_architecture(config, shapes)
    result = estimate(meta, arch, bpw=bpw, keep_head=keep_head, lattice=lattice)
    return Journey(
        schema_version=SCHEMA_VERSION, model=meta, architecture=arch,
        pipeline=build_pipeline(arch), tricks=build_tricks(arch), result=result,
    )
