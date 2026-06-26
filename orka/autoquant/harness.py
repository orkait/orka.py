"""pi-reason generate->verify->refine LLM loop for hard-call tensors. Transport is injected
as llm_fn(messages)->str (production = litellm; tests = canned). The model receives the
role, shape, signals and objective and returns a JSON quant config, schema-validated here."""
from __future__ import annotations
import json
from orka.autoquant.probes import Signals
from orka.autoquant.schema import TensorConfig

_VALID_METHODS = {"rvq", "int8", "fp16"}


def _prompt(role, shape, s: Signals, objective):
    return [
        {"role": "system", "content":
         "You are a quantization expert. Given one tensor's role, shape, and distortion "
         "signals, output ONLY a JSON object: {method:rvq|int8|fp16, bits:int, stages:int, "
         "normalization:str, keep_fp16:bool, rationale:str}. The output head must never be "
         "RVQ (use int8). Norms/biases stay fp16."},
        {"role": "user", "content": json.dumps({
            "role": role, "shape": list(shape), "objective": objective,
            "sqnr_curve": s.sqnr_curve, "rd_knee_bits": s.rd_knee_bits,
            "sensitivity": s.sensitivity})},
    ]


def _parse(text: str) -> dict:
    try:
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"LLM did not return JSON: {text[:80]!r}") from e
    if d.get("method") not in _VALID_METHODS:
        raise ValueError(f"invalid method {d.get('method')!r}")
    return d


def decide_with_llm(role, shape, signals: Signals, objective: str, *, llm_fn) -> TensorConfig:
    d = _parse(llm_fn(_prompt(role, shape, signals, objective)))
    return TensorConfig(
        method=d["method"], bits=int(d.get("bits", 8)), stages=int(d.get("stages", 0)),
        normalization=d.get("normalization", "block-max"),
        keep_fp16=bool(d.get("keep_fp16", False)), source="llm",
        confidence=0.75, rationale=d.get("rationale", "llm"))
