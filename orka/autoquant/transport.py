"""Production LLM transport for autoquant's hard-call escalation. Lazily uses whichever
backend is installed (litellm preferred, then anthropic); raises NoLLMBackend if none, so
cmd_autoquant falls back to the deterministic policy. A role router picks a lite model for
ordinary escalations and a strong model for stubborn ones."""
from __future__ import annotations
import os
import importlib.util

LITE_MODEL = os.environ.get("ORKA_LLM_LITE", "claude-sonnet-4-6")
STRONG_MODEL = os.environ.get("ORKA_LLM_STRONG", "claude-opus-4-8")


class NoLLMBackend(Exception):
    pass


def route_model(hard: bool) -> str:
    """Strong model for hard calls (stubborn tensors), lite otherwise."""
    return STRONG_MODEL if hard else LITE_MODEL


def _litellm_completion(model, messages) -> str:
    import litellm
    r = litellm.completion(model=model, messages=messages, temperature=0)
    return r["choices"][0]["message"]["content"]


def _anthropic_completion(model, messages) -> str:
    import anthropic
    client = anthropic.Anthropic()
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(model=model, max_tokens=512, system=sys,
                               messages=user, temperature=0)
    return r.content[0].text


def _available_default():
    """[(name, completion)] for installed SDKs, preference order."""
    out = []
    if importlib.util.find_spec("litellm"):
        out.append(("litellm", _litellm_completion))
    if importlib.util.find_spec("anthropic"):
        out.append(("anthropic", _anthropic_completion))
    return out


def make_llm_fn(hard: bool = False, *, _backends=None):
    """Return an llm_fn(messages)->str backed by the first available SDK.
    Raises NoLLMBackend if none is installed. Tests pass _backends=[completion_callable]."""
    model = route_model(hard)
    backends = _backends if _backends is not None else [c for _, c in _available_default()]
    if not backends:
        raise NoLLMBackend("no LLM backend installed (pip install litellm or anthropic)")

    def llm_fn(messages):
        return backends[0](model, messages)

    return llm_fn
