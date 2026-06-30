"""Architecture-aware per-tensor identification: ONE source of truth for the questions the
quantizers ask about every tensor - "is this the output head / a vocab-width embedding / a
recurrent (SSM) block?" - independent of how a given architecture NAMES its layers.

Why this module exists (the named force): the same identification was duplicated across
four call sites - the lattice packer, the VQ packer's error-compensation gate, pillar
protection, and the name helpers - each assembling the structural signals slightly
differently, and they drifted (a ``self_attn``/``mlp`` allow-list covered 9% of a hybrid; a
``mamba``-only skip missed a pure-Mamba ``mixer``). Centralizing IDENTIFICATION here - while
each pipeline keeps its own quantize / keep-dense / skip-OBS POLICY - makes the call sites
ask the same question the same way and turns "support a new architecture" into a one-rule
change instead of a four-site edit.

Detection is structural-primary, name-fallback:
  * output head / embedding -> a 2-D weight whose out dim == vocab_size (or, on a live
    model, the module returned by ``get_output_embeddings()`` by identity); not a name.
  * recurrent / SSM block   -> a tensor whose owning module also holds a recurrence STATE
    param (``A_log`` / ``dt_bias`` / ...); keys on the recurrence math, not the block name.
The ``is_output_head`` / ``is_recurrent_block`` name predicates remain ONLY as the fallback
used when neither shapes nor a live model are available.

Out of scope (deliberately separate concerns): ``family.classify_tensor_family`` (codebook
grouping in family mode) and ``autoquant.roles.classify_role`` (bit-allocation policy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

_OUTPUT_HEAD_MARKERS = ("lm_head", "embed_out", "output.weight")
# Documented state-space / recurrent block names (name fallback only). Mamba/Mamba2 register
# the block as ``mixer`` in the reference impl; FalconH1 names it ``mamba``.
_RECURRENT_MARKERS = ("mamba", "mixer", "ssm", "state_space", "state-space")
# Recurrence STATE parameters: their presence as a sibling of a Linear marks that Linear as
# living inside a state-space / linear-recurrent block, INDEPENDENT of the block's name -
# the math-defining params (log state-transition matrix, selective timestep, token-shift
# decay). Far more robust than matching the block's name. Lowercased.
_RECURRENCE_STATE_PARAMS = frozenset(
    {"a_log", "dt_bias", "dt_proj", "time_decay", "time_mix", "time_maa", "time_faaaa"}
)


def _base(name: str) -> str:
    """Drop a trailing ``.weight`` so a state_dict tensor name ('lm_head.weight') and a live
    module path ('lm_head') compare equal."""
    return name[:-7] if name.endswith(".weight") else name


def is_output_head(name: str) -> bool:
    """NAME-BASED FALLBACK for the output head - prefer ``ArchProfile`` (vocab-width) when
    shapes or a live model are available. True for the final logit projection (lm_head /
    embed_out / output.weight), which must stay high-precision (quantizing the logits
    explodes perplexity, and block-OBS over a softmax is invalid)."""
    return any(m in name.lower() for m in _OUTPUT_HEAD_MARKERS)


def is_recurrent_block(name: str) -> bool:
    """NAME-BASED FALLBACK for recurrent/SSM tensors - prefer ``ArchProfile`` (sibling state
    params) when the full name set is available. Covers the documented SSM block names
    (mamba/mixer/ssm) but, being name-based, cannot know an arbitrary new one. Used to skip
    block-OBS error compensation; plain weight quant of these matrices is still fine."""
    return any(m in name.lower() for m in _RECURRENT_MARKERS)


def recurrent_block_names(tensor_names: Iterable[str]) -> set[str]:
    """STRUCTURAL recurrent/SSM detection: tensor names that live inside a module which also
    owns a recurrence state param (``A_log``/``dt_bias``/...). Keys on the recurrence math,
    not the block name, so it holds for Mamba (``...mamba.in_proj``), pure-Mamba
    (``...mixer.in_proj``), or any future naming. Takes the FULL tensor-name set (including
    1-D params like ``A_log`` that are not quantization candidates)."""
    names = list(tensor_names)
    blocks = {n.rsplit(".", 1)[0] for n in names
              if n.rsplit(".", 1)[-1].lower() in _RECURRENCE_STATE_PARAMS}
    if not blocks:
        return set()
    prefixes = tuple(b + "." for b in blocks)
    return {n for n in names if n.startswith(prefixes)}


def output_head_names(tensor_shapes: Mapping, vocab_size: int | None = None) -> set[str]:
    """STRUCTURAL output-head / vocab-projection detection: 2-D weight names whose output
    dimension equals ``vocab_size`` (catches both ``lm_head`` and the input embedding,
    regardless of name). When ``vocab_size`` is unknown, uses the dominant large output dim
    (vocab >> hidden in every standard LM). These stay high-precision in pipelines that
    protect them."""
    twod = {n: tuple(s) for n, s in tensor_shapes.items() if len(s) == 2}
    if not twod:
        return set()
    vocab = vocab_size if vocab_size else max(s[0] for s in twod.values())
    return {n for n, s in twod.items() if s[0] == vocab}


@dataclass(frozen=True)
class ArchProfile:
    """Immutable per-checkpoint identification, built once and queried by every pipeline.

    Holds the resolved vocab-width (head/embedding) and recurrent-block names (as ``.weight``-
    stripped module paths, so state_dict names and live module paths compare equal) so the
    structural scan happens once. The predicate methods are structural-primary with a name
    fallback. Build with ``from_shapes`` (state_dict names+shapes) or ``from_model`` (a live
    HF model, where the head is found by ``get_output_embeddings()`` identity)."""

    vocab_size: int | None
    head_names: frozenset[str]
    recurrent_names: frozenset[str]

    @classmethod
    def from_shapes(cls, tensor_shapes: Mapping, vocab_size: int | None = None) -> "ArchProfile":
        twod = [tuple(s) for s in tensor_shapes.values() if len(s) == 2]
        vocab = vocab_size or (max((s[0] for s in twod), default=0) or None)
        return cls(
            vocab_size=vocab,
            head_names=frozenset(_base(n) for n in output_head_names(tensor_shapes, vocab)),
            recurrent_names=frozenset(_base(n) for n in recurrent_block_names(tensor_shapes.keys())),
        )

    @classmethod
    def from_model(cls, model) -> "ArchProfile":
        """Live-model profile: the output head is found by ``get_output_embeddings()``
        IDENTITY (the canonical signal) plus any Linear of vocab width; recurrent blocks by
        the sibling state params over ``named_parameters()``."""
        import torch.nn as nn

        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        head_ids = set()
        try:
            oe = model.get_output_embeddings()
            if oe is not None:
                head_ids.add(id(oe))
        except Exception:
            pass
        heads = set()
        for name, mod in model.named_modules():
            if id(mod) in head_ids or (
                isinstance(mod, nn.Linear) and vocab and mod.out_features == vocab
            ):
                heads.add(name)
        return cls(
            vocab_size=vocab,
            head_names=frozenset(heads),
            recurrent_names=frozenset(_base(n) for n in recurrent_block_names(
                [n for n, _ in model.named_parameters()])),
        )

    def is_output_head(self, name: str, shape=None) -> bool:
        """True for the output head / vocab-width embedding. Structural first (resolved set
        or vocab-width shape), name fallback last."""
        if _base(name) in self.head_names:
            return True
        if shape is not None and len(shape) == 2 and self.vocab_size and shape[0] == self.vocab_size:
            return True
        return is_output_head(name)

    def is_recurrent(self, name: str) -> bool:
        """True for a recurrent/SSM block tensor (structural set, name fallback last)."""
        return _base(name) in self.recurrent_names or is_recurrent_block(name)

    def error_comp_skip_reason(self, name: str) -> str | None:
        """Why block-OBS error compensation must be skipped for ``name`` (a human reason), or
        None if it may run. Ordered rules - extend HERE (one place) for a new role that
        breaks block-OBS's linear-output-error proxy. Verified roles: the output head
        (softmax downstream -> skews logits, WORSE ppl) and recurrent/SSM blocks (nonlinear
        scan downstream; FalconH1 4bpw 1.10 -> 1.50 with error-comp on these)."""
        if self.is_output_head(name):
            return "output head (softmax downstream)"
        if self.is_recurrent(name):
            return "recurrent/SSM block (nonlinear scan downstream)"
        return None
