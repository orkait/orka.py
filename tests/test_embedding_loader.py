"""Installing a row-lazy embedding must be gated on ACCESS PATTERN, not on a config flag.

``VQEmbedding`` shipped without a caller, so nothing decided when it was safe to use. The
rule it needs: an embedding is a lookup weight only while nothing multiplies by it. A tied
checkpoint whose head is still attached uses the same matrix as a logit projection - serving
that row-lazily would quantize a compute weight. The same checkpoint with the head discarded
(the encoder deployment) is pure lookup and safe.

These tests pin that distinction, plus the name resolution that makes the encoder case
reachable at all: the artifact names tensors from the checkpoint root, while the eligible
deployment is usually a submodule.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from orka.inference.embedding_loader import (  # noqa: E402
    install_vq_embeddings,
    is_pure_lookup,
)
from orka.inference.vq_embedding import VQEmbedding  # noqa: E402
from orka.pipeline.decode import _decode_tensor  # noqa: E402
from orka.pipeline.pack import pack_checkpoint  # noqa: E402

VOCAB, DIM = 384, 64


class Encoder(nn.Module):
    """Lookup-only: an embedding and nothing that projects back to vocab."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIM)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, ids):
        return self.embed_tokens(ids)


class TiedLM(nn.Module):
    """The same table, additionally used as the output projection."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)
        self.lm_head.weight = self.encoder.embed_tokens.weight

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class UntiedLM(TiedLM):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)   # own weights, breaks the tie


def _packed(tmp: Path, name: str = "model.embed_tokens.weight", **overrides):
    torch.manual_seed(0)
    src = tmp / "m.pt"
    torch.save({name: torch.randn(VOCAB, DIM)}, src)
    art = tmp / "a.orka"
    kwargs = dict(group_size=8, codebook_sizes=[256, 256], codebook_mode="per-tensor",
                  backend="numpy", device="cpu", normalization="block-max",
                  sample_vectors=256, iterations=4, em_aq_passes=0)
    kwargs.update(overrides)
    manifest = pack_checkpoint(source=src, out_dir=art, **kwargs)
    return art, manifest


def test_tied_head_makes_the_table_a_compute_weight_and_is_refused():
    ok, why = is_pure_lookup(TiedLM())
    assert ok is False
    assert "tied" in why


def test_untied_head_leaves_the_table_a_pure_lookup():
    ok, why = is_pure_lookup(UntiedLM())
    assert (ok, why) == (True, "")


def test_encoder_without_a_head_is_eligible():
    ok, why = is_pure_lookup(Encoder())
    assert (ok, why) == (True, "")


def test_module_without_input_embeddings_is_refused():
    ok, why = is_pure_lookup(nn.Linear(4, 4))
    assert ok is False
    assert "get_input_embeddings" in why


def test_install_refuses_the_tied_model_and_leaves_it_dense():
    with tempfile.TemporaryDirectory() as tmp:
        art, _ = _packed(Path(tmp), name="encoder.embed_tokens.weight")
        model = TiedLM()
        rep = install_vq_embeddings(model, art)
        assert rep["installed"] is False
        assert "tied" in rep["reason"]
        assert isinstance(model.encoder.embed_tokens, nn.Embedding)


def test_allow_tied_overrides_the_gate_for_callers_that_know_better():
    with tempfile.TemporaryDirectory() as tmp:
        art, _ = _packed(Path(tmp), name="encoder.embed_tokens.weight")
        model = TiedLM()
        rep = install_vq_embeddings(model, art, allow_tied=True)
        assert rep["installed"] is True
        assert "tied" in rep["forced_tied"]


def test_submodule_root_resolves_by_unique_suffix():
    """The artifact says 'model.embed_tokens.weight'; the encoder alone says
    'embed_tokens.weight'. That mismatch is the normal case, not an edge case."""
    with tempfile.TemporaryDirectory() as tmp:
        art, _ = _packed(Path(tmp), name="model.embed_tokens.weight")
        enc = Encoder()
        rep = install_vq_embeddings(enc, art)
        assert rep["installed"] is True, rep["reason"]
        assert rep["tensor"] == "model.embed_tokens.weight"
        assert isinstance(enc.embed_tokens, VQEmbedding)


def test_ambiguous_suffix_is_refused_rather_than_guessed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        art, manifest = _packed(tmp, name="a.embed_tokens.weight")
        # forge a second tensor with the same suffix
        mpath = art / "manifest.json"
        m = json.loads(mpath.read_text())
        twin = dict(m["tensors"][0])
        twin["name"] = "b.embed_tokens.weight"
        m["tensors"].append(twin)
        mpath.write_text(json.dumps(m))
        rep = install_vq_embeddings(Encoder(), art)
        assert rep["installed"] is False
        assert "ambiguous" in rep["reason"]


def test_missing_tensor_is_reported_not_raised():
    with tempfile.TemporaryDirectory() as tmp:
        art, _ = _packed(Path(tmp), name="something.else.weight")
        rep = install_vq_embeddings(Encoder(), art)
        assert rep["installed"] is False
        assert "not quantized" in rep["reason"]


def test_installed_module_is_bit_exact_and_reports_a_real_saving():
    with tempfile.TemporaryDirectory() as tmp:
        art, manifest = _packed(Path(tmp), name="model.embed_tokens.weight")
        meta = manifest["tensors"][0]
        ref = np.asarray(_decode_tensor(art, meta), dtype=np.float32).reshape(VOCAB, DIM)

        enc = Encoder()
        rep = install_vq_embeddings(enc, art, out_dtype=torch.float32)
        assert rep["installed"] is True

        ids = torch.tensor([0, 1, 17, VOCAB - 1])
        got = enc(ids).detach().numpy()
        assert np.array_equal(got, ref[ids.numpy()])

        # the saving must be arithmetic on real buffers, not an assumption
        assert rep["dense_bytes"] == VOCAB * DIM * 4
        assert rep["resident_bytes"] == rep["breakdown"]["total"]
        assert rep["saved_bytes"] == rep["dense_bytes"] - rep["resident_bytes"]
        assert rep["resident_bytes"] < rep["dense_bytes"]


def test_forward_shape_matches_nn_embedding_for_batched_ids():
    with tempfile.TemporaryDirectory() as tmp:
        art, _ = _packed(Path(tmp), name="model.embed_tokens.weight")
        enc = Encoder()
        install_vq_embeddings(enc, art, out_dtype=torch.float32)
        ids = torch.randint(0, VOCAB, (3, 5))
        assert enc(ids).shape == (3, 5, DIM)
