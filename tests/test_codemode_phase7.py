"""Regression gates for the phase-7 codemode fixes.

Each test pins one behaviour that was either wrong or unpinned:
  * the k-means++ seeding gate is a config knob, not a buried literal
  * the fused torch assign no longer advertises a row-norm cache it never read
  * mse_scale waits for the background writer before reading back index streams
  * the giant (host-resident) pack path matches the normal path on CUDA
  * the batched semantic-hub scan matches the per-index scan it replaced
  * a planar spec is rate-checked in bits per WEIGHT, not bits per vector
  * the prefill crossover is env-tunable
  * no domain subpackage imports an entry-point package
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------- k-means++ gate
def test_kmeans_pp_gate_is_configurable_numpy(monkeypatch):
    from orka.codebook._kmeans_numpy import _kmeans_parallel_init_numpy

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4000, 8)).astype(np.float32)

    def sampled(k, seed=5):
        r = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        return rows[r.choice(rows.shape[0], size=k, replace=False)]

    monkeypatch.setenv("ORKA_KMEANS_PP_MAX_K", "64")
    above = _kmeans_parallel_init_numpy(rows, 128, seed=5)
    assert np.array_equal(above, sampled(128)), "above the gate must be a plain sample"

    monkeypatch.setenv("ORKA_KMEANS_PP_MAX_K", "256")
    below = _kmeans_parallel_init_numpy(rows, 128, seed=5)
    assert below.shape == (128, 8)
    assert not np.array_equal(below, sampled(128)), "below the gate k-means|| must run"


def test_kmeans_pp_gate_is_configurable_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    from orka.codebook._kmeans_torch import _kmeans_pp_init_torch

    torch.manual_seed(0)
    rows = torch.randn(4000, 8)

    def sampled(k, seed=5):
        g = torch.Generator(device=rows.device).manual_seed(int(seed) & ((1 << 63) - 1))
        return rows[torch.randperm(rows.shape[0], generator=g, device=rows.device)[:k]]

    monkeypatch.setenv("ORKA_KMEANS_PP_MAX_K", "64")
    assert torch.equal(_kmeans_pp_init_torch(rows, 128, seed=5), sampled(128))

    monkeypatch.setenv("ORKA_KMEANS_PP_MAX_K", "256")
    below = _kmeans_pp_init_torch(rows, 128, seed=5)
    assert below.shape == (128, 8)
    assert not torch.equal(below, sampled(128))


# ------------------------------------------------------- dead row-norm argument
def test_torch_assign_takes_no_row_norm_cache():
    pytest.importorskip("torch")
    from orka.codebook._kmeans_torch import _torch_assign

    params = inspect.signature(_torch_assign).parameters
    assert "r_norm_sq" not in params, (
        "the fused kernel scores with ||c||^2 - 2v.c, so a caller-supplied row-norm "
        "cache is unread; do not re-add the argument"
    )


def test_fused_assign_matches_bruteforce_argmin():
    torch = pytest.importorskip("torch")
    from orka.codebook._kmeans_torch import _torch_assign

    torch.manual_seed(3)
    rows = torch.randn(2000, 8)
    cb = torch.randn(64, 8)
    idx, _ = _torch_assign(rows, cb, "cpu", compute_mse=False)
    ref = torch.cdist(rows, cb).argmin(dim=1)
    assert torch.equal(idx.cpu(), ref)


# -------------------------------------------- mse_scale reads after the writer
def test_mse_scale_survives_a_slow_background_writer(tmp_path, monkeypatch):
    """_refine_scales_ls rebuilds the VQ reconstruction from the index/codebook files, which
    the BackgroundWriter writes asynchronously. Without a flush the read can miss a stream
    that is still queued and the refinement silently no-ops, so the manifest reports
    mse_scale=true with mse_scale_applied=false. Delay every index write to force the
    ordering the barrier has to handle."""
    torch = pytest.importorskip("torch")
    import time

    from orka.core._format import _write_stage_indices as real_write
    from orka.pipeline import pack_pipeline as PP

    def slow_write(path, indices, index_bits, stage_meta):
        time.sleep(0.15)
        return real_write(path, indices, index_bits, stage_meta)

    monkeypatch.setattr(PP, "_write_stage_indices", slow_write)

    torch.manual_seed(11)
    src = tmp_path / "m.pt"
    torch.save({"model.layers.0.mlp.up_proj.weight": torch.randn(32, 64)}, src)

    from orka.pipeline.pack import pack_checkpoint
    manifest = pack_checkpoint(
        source=src, out_dir=tmp_path / "a.orka", group_size=8, codebook_size=16,
        codebook_mode="per-tensor", backend="torch", device="cpu",
        normalization="block-max", sample_vectors=64, iterations=3, em_aq_passes=0,
        mse_scale=True,
    )
    assert manifest["mse_scale"] is True
    assert all(t.get("mse_scale_applied") for t in manifest["tensors"]), (
        "refinement must not be skipped just because the writer had not flushed yet"
    )


# ------------------------------------------------- giant path == normal path
def _pack(src: Path, out: Path, **kw):
    from orka.pipeline.pack import pack_checkpoint

    return pack_checkpoint(
        source=src, out_dir=out, group_size=8, codebook_sizes=[16, 16],
        codebook_mode="per-tensor", backend="torch", device=kw.pop("device", "cpu"),
        normalization="block-max", sample_vectors=64, iterations=4, em_aq_passes=0,
        **kw,
    )


def _metrics(manifest: dict):
    return [(t["name"], t["mse"], t["sqnr"], t["index_bytes"]) for t in manifest["tensors"]]


def test_giant_path_matches_normal_path_on_cuda(tmp_path, monkeypatch):
    """The giant path (host-resident candidate + tiled device assign) must reproduce the
    normal path's numbers on CUDA. tests/test_emaq_giant_gate.py pins this on CPU, where
    the offload is a no-op and the tiled assign never crosses a device boundary."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to exercise device residency")
    from orka._runtime import _apply_gpu_memory_cap
    from orka.codebook import _kmeans_torch

    _apply_gpu_memory_cap("torch", "cuda", 10.0)
    torch.manual_seed(5)
    src = tmp_path / "m.pt"
    torch.save({
        "model.layers.0.mlp.up_proj.weight": torch.randn(64, 32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }, src)

    normal = _pack(src, tmp_path / "normal.orka", device="cuda")
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 1)   # everything is "giant"
    giant = _pack(src, tmp_path / "giant.orka", device="cuda")
    assert _metrics(normal) == _metrics(giant)


# ------------------------------------------------------ batched semantic hubs
def _hubs_reference(embeddings: np.ndarray, threshold: float):
    """The per-index scan the batched implementation replaced."""
    import torch

    t = torch.from_numpy(embeddings)
    t = torch.nn.functional.normalize(t, p=2, dim=1)
    hubs = []
    processed = torch.zeros(t.shape[0], dtype=torch.bool)
    for r in (range(min(2500, t.shape[0])),
              range(max(0, t.shape[0] - 2500), t.shape[0])):
        for i in r:
            if processed[i]:
                continue
            sims = torch.mm(t[i:i + 1], t.T).squeeze(0)
            matches = torch.where(sims > threshold)[0]
            if len(matches) > 1:
                hubs.append({
                    "master_tid": int(i),
                    "member_count": int(len(matches)),
                    "member_tids": matches.tolist(),
                    "avg_similarity": float(sims[matches].mean().item()),
                })
                processed[matches] = True
    return sorted(hubs, key=lambda x: x["member_count"], reverse=True)


@pytest.mark.parametrize("chunk", ["1", "7", "512"])
def test_semantic_hubs_batched_matches_per_index(monkeypatch, chunk):
    pytest.importorskip("torch")
    from orka.quant.semantic import find_semantic_hubs

    rng = np.random.default_rng(1)
    emb = rng.normal(size=(120, 16)).astype(np.float32)
    emb[10] = emb[3]          # exact duplicates -> a hub
    emb[11] = emb[3]
    emb[119] = emb[118]       # a hub in the tail range

    monkeypatch.setenv("ORKA_SEMANTIC_HUB_CHUNK", chunk)
    got = find_semantic_hubs(emb, threshold=0.999)
    want = _hubs_reference(emb, threshold=0.999)
    assert [(h["master_tid"], h["member_count"], h["member_tids"]) for h in got] == \
           [(h["master_tid"], h["member_count"], h["member_tids"]) for h in want]


# ------------------------------------------------------- planar spec rate check
def test_planar_candidates_still_parse():
    from orka.quant.allocate import PLANAR_CANDIDATE_SPECS
    from orka.quant.spec import parse_quant_spec

    for spec in PLANAR_CANDIDATE_SPECS:
        assert parse_quant_spec(spec), f"shipped planar candidate must parse: {spec}"


def test_planar_spec_above_fp32_is_rejected():
    from orka.quant.allocate import _spec_bits_per_vector
    from orka.quant.spec import QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT, parse_quant_spec

    spec = "rvq-" + "-".join(["s8"] * 8)          # 64 bits per WEIGHT, not per vector
    with pytest.raises(ValueError, match="bits per weight"):
        parse_quant_spec(spec)

    ok = "rvq-" + "-".join(["s8"] * (QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT // 8))
    stages = parse_quant_spec(ok)
    assert _spec_bits_per_vector(stages, 8) / 8 == QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT


def test_vq_stage_bits_are_unaffected_by_the_scalar_ceiling():
    from orka.quant.spec import parse_quant_spec

    # 4 x 16-bit VQ stages = 64 per-vector bits: at the per-vector ceiling, well past the
    # scalar per-weight ceiling. Must still parse - the units are different.
    assert parse_quant_spec("rvq-16-16-16-16") == [1 << 16] * 4


# ----------------------------------------------------------- prefill crossover
def test_prefill_min_tokens_is_env_tunable(monkeypatch):
    pytest.importorskip("torch")
    from orka import config
    from orka.inference import cuda_decode

    assert config.prefill_min_tokens() == config.DEFAULT_PREFILL_MIN_TOKENS
    monkeypatch.setenv("ORKA_PREFILL_MIN_TOKENS", "17")
    assert config.prefill_min_tokens() == 17
    assert cuda_decode.prefill_min_tokens() == 17


# ------------------------------------------------- layering: no domain -> deploy
def test_no_domain_subpackage_imports_deploy_or_cli():
    """`deploy` and `cli` are entry points. A domain module reaching into them inverts the
    layering (a dead quant.semantic -> deploy.kaggle import did exactly that)."""
    root = Path(__file__).resolve().parent.parent / "orka"
    domains = ("core", "_runtime", "quant", "codebook", "transforms", "pipeline",
               "inference", "artifact", "eval", "qat", "autoquant", "integrations")
    offenders = []
    for domain in domains:
        for path in (root / domain).rglob("*.py"):
            text = path.read_text()
            for bad in ("orka.deploy", "orka.cli"):
                if bad in text:
                    offenders.append(f"{path.relative_to(root.parent)} -> {bad}")
    assert not offenders, "layering inversion: " + "; ".join(offenders)
