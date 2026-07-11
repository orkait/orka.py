"""Giant-path membership must be decided by ELEMENT count, not row count.

The old row gate misrouted scalar stages: a [45M-elem] tensor reshaped to
[45M, 1] for an s-stage crossed the 20M-row threshold and silently took the
CPU-resident tiled path (~700 tiny H2D chunks) although its memory footprint
is 24x below the 1B-head calibration point."""
import pytest

torch = pytest.importorskip("torch")

from orka.codebook import _kmeans_torch  # noqa: E402
from orka.codebook._kmeans_torch import _is_giant_matrix  # noqa: E402


def test_scalar_view_agrees_with_vector_view():
    n_vec, d = 6_000_000, 8
    assert _is_giant_matrix(n_vec, d) == _is_giant_matrix(n_vec * d, 1)


def test_45m_element_scalar_stage_is_not_giant():
    assert not _is_giant_matrix(45_000_000, 1)


def test_1b_head_stays_giant_in_both_layouts():
    n_vec, d = 127_000_000, 8
    assert _is_giant_matrix(n_vec, d)
    assert _is_giant_matrix(n_vec * d, 1)


def test_threshold_tracks_env_override(monkeypatch):
    monkeypatch.setattr(_kmeans_torch, "_LARGE_ASSIGN_ROWS", 100)
    assert _is_giant_matrix(101, 8)
    assert not _is_giant_matrix(101, 1)
