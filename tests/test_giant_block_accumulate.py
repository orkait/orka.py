"""The blocked (low-RAM) decode+accumulate+subtract for giant tensors must be
byte-identical to the full-materialization path it replaces. Only the RAM
footprint changes, never the numbers."""
import pytest

torch = pytest.importorskip("torch")

from orka.pipeline.pack_pipeline import _stage_decode_accumulate_blocked  # noqa: E402


def _full_path(vectors_orig, cb, indices, decoded_sum):
    """The original lines 355-367 of _quantize_and_record_stage, non-scalar."""
    decoded = cb.index_select(0, indices)
    decoded_sum = decoded if decoded_sum is None else decoded_sum + decoded
    residual = vectors_orig - decoded_sum
    return decoded_sum, residual


@pytest.mark.parametrize("n_rows,d,k", [(1000, 8, 16), (50000, 8, 256), (777, 4, 12)])
@pytest.mark.parametrize("stage0", [True, False])
@pytest.mark.parametrize("chunk", [1 << 20, 128])
def test_blocked_equals_full(n_rows, d, k, stage0, chunk):
    torch.manual_seed(0)
    vo = torch.randn(n_rows, d, dtype=torch.float32)
    cb = torch.randn(k, d, dtype=torch.float32)
    idx = torch.randint(0, k, (n_rows,), dtype=torch.int64)
    prev = None if stage0 else torch.randn(n_rows, d, dtype=torch.float32)

    exp_ds, exp_res = _full_path(vo, cb, idx, None if prev is None else prev.clone())

    c = {"vectors_orig": vo, "decoded_sum": None if prev is None else prev.clone()}
    _stage_decode_accumulate_blocked(c, cb, idx, is_scalar_stage=False, chunk_rows=chunk)

    assert torch.equal(c["decoded_sum"], exp_ds), "decoded_sum differs"
    assert torch.equal(c["vectors_residual"], exp_res), "residual differs"


def test_blocked_scalar_stage():
    """Scalar stage: v_res is reshaped [-1, 1], indices are per element (N*d)."""
    torch.manual_seed(1)
    n_rows, d, k = 2000, 8, 32
    vo = torch.randn(n_rows, d, dtype=torch.float32)
    cb = torch.randn(k, 1, dtype=torch.float32)
    idx = torch.randint(0, k, (n_rows * d,), dtype=torch.int64)

    decoded = cb.index_select(0, idx).reshape(n_rows, d)
    exp_ds = decoded
    exp_res = vo - exp_ds

    c = {"vectors_orig": vo, "decoded_sum": None}
    _stage_decode_accumulate_blocked(c, cb, idx, is_scalar_stage=True, chunk_rows=333)
    assert torch.equal(c["decoded_sum"], exp_ds)
    assert torch.equal(c["vectors_residual"], exp_res)
