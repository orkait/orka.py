"""Chunked bit-plane packing must be byte-identical to the single-shot
reference. Only the peak RAM changes (the [N, bits] bitmat is built per chunk
instead of for the whole stream), never the bytes."""
import numpy as np
import pytest

from orka.core._format import _pack_indices, _unpack_indices


def _pack_reference(indices, bits):
    """The original single-shot implementation, kept as the byte oracle."""
    arr = np.asarray(indices, dtype=np.uint64).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    bitmat = ((arr[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bitmat.reshape(-1))


@pytest.mark.parametrize("bits", [1, 3, 5, 8, 12, 16, 20])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 100, 4096, 4097, 10000])
def test_chunked_pack_matches_reference(bits, count):
    rng = np.random.default_rng(count * 31 + bits)
    idx = rng.integers(0, 1 << bits, size=count, dtype=np.uint64)
    ref = _pack_reference(idx, bits)
    for chunk in (8, 64, 1 << 22):
        packed = _pack_indices(idx, bits, chunk_rows=chunk)
        assert np.array_equal(packed, ref), f"bytes differ at chunk={chunk}"
        back = _unpack_indices(packed, bits, count, chunk_rows=chunk)
        assert np.array_equal(back, idx.astype(np.int64)), f"roundtrip differs at chunk={chunk}"


def test_default_chunk_is_byte_aligned():
    from orka.core._format import _PACK_CHUNK_INDICES

    assert _PACK_CHUNK_INDICES % 8 == 0
