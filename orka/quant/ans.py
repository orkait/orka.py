"""Static rANS entropy coder for quantization indices - CPU reference + GPU.

orka stores VQ/lattice indices with zlib, which sits ~16% above the symbol entropy
at high K. A static range-ANS coder hits the entropy (the index histogram is the
model) and the decode parallelizes on the GPU via *interleaved* streams (each with
its own rANS state, vectorized across the batch).

This file: the frequency-table builder + a scalar reference encoder/decoder (the
correctness oracle). The interleaved GPU codec (`ans_gpu.py` / functions below) is
validated bit-exact against this reference.

Format: 32-bit state, 16-bit renorm, `precision`-bit frequency table (total = 2^precision).
"""
from __future__ import annotations

import numpy as np

RANS_L = 1 << 16          # lower bound of the normalized state interval
RENORM_BITS = 16
RENORM_MASK = (1 << RENORM_BITS) - 1


def build_freq_table(symbols: np.ndarray, precision: int = 12):
    """Histogram -> frequencies normalized so they sum to exactly 2^precision, each >=1.

    Returns (freq, cum, n_symbols) with freq[s], cum[s] (cumulative start), int64.
    """
    if precision < 1 or precision > 16:
        raise ValueError("precision must be in [1, 16]")
    M = 1 << precision
    symbols = symbols.astype(np.int64).reshape(-1)
    n_sym = int(symbols.max()) + 1 if symbols.size else 1
    hist = np.bincount(symbols, minlength=n_sym).astype(np.int64)
    used = hist > 0
    # normalize to sum M, giving every USED symbol at least 1 (so it stays decodable)
    freq = np.zeros(n_sym, dtype=np.int64)
    scaled = np.maximum(1, np.round(hist[used] * (M / hist.sum()))).astype(np.int64)
    freq[used] = scaled
    # fix rounding so the total is exactly M (steal/add from the largest bin)
    diff = M - int(freq.sum())
    if diff != 0:
        j = int(np.argmax(freq))
        freq[j] += diff
        if freq[j] < 1:  # pathological tiny table; fall back to uniform-ish
            freq[used] = 1
            freq[int(np.argmax(hist))] += M - int(freq.sum())
    cum = np.zeros(n_sym + 1, dtype=np.int64)
    cum[1:] = np.cumsum(freq)
    return freq, cum, n_sym


def slot_to_symbol(freq: np.ndarray, precision: int) -> np.ndarray:
    """Inverse lookup: slot in [0, 2^precision) -> symbol (for decode)."""
    M = 1 << precision
    lut = np.zeros(M, dtype=np.int64)
    cum = np.cumsum(freq)
    start = 0
    for s, c in enumerate(cum):
        lut[start:c] = s
        start = int(c)
    return lut


def rans_encode_scalar(symbols: np.ndarray, freq: np.ndarray, cum: np.ndarray, precision: int):
    """Reference encoder. Returns a uint16 word array; decode reads it back-to-front."""
    M = 1 << precision
    x = RANS_L
    out = []
    syms = symbols.astype(np.int64).reshape(-1)
    for s in syms[::-1]:                       # rANS is a stack: encode in reverse
        f = int(freq[s])
        x_max = ((RANS_L >> precision) << RENORM_BITS) * f
        while x >= x_max:
            out.append(x & RENORM_MASK)
            x >>= RENORM_BITS
        x = ((x // f) << precision) + (x % f) + int(cum[s])
    out.append(x & RENORM_MASK)                # flush 32-bit final state (2 words)
    out.append((x >> RENORM_BITS) & RENORM_MASK)
    return np.array(out, dtype=np.uint16)


def rans_decode_scalar(words: np.ndarray, freq: np.ndarray, cum: np.ndarray, lut: np.ndarray, n: int, precision: int):
    """Reference decoder. Inverse of rans_encode_scalar."""
    M = 1 << precision
    mask = M - 1
    pos = len(words) - 1
    x = (int(words[pos]) << RENORM_BITS) | int(words[pos - 1])
    pos -= 2
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        slot = x & mask
        s = int(lut[slot])
        out[i] = s
        x = int(freq[s]) * (x >> precision) + slot - int(cum[s])
        while x < RANS_L:
            x = (x << RENORM_BITS) | int(words[pos])
            pos -= 1
    return out


# ---------------------------------------------------------------------------
# Interleaved GPU rANS: K independent streams, vectorized across the batch.
# With 16-bit renorm + <=16-bit precision the state interval is exactly one
# renorm wide, so each symbol emits/reads AT MOST one 16-bit word - no inner
# while-loop, so the whole codec vectorizes over K streams on the GPU.
# ---------------------------------------------------------------------------

def rans_encode_gpu(symbols, freq, cum, precision: int, K: int = 4096, device: str = "cuda"):
    import torch

    M = 1 << precision
    sym = torch.as_tensor(np.asarray(symbols).reshape(-1), dtype=torch.int64, device=device)
    n = int(sym.numel())
    L_s = (n + K - 1) // K
    pad = L_s * K - n
    if pad:
        # pad with a symbol that is GUARANTEED to have freq>0 (the most frequent one);
        # symbol 0 may be unused (freq 0) -> division by zero in the encode step.
        pad_sym = int(np.asarray(freq).argmax())
        sym = torch.cat([sym, torch.full((pad,), pad_sym, dtype=torch.int64, device=device)])
    streams = sym.reshape(L_s, K).t().contiguous()              # [K, L_s], stream k = padded[t*K+k]
    freq_t = torch.as_tensor(freq, dtype=torch.int64, device=device)
    cum_t = torch.as_tensor(cum[:-1] if cum.shape[0] == freq.shape[0] + 1 else cum,
                            dtype=torch.int64, device=device)
    x = torch.full((K,), RANS_L, dtype=torch.int64, device=device)
    maxw = L_s + 2
    out = torch.zeros((K, maxw), dtype=torch.int64, device=device)
    wpos = torch.zeros(K, dtype=torch.int64, device=device)
    ar = torch.arange(K, device=device)
    base = (RANS_L >> precision) << RENORM_BITS
    for t in range(L_s - 1, -1, -1):
        s = streams[:, t]
        f = freq_t[s]; c = cum_t[s]
        emit = x >= base * f
        ke = emit.nonzero(as_tuple=True)[0]
        if ke.numel():
            out[ke, wpos[ke]] = x[ke] & RENORM_MASK
            wpos[ke] += 1
            x[ke] >>= RENORM_BITS
        x = ((x // f) << precision) + (x % f) + c
    for _ in range(2):                                          # flush 32-bit state
        out[ar, wpos] = x & RENORM_MASK
        wpos += 1
        x >>= RENORM_BITS
    return out.to(torch.int32), wpos.to(torch.int32), n, L_s


def rans_decode_gpu(out, wpos, freq, cum, lut, n: int, L_s: int, precision: int, K: int, device: str = "cuda"):
    import torch

    M = 1 << precision
    mask = M - 1
    out = out.to(torch.int64).to(device)
    wpos = wpos.to(torch.int64).to(device)
    freq_t = torch.as_tensor(freq, dtype=torch.int64, device=device)
    cum_t = torch.as_tensor(cum[:-1] if cum.shape[0] == freq.shape[0] + 1 else cum,
                            dtype=torch.int64, device=device)
    lut_t = torch.as_tensor(lut, dtype=torch.int64, device=device)
    ar = torch.arange(K, device=device)
    # init state from the 2 flush words at the top of each stream
    x = (out[ar, wpos - 1] << RENORM_BITS) | out[ar, wpos - 2]
    rpos = wpos - 3
    streams = torch.empty((K, L_s), dtype=torch.int64, device=device)
    for t in range(L_s):
        slot = x & mask
        s = lut_t[slot]
        streams[:, t] = s
        x = freq_t[s] * (x >> precision) + slot - cum_t[s]
        read = x < RANS_L
        kr = read.nonzero(as_tuple=True)[0]
        if kr.numel():
            x[kr] = (x[kr] << RENORM_BITS) | out[kr, rpos[kr]]
            rpos[kr] -= 1
    flat = streams.t().contiguous().reshape(-1)[:n]
    return flat


# ---------------------------------------------------------------------------
# Self-contained blob codec: drop-in zlib replacement. The blob carries the
# frequency table + per-stream states, so it decodes standalone. Static rANS
# pays a freq-table cost (n_sym x 2B), so it wins over zlib on streams that are
# large enough to amortize it (~>=50k symbols); for tiny streams prefer zlib.
# ---------------------------------------------------------------------------

_MAGIC = b"rANS"


def _auto_K(n: int) -> int:
    # balance GPU parallelism vs the 32-bit/stream flush overhead (~K*32/n bpw)
    return max(64, min(16384, n // 512))


def ans_compress(symbols, precision: int | None = None, K: int | None = None, device: str = "cuda") -> bytes:
    import struct
    import math
    import torch

    sym = np.asarray(symbols).reshape(-1).astype(np.int64)
    n = int(sym.size)
    if precision is None:
        n_sym = int(sym.max()) + 1 if n else 1
        # precision must satisfy 2^precision >= n_sym (every symbol gets freq>=1),
        # with headroom for the distribution; cap at 16.
        precision = min(16, max(12, int(math.ceil(math.log2(max(2, n_sym)))) + 1))
    if n == 0:
        return _MAGIC + struct.pack("<QIBI", 0, 0, precision, 0)
    K = K or _auto_K(n)
    freq, cum, n_sym = build_freq_table(sym, precision)
    if int(freq.max()) > 0xFFFF:
        raise ValueError("freq exceeds uint16; use precision <= 16 with fewer symbols")
    out, wpos, n_ret, L_s = rans_encode_gpu(sym, freq, cum, precision, K, device)
    wpos_np = wpos.cpu().numpy().astype(np.uint32)
    out_np = out.cpu().numpy()
    words = np.concatenate([out_np[k, : wpos_np[k]] for k in range(K)]).astype(np.uint16)
    head = _MAGIC + struct.pack("<QIBI", n, K, precision, n_sym)
    return (head + freq.astype(np.uint16).tobytes() + wpos_np.tobytes() + words.tobytes())


def ans_decompress(blob: bytes, device: str = "cuda"):
    import struct
    import torch

    if blob[:4] != _MAGIC:
        raise ValueError("not an rANS blob")
    n, K, precision, n_sym = struct.unpack("<QIBI", blob[4:21])
    off = 21
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    freq = np.frombuffer(blob, dtype=np.uint16, count=n_sym, offset=off).astype(np.int64); off += n_sym * 2
    wpos = np.frombuffer(blob, dtype=np.uint32, count=K, offset=off).astype(np.int64); off += K * 4
    words = np.frombuffer(blob, dtype=np.uint16, offset=off).astype(np.int64)
    cum = np.zeros(n_sym + 1, dtype=np.int64); cum[1:] = np.cumsum(freq)
    lut = slot_to_symbol(freq, precision)
    # rebuild the padded [K, maxw] out buffer from the concatenated words
    maxw = int(wpos.max())
    out = torch.zeros((K, maxw), dtype=torch.int64, device=device)
    offs = np.zeros(K + 1, dtype=np.int64); offs[1:] = np.cumsum(wpos)
    words_t = torch.as_tensor(words, dtype=torch.int64, device=device)
    for k in range(K):  # K is modest; this just scatters slices back
        w = wpos[k]
        if w:
            out[k, :w] = words_t[offs[k]: offs[k] + w]
    L_s = (n + K - 1) // K
    wpos_t = torch.as_tensor(wpos, dtype=torch.int64, device=device)
    return rans_decode_gpu(out, wpos_t, freq, cum, lut, n, L_s, precision, K, device)


