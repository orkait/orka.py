"""Universal .orka -> orka.llama GGUF exporter. Architecture-agnostic by design.

The fork's `conversion` package (its convert_hf_to_gguf) owns EVERYTHING per-arch:
KV metadata, vocab, tensor-name mapping, and per-arch weight transforms. This script
only changes two things:

  1. the weight SOURCE: tensors come from the .orka artifact (decoded values for
     quantized tensors, passthrough for the rest) instead of the raw HF checkpoint,
     so the exported model IS the compressed model;
  2. the weight ENCODING: eligible quantized linears are emitted as orka RVQ side
     tensors ({name}.idxlo{s}/idxhi{s}/cb{s}/scales[, corr_ptr/corr_col/corr_val])
     instead of dense data - matched on the fork loader's arch-agnostic contract.

A new architecture needs ZERO changes here: the day upstream conversion supports it,
this exporter does too.

Side tensors are emitted only when the conversion pipeline's transform of the weight
is the identity or a pure row permutation (detected generically by row hashing - e.g.
qwen3.5's V-head reorder). Any other transform falls back to dense emission of the
transformed decoded weight, which is always correct.

Eligibility: 2D, >= --min-rvq-params, every stage a group-`orka.group_size` vector
stage, and not token_embd/output (those are consumed by get_rows, not mat-mul).

Requires a llama.cpp checkout for its `conversion/` package, located via LLAMA_CPP_DIR
(ORKA_LLAMA_DIR is still accepted). Upstream llama.cpp is sufficient - the fork is needed to
SERVE the result, not to produce it. An installed `gguf` is preferred over the checkout's
gguf-py.

Usage:
    LLAMA_CPP_DIR=~/llama.cpp \
    python scripts/export_gguf_orka.py <artifact.orka> <hf_model_dir> <out.gguf>
        [--outtype q8_0|f16] [--skip-tensors REGEX] [--set-hparam key=value ...]
        [--min-rvq-params N]

--min-rvq-params defaults to 4,000,000: the codebook tax (stages x K x group_size x dtype
bytes) is charged per tensor regardless of its size, so RVQ does not pay on small tensors.
Lower it to exercise the path on small models.

Ornith-9B text-only example:
    python scripts/export_gguf_orka.py ornith.orka ornith-9b out.gguf \
        --outtype q8_0 --skip-tensors '^model\\.visual\\.' \
        --set-hparam mtp_num_hidden_layers=0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This script is a bridge: it decodes an .orka artifact (orka.py internals, imported below)
# and writes it through llama.cpp's own conversion classes. The llama.cpp side is a plain
# upstream dependency - `conversion/` is upstream, and orka.llama's only change to it is a
# tokenizer checksum that is itself upstream now. Any recent checkout works; the fork is not
# required to EXPORT, only to SERVE the result.
_LLAMA_ENV = ("LLAMA_CPP_DIR", "ORKA_LLAMA_DIR")
_llama_dir = next((os.environ[k] for k in _LLAMA_ENV if os.environ.get(k)), None)
_FORK = Path(_llama_dir) if _llama_dir else \
    Path(__file__).resolve().parent.parent.parent / "orka.llama"
if not (_FORK / "conversion").is_dir():
    raise SystemExit(
        f"llama.cpp checkout not found at {_FORK}\n"
        f"  set LLAMA_CPP_DIR to a llama.cpp (or orka.llama) checkout containing conversion/\n"
        f"  e.g. LLAMA_CPP_DIR=~/llama.cpp {Path(__file__).name} ..."
    )
# Prefer an installed gguf; fall back to the checkout's gguf-py only if it is absent.
try:
    import gguf  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_FORK / "gguf-py"))
sys.path.insert(0, str(_FORK))

import gguf  # noqa: E402
import torch  # noqa: E402
from conversion import ModelBase, ModelType, get_model_architecture, get_model_class  # noqa: E402

from orka.core._format import _pack_index_planes  # noqa: E402
from orka.inference.vq_linear import build_vq_linear  # noqa: E402
from orka.pipeline.decode import _decode_tensor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("hf_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--outtype", default="q8_0", choices=["q8_0", "f16", "bf16", "f32"])
    p.add_argument("--skip-tensors", default=None,
                   help="regex of HF tensor names to exclude (e.g. a vision tower)")
    p.add_argument("--set-hparam", action="append", default=[],
                   help="override an hparams key, e.g. mtp_num_hidden_layers=0")
    p.add_argument("--min-rvq-params", type=int, default=4_000_000)
    p.add_argument("--no-corrections", action="store_true",
                   help="drop salient/outlier CSR corrections (smaller + faster decode, "
                        "lower quality) - for isolating the per-token correction cost")
    return p.parse_args()


def _coerce(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


class OrkaArtifact:
    """Read side of a .orka artifact: names, decoded tensors, RVQ side payloads."""

    def __init__(self, path: Path, min_rvq_params: int):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.tmeta = {t["name"]: t for t in self.manifest["tensors"]}
        self.group = int(self.manifest["group_size"])
        self.min_rvq_params = min_rvq_params
        from safetensors import safe_open
        self._pt = safe_open(str(path / "passthrough.safetensors"), "pt")
        self.names = sorted(set(self._pt.keys()) | set(self.tmeta))

    def tensor(self, name: str) -> torch.Tensor:
        if name in self.tmeta:
            tm = self.tmeta[name]
            arr = np.asarray(_decode_tensor(self.path, tm), dtype=np.float32)
            return torch.from_numpy(arr.reshape([int(x) for x in tm["shape"]]))
        return self._pt.get_tensor(name).float()

    def rvq_eligible(self, name: str) -> bool:
        tm = self.tmeta.get(name)
        if tm is None or len(tm["shape"]) != 2:
            return False
        if int(np.prod(tm["shape"])) < self.min_rvq_params:
            return False
        return all(
            int(s.get("group_size", tm["group_size"])) == self.group
            for s in tm["stages"]
        )

    def side_payload(self, name: str, perm: np.ndarray | None):
        """(stage list, scales, corr) row-major, rows permuted by ``perm`` if given."""
        layer = build_vq_linear(self.path, self.tmeta[name], bias=None, device="cpu")
        M, K = layer.out_features, layer.in_features
        GPR, BPR = K // layer.group_size, K // layer.block_size
        gm = bool(getattr(layer, "_group_major", False))
        stages = []
        for s in range(layer.n_stages):
            cb = getattr(layer, f"codebook_{s}").cpu().numpy().astype(np.float16)
            width = max(1, int(round(np.log2(cb.shape[0]))))
            idx = layer._stage_indices_int(s).cpu().numpy().astype(np.int64)
            idx = idx.reshape(GPR, M).T if gm else idx.reshape(M, GPR)
            if perm is not None:
                idx = idx[perm]
            lo, hi = _pack_index_planes(np.ascontiguousarray(idx).reshape(-1), width)
            stages.append((lo, hi if hi.size else np.zeros(1, np.uint8), cb.reshape(-1)))
        sc = layer.scales.cpu().numpy().astype(np.float16)
        sc = (sc.reshape(BPR, M).T if gm else sc.reshape(M, BPR))
        if perm is not None:
            sc = sc[perm]
        sc = np.ascontiguousarray(sc).reshape(-1)
        corr = None
        if int(layer.corr_col.numel()):
            ptr = layer.corr_rowptr.cpu().numpy().astype(np.int64)
            col = layer.corr_col.cpu().numpy().astype(np.int32)
            val = layer.corr_val.cpu().numpy().astype(np.float16)
            if perm is not None:
                counts = np.diff(ptr)[perm]
                new_ptr = np.zeros(M + 1, dtype=np.int64)
                np.cumsum(counts, out=new_ptr[1:])
                new_col = np.empty_like(col)
                new_val = np.empty_like(val)
                for i, r in enumerate(perm):
                    new_col[new_ptr[i]:new_ptr[i + 1]] = col[ptr[r]:ptr[r + 1]]
                    new_val[new_ptr[i]:new_ptr[i + 1]] = val[ptr[r]:ptr[r + 1]]
                ptr, col, val = new_ptr, new_col, new_val
            corr = (ptr.astype(np.int32), col, val)
        return stages, sc, corr, (M, K), layer.block_size


def row_permutation(src: torch.Tensor, dst: torch.Tensor) -> np.ndarray | None:
    """perm with dst[i] == src[perm[i]] for a pure row shuffle, else None.

    Generic detector for conversion transforms that only reorder output rows
    (e.g. qwen3.5's V-head grouped->tiled reorder). Row-hash based; bails to
    None (dense fallback) on shape change or ambiguous duplicate rows."""
    if src.shape != dst.shape or src.ndim != 2:
        return None
    s = np.ascontiguousarray(src.detach().cpu().numpy())
    d = np.ascontiguousarray(dst.detach().cpu().numpy())
    def rh(m):
        return [hashlib.blake2b(m[i].tobytes(), digest_size=16).digest() for i in range(m.shape[0])]
    sh, dh = rh(s), rh(d)
    pos: dict[bytes, list[int]] = {}
    for i, h in enumerate(sh):
        pos.setdefault(h, []).append(i)
    if any(len(v) > 1 for v in pos.values()):
        return None
    try:
        perm = np.array([pos[h][0] for h in dh], dtype=np.int64)
    except KeyError:
        return None
    if np.array_equal(perm, np.arange(len(perm))):
        return perm  # identity is a valid (trivial) permutation
    return perm


def main():
    args = parse_args()
    art = OrkaArtifact(args.artifact, args.min_rvq_params)
    skip_re = re.compile(args.skip_tensors) if args.skip_tensors else None

    hparams = ModelBase.load_hparams(args.hf_dir, False)
    for kv in args.set_hparam:
        k, _, v = kv.partition("=")
        hparams[k] = _coerce(v)
        txt = hparams.get("text_config")
        if isinstance(txt, dict) and k in txt:
            txt[k] = _coerce(v)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    base_cls = get_model_class(arch)
    stats = {"rvq": 0, "rvq_perm": 0, "dense_transformed": 0, "dense": 0, "nnz": 0}

    canon2art: dict[str, str] = {}

    class OrkaModel(base_cls):  # type: ignore[misc, valid-type]
        model_arch = base_cls.model_arch

        def index_tensors(self, remote_hf_model_id=None):
            # Weight SOURCE swap: artifact instead of the HF checkpoint, run through the
            # SAME upstream filter chain (multimodal skips, name canonicalization).
            tensors = {}
            for name in art.names:
                if skip_re is not None and skip_re.search(name):
                    continue
                data_gen = lambda n=name: art.tensor(n)  # noqa: E731
                if titem := self.filter_tensors((name, data_gen)):
                    tname, tgen = titem
                    tensors[tname] = tgen
                    canon2art[tname] = name
            return tensors

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            self.gguf_writer.add_uint32("orka.rvq", 1)
            self.gguf_writer.add_uint32("orka.group_size", art.group)
            self.gguf_writer.add_uint32("orka.block_size", 32)
            self.gguf_writer.add_uint32("orka.group_major", 0)

        def _emit_side(self, lc: str, hf_name: str, perm):
            stages, sc, corr, (M, K), block = art.side_payload(hf_name, perm)
            for s, (lo, hi, cb) in enumerate(stages):
                self.gguf_writer.add_tensor(f"{lc}.idxlo{s}", lo.view(np.int8))
                self.gguf_writer.add_tensor(f"{lc}.idxhi{s}", hi.view(np.int8))
                self.gguf_writer.add_tensor(f"{lc}.cb{s}", cb)
            self.gguf_writer.add_tensor(f"{lc}.scales", sc)
            if corr is not None and not args.no_corrections:
                ptr, col, val = corr
                self.gguf_writer.add_tensor(f"{lc}.corr_ptr", ptr)
                self.gguf_writer.add_tensor(f"{lc}.corr_col", col)
                self.gguf_writer.add_tensor(f"{lc}.corr_val", val)
                stats["nnz"] += int(col.size)

        def modify_tensors(self, data_torch, name, bid):
            out = list(super().modify_tensors(data_torch, name, bid))
            art_name = canon2art.get(name, name)
            kept = []
            for lc, t in out:
                if (
                    len(out) == 1
                    and lc.endswith(".weight")
                    and lc not in ("token_embd.weight", "output.weight")
                    and art.rvq_eligible(art_name)
                ):
                    perm = (
                        np.arange(t.shape[0], dtype=np.int64)
                        if t is data_torch
                        else row_permutation(data_torch, t)
                    )
                    if perm is not None:
                        trivial = bool(np.array_equal(perm, np.arange(len(perm))))
                        self._emit_side(lc, art_name, None if trivial else perm)
                        stats["rvq" if trivial else "rvq_perm"] += 1
                        print(f"rvq{'(perm)' if not trivial else ''}  {lc} <- {art_name}")
                        continue
                    stats["dense_transformed"] += 1
                    print(f"dense(transformed) {lc} <- {art_name}")
                else:
                    stats["dense"] += 1
                kept.append((lc, t))
            return kept

    ftype = {
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "f32": gguf.LlamaFileType.ALL_F32,
    }[args.outtype]
    model = OrkaModel(args.hf_dir, ftype, args.out, eager=True, hparams=hparams)
    model.write()
    print(
        f"wrote {args.out}: rvq={stats['rvq']} rvq_perm={stats['rvq_perm']} "
        f"dense_transformed={stats['dense_transformed']} dense={stats['dense']} "
        f"corr_nnz={stats['nnz']}"
    )


if __name__ == "__main__":
    main()
