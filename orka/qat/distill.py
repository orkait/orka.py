"""Post-pack codebook distillation.

Indices stay frozen; stage codebooks are continuous parameters, so the decode
chain (stage sum -> outlier/pillar injection -> un-rotate -> un-normalize ->
salient injection) is differentiable in the codebooks. Adam minimizes
reconstruction error against the original weights, optionally column-weighted
by calibration activation energy E[x^2] (output-error proxy).

The artifact format is unchanged: only codebook bytes are rewritten and the
manifest metrics refreshed via the production numpy decoder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from orka.core._checkpoint import _load_tensors
from orka.core._format import (
    _cast_codebook_storage,
    _float_value_dtype,
    _read_codebook,
    _read_float_vector,
    _read_indices,
    _read_outliers,
    _read_pillars,
    _read_salient,
    _write_codebook,
)
from orka.core._tensor import _numpy_float32_array
from orka.eval.metrics import quality_metrics_from_flat
from orka.pipeline.decode import _decode_tensor
from orka.transforms.normalize import stores_block_scales
from orka.transforms.rotate import _generate_orthogonal_numpy, _hadamard_block_size


def _fwht_autograd(x):
    """Out-of-place FWHT on the last dim (power of 2). Autograd-safe."""
    import torch

    n = int(x.shape[-1])
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 last dim, got {n}")
    lead = x.shape[:-1]
    h = 1
    while h < n:
        v = x.reshape(*lead, n // (2 * h), 2, h)
        a = v[..., 0, :]
        b = v[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, n)
        h *= 2
    return x * (1.0 / math.sqrt(n))


def _injection_consts(positions, values, length, device):
    import numpy as np
    import torch

    mask = torch.zeros(length, dtype=torch.bool, device=device)
    vals = torch.zeros(length, dtype=torch.float32, device=device)
    pos = torch.from_numpy(np.asarray(positions, dtype=np.int64)).to(device)
    val = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    keep = pos < length
    mask[pos[keep]] = True
    vals[pos[keep]] = val[keep]
    return mask, vals


def _load_decode_consts(out_dir: Path, tm: dict, device: str) -> dict:
    """Read every fixed quantity of one tensor's decode chain."""
    import numpy as np
    import torch

    group_size = int(tm["group_size"])
    padded = int(tm["padded_values"])
    packed = int(tm["packed_values"])
    shape = [int(x) for x in tm["shape"]]
    stages = tm.get("stages") or [
        {
            "codebook": tm["codebook"],
            "codebook_size": int(tm["codebook_size"]),
            "index_bits": int(tm["index_bits"]),
            "indices": tm["indices"],
            "group_size": group_size,
        }
    ]

    stage_data = []
    for stage in stages:
        s_g = int(stage.get("group_size", group_size))
        s_count = math.ceil(padded / s_g)
        cb_np = _read_codebook(
            out_dir / stage["codebook"], s_g, stage.get("codebook_dtype", "float32")
        )
        idx_np = np.asarray(
            _read_indices(
                out_dir / stage["indices"],
                int(stage["index_bits"]),
                s_count,
                packed=bool(stage.get("packed", False)),
                encoding=stage.get("encoding", "raw"),
            ),
            dtype=np.int64,
        )
        stage_data.append(
            {
                "codebook": torch.from_numpy(cb_np.copy()).to(device),
                "indices": torch.from_numpy(idx_np).to(device),
                "path": stage["codebook"],
                "dtype": stage.get("codebook_dtype", "float32"),
                "meta": stage,
            }
        )

    consts: dict = {
        "stages": stage_data,
        "padded": padded,
        "packed": packed,
        "shape": shape,
        "rows": shape[0],
        "cols": max(1, int(np.prod(shape[1:]))) if len(shape) > 1 else 1,
    }

    outl = tm.get("outliers")
    if outl:
        positions, values = _read_outliers(
            out_dir / outl["positions"],
            out_dir / outl["values"],
            int(outl["count"]),
            outl.get("positions_dtype", "uint32"),
            outl.get("values_dtype", "float32"),
        )
        if positions.size:
            consts["outliers"] = _injection_consts(positions, values, packed, device)

    pillars = tm.get("pillars")
    if pillars:
        positions, values = _read_pillars(
            out_dir / pillars["positions"], out_dir / pillars["values"]
        )
        if positions.size:
            consts["pillars"] = _injection_consts(positions, values, packed, device)

    rotation = tm.get("rotation", "none")
    consts["rotation"] = rotation
    if rotation == "orthogonal":
        q = _generate_orthogonal_numpy(consts["cols"], int(tm.get("rotation_seed") or 0))
        consts["q"] = torch.from_numpy(q).to(device)
    elif rotation == "hadamard":
        consts["hadamard_block"] = _hadamard_block_size(consts["cols"])

    norm = tm.get("normalization", "none")
    consts["normalization"] = norm
    scale_np_dtype = _float_value_dtype(tm.get("scale_dtype") or "float32")
    if stores_block_scales(norm):
        scales = _read_float_vector(out_dir / tm["scales"], int(tm["scale_count"]), tm.get("scale_dtype") or "float32")
        consts["block_scales"] = torch.from_numpy(scales).to(device)
        consts["block_scale_size"] = int(tm.get("block_scale_size") or 32)
        if norm == "awq-block-max" and tm.get("awq_col_scales"):
            awq_meta = tm["awq_col_scales"]
            awq = _read_float_vector(out_dir / awq_meta["path"], int(awq_meta["count"]), awq_meta.get("dtype") or "float32")
            consts["awq_col"] = torch.from_numpy(awq).to(device)
    elif norm == "awq":
        scales = _read_float_vector(out_dir / tm["scales"], int(tm["scale_count"]), tm.get("scale_dtype") or "float32")
        consts["awq_col"] = torch.from_numpy(scales).to(device)

    salient = tm.get("salient")
    if salient:
        s_idx, s_val = _read_salient(
            out_dir / salient["indices"],
            out_dir / salient["weights"],
            int(salient["count"]),
            int(salient["indices_bits"]),
            salient.get("weights_dtype", "float32"),
        )
        if s_idx.size:
            block_size = int(tm.get("block_scale_size") or 32)
            flat_idx = (
                np.arange(s_idx.shape[0], dtype=np.int64) * block_size
                + s_idx.astype(np.int64)
            )
            consts["salient"] = _injection_consts(flat_idx, s_val, packed, device)

    return consts


def _differentiable_decode(codebooks, consts):
    """Mirror of ``_decode_tensor`` with out-of-place autograd-safe ops."""
    import torch

    padded = consts["padded"]
    packed = consts["packed"]
    decoded = None
    for cb, stage in zip(codebooks, consts["stages"]):
        part = cb[stage["indices"]].reshape(-1)[:padded]
        decoded = part if decoded is None else decoded + part
    decoded = decoded[:packed]

    if "outliers" in consts:
        mask, vals = consts["outliers"]
        decoded = torch.where(mask, vals, decoded)
    if "pillars" in consts:
        mask, vals = consts["pillars"]
        decoded = torch.where(mask, vals, decoded)

    rows, cols = consts["rows"], consts["cols"]
    if consts["rotation"] == "orthogonal":
        decoded = (decoded[: rows * cols].reshape(rows, cols) @ consts["q"].T).reshape(-1)
    elif consts["rotation"] == "hadamard":
        b = consts["hadamard_block"]
        mat = decoded[: rows * cols].reshape(rows, cols // b, b)
        decoded = _fwht_autograd(mat).reshape(-1)

    norm = consts["normalization"]
    if "block_scales" in consts:
        bs = consts["block_scale_size"]
        n = decoded.shape[0]
        pad = (-n) % bs
        if pad:
            decoded = torch.cat(
                [decoded, torch.zeros(pad, dtype=decoded.dtype, device=decoded.device)]
            )
        blocks = decoded.reshape(-1, bs)
        decoded = (blocks * consts["block_scales"][: blocks.shape[0], None]).reshape(-1)
        if pad:
            decoded = decoded[:n]
    if "awq_col" in consts:
        decoded = (
            decoded[: rows * cols].reshape(rows, cols) * consts["awq_col"][None, :]
        ).reshape(-1)

    if "salient" in consts:
        mask, vals = consts["salient"]
        decoded = torch.where(mask, vals, decoded)

    return decoded


def _column_importance(activations, name: str, cols: int, device):
    import torch

    if not activations or name not in activations:
        return None
    # Move to the compute device FIRST so the E[x^2] reduction runs on the GPU
    # instead of CPU (the activations arrive as CPU tensors from calibration).
    acts = torch.as_tensor(activations[name], dtype=torch.float32, device=device)
    if acts.dim() != 2 or int(acts.shape[1]) != cols:
        return None
    h = acts.pow(2).mean(dim=0).clamp(min=1e-8)
    return h / h.mean()


def _output_loss_matrix(activations, name: str, cols: int, device, max_samples: int = 512):
    """X^T (subsampled, RMS-normalized) for the true output-space loss
    ||(W_hat - W) @ X^T||^2 = tr(dW H dW^T) - the full-Hessian objective,
    versus the diagonal proxy of ``_column_importance``."""
    import torch

    if not activations or name not in activations:
        return None
    # Device-first: the RMS-norm + transpose build the Hessian factor on the GPU.
    acts = torch.as_tensor(activations[name], dtype=torch.float32, device=device)
    if acts.dim() != 2 or int(acts.shape[1]) != cols:
        return None
    if int(acts.shape[0]) > max_samples:
        step = max(1, int(acts.shape[0]) // max_samples)
        acts = acts[::step][:max_samples]
    rms = acts.pow(2).mean().sqrt().clamp(min=1e-8)
    return (acts / rms).T.contiguous()


def _distill_tensor(
    out_dir: Path,
    tm: dict,
    source_tensor,
    *,
    steps: int,
    lr: float,
    device: str,
    activations: dict | None,
    patience: int = 25,
    output_space: bool = True,
) -> dict:
    import numpy as np
    import torch

    consts = _load_decode_consts(out_dir, tm, device)
    packed = consts["packed"]
    target = torch.from_numpy(
        _numpy_float32_array(source_tensor).reshape(-1)[:packed].copy()
    ).to(device)
    full_matrix = packed == consts["rows"] * consts["cols"]

    # Candidate objectives. Which one generalizes is layer-dependent (full-H
    # wins on o_proj/mlp inputs, the diagonal proxy on highly anisotropic
    # q/k inputs - measured on SmolLM2), so with activations available BOTH
    # run and an internal held-out split picks the winner per tensor.
    xt_fit = None
    x_val = None
    h = None
    if activations and tm["name"] in activations and full_matrix:
        acts = torch.as_tensor(activations[tm["name"]], dtype=torch.float32, device=device)
        if acts.dim() == 2 and int(acts.shape[1]) == consts["cols"] and int(acts.shape[0]) >= 64:
            n = int(acts.shape[0])
            val = acts[3::4][: max(16, n // 4)]
            fit = {tm["name"]: torch.cat([acts[0::4], acts[1::4], acts[2::4]], dim=0)}
            if output_space:
                xt_fit = _output_loss_matrix(fit, tm["name"], consts["cols"], device)
            h = _column_importance(fit, tm["name"], consts["cols"], device)
            x_val = val
        else:
            h = _column_importance(activations, tm["name"], consts["cols"], device)

    def _diag_loss(decoded):
        diff = decoded - target
        if h is not None and full_matrix:
            return (
                diff.reshape(consts["rows"], consts["cols"]).pow(2) * h[None, :]
            ).mean()
        return diff.pow(2).mean()

    def _output_loss(decoded):
        # Damped full-Hessian objective (GPTQ-style damping): the isotropic
        # term regularizes directions absent from the calibration sample.
        mat = (decoded - target).reshape(consts["rows"], consts["cols"])
        return (mat @ xt_fit).pow(2).mean() + 0.01 * consts["cols"] * mat.pow(2).mean()

    def _val_error(state):
        with torch.no_grad():
            decoded = _differentiable_decode(state, consts)
            mat = (decoded - target).reshape(consts["rows"], consts["cols"])
            return float((x_val @ mat.T).pow(2).mean().item())

    def _optimize(loss_fn):
        params = [
            torch.nn.Parameter(stage["codebook"].clone()) for stage in consts["stages"]
        ]
        opt = torch.optim.Adam(params, lr=lr)
        with torch.no_grad():
            init = float(loss_fn(_differentiable_decode(params, consts)).item())
        best = init
        state = [p.detach().clone() for p in params]
        since = 0
        for _step in range(steps):
            opt.zero_grad()
            loss = loss_fn(_differentiable_decode(params, consts))
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur = float(loss_fn(_differentiable_decode(params, consts)).item())
            if cur < best - 1e-12:
                best, state, since = cur, [p.detach().clone() for p in params], 0
            else:
                since += 1
                if since >= patience:
                    break
        return state, init, best

    chosen_objective = "diag" if h is not None else "plain"
    best_state, initial_loss, best_loss = _optimize(_diag_loss)
    if xt_fit is not None and x_val is not None:
        out_state, out_init, out_best = _optimize(_output_loss)
        if _val_error(out_state) < _val_error(best_state):
            best_state, initial_loss, best_loss = out_state, out_init, out_best
            chosen_objective = "output"

    for stage, cb in zip(consts["stages"], best_state):
        cast_cb, actual_dtype = _cast_codebook_storage(cb.cpu(), dtype=stage["dtype"])
        if actual_dtype != stage["dtype"]:
            stage["meta"]["codebook_dtype"] = actual_dtype
        _write_codebook(out_dir / stage["path"], cast_cb, dtype=actual_dtype)

    # Refresh manifest metrics through the PRODUCTION decoder so verify stays
    # exact - this also cross-checks the differentiable mirror.
    decoded_np = _decode_tensor(out_dir, tm)
    metrics = quality_metrics_from_flat(
        np.asarray(target.cpu().numpy(), dtype=np.float32), decoded_np
    )
    for key in (
        "mse", "sse", "rmse", "mae", "max_abs_error", "source_l2_sq",
        "reconstructed_l2_sq", "dot", "relative_rmse", "cosine_similarity", "sqnr",
    ):
        tm[key] = metrics[key]

    return {
        "name": tm["name"],
        "initial_loss": initial_loss,
        "final_loss": best_loss,
        "mse": metrics["mse"],
        "improved": best_loss < initial_loss,
        "objective": chosen_objective,
    }


def distill_artifact(
    artifact_dir: Path,
    *,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    activations: dict | None = None,
    max_tensors: int | None = None,
    progress: bool = True,
    output_space: bool = True,
) -> dict:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("distill requires torch") from exc

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    packed_meta = {t["name"]: t for t in manifest.get("tensors", [])}

    results = []
    done = 0
    for name, tensor in _load_tensors(source):
        if name not in packed_meta:
            continue
        if max_tensors is not None and done >= max_tensors:
            break
        if progress:
            print(f"Distilling {name} ({done + 1})...", flush=True)
        results.append(
            _distill_tensor(
                artifact_dir,
                packed_meta[name],
                tensor,
                steps=steps,
                lr=lr,
                device=device,
                activations=activations,
                output_space=output_space,
            )
        )
        done += 1

    manifest["distilled"] = {
        "steps": steps,
        "lr": lr,
        "tensor_count": len(results),
        "weighted": activations is not None,
        "output_space": output_space and activations is not None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    improved = sum(1 for r in results if r["improved"])
    return {
        "artifact": str(artifact_dir),
        "tensor_count": len(results),
        "improved_count": improved,
        "results": results,
    }
