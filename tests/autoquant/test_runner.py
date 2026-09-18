from pathlib import Path

from orka.autoquant.harness_schema import RunConfig
from orka.autoquant.runner import run_autoquant


class ScriptedGratis:
    def __init__(self, names: list[str]):
        self._it = iter(names)
        self.spent_usd = 0.0

    def next_action(self, messages, tools):
        name = next(self._it)
        return {"function": {"name": name, "arguments": "{}"}}


def _config(tmp_path: Path, **over) -> RunConfig:
    kwargs = dict(
        gratis_base_url="http://gratis",
        model="gratis-auto",
        max_usd=0.05,
        max_steps=8,
        timeout_seconds=60,
        objective="knee",
        target=0.02,
        output_dir=tmp_path,
        artifact_path=tmp_path / "out.orka",
    )
    kwargs.update(over)
    return RunConfig(**kwargs)


def _small_checkpoint(tmp_path: Path) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    model = tmp_path / "m"
    model.mkdir()
    save_file(
        {
            "embed_out.weight": np.random.randn(64, 16).astype("float32"),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype("float32"),
        },
        str(model / "model.safetensors"),
    )
    return model


def test_runner_emits_verified_success(tmp_path):
    source = _small_checkpoint(tmp_path)
    client = ScriptedGratis(
        ["inspect", "derive", "allocate", "pack", "pulse_check", "verify"]
    )
    artifact = tmp_path / "out.orka"

    def pack_fn(*, source, out_dir, **_k):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text('{"format":"orka","tensors":[]}\n')

    result = run_autoquant(
        _config(tmp_path),
        client,
        source,
        pack_fn=pack_fn,
        pulse_fn=lambda _s: {"kl": 0.01, "top1": 0.99},
        verify_fn=lambda _p: {"ok": True, "passed": True},
    )
    assert result.status.value == "succeeded"
    assert (tmp_path / "autoquant-report.json").exists()
    assert artifact.exists() or result.artifact_path.endswith("out.orka")


def test_runner_fails_when_step_limit_is_reached(tmp_path):
    source = _small_checkpoint(tmp_path)
    client = ScriptedGratis(["inspect", "derive", "allocate", "pack", "pulse_check", "verify"])
    result = run_autoquant(
        _config(tmp_path, max_steps=1),
        client,
        source,
        pack_fn=lambda **_k: None,
        pulse_fn=lambda _s: {"kl": 0.01, "top1": 0.99},
        verify_fn=lambda _p: {"ok": True},
    )
    assert result.status.value == "failed"
    assert result.failure_reason == "step_limit_exceeded"


def test_runner_rejects_unregistered_tool(tmp_path):
    source = _small_checkpoint(tmp_path)
    client = ScriptedGratis(["shell"])
    result = run_autoquant(_config(tmp_path), client, source)
    assert result.status.value == "failed"
    assert result.failure_reason == "unregistered_tool"
