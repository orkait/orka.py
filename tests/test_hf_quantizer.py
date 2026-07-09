"""Native transformers quantizer wiring for .orka.

Locks the pieces that do not need a full HF architecture to load:
  * the "orka" quant method registers (idempotently) into transformers' mapping,
  * OrkaConfig round-trips,
  * the serializer enforces the kernel constraints (group_size=8, block_size=32) and
    emits a quantization_config + packed VQLinear buffer keys for a valid artifact.

The full from_pretrained -> VQLinear -> generate path is validated end-to-end on a real
pythia-160m artifact (see the PR); it needs a real model arch so it is not unit-tested here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orka.pipeline.pack import pack_checkpoint


def _write_model(root: Path) -> Path:
    src = root / "model.json"
    src.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.self_attn.q_proj.weight": [
                        [float(i + j) for j in range(8)] for i in range(8)
                    ]
                }
            }
        )
    )
    return src


def _scaffold(root: Path) -> Path:
    """Minimal config dir the serializer copies (no real arch needed for these tests)."""
    d = root / "scaffold"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "gpt_neox"}))
    return d


def _pack(root: Path, group_size: int, block: int) -> Path:
    artifact = root / f"art_g{group_size}.orka"
    pack_checkpoint(
        _write_model(root),
        artifact,
        group_size=group_size,
        codebook_size=4,
        iterations=2,
        codebook_mode="per-tensor",
        sample_vectors=None,
        backend="numpy",
        normalization="block-max",
        block_scale_size=block,
        em_aq_passes=0,
    )
    return artifact


class OrkaQuantizerRegistrationTest(unittest.TestCase):
    def test_registers_idempotently(self):
        from transformers.quantizers.auto import (
            AUTO_QUANTIZATION_CONFIG_MAPPING,
            AUTO_QUANTIZER_MAPPING,
        )

        from orka.integrations.hf_quantizer import register_orka_quantizer

        register_orka_quantizer()
        register_orka_quantizer()  # second call must be a no-op, not raise
        self.assertIn("orka", AUTO_QUANTIZER_MAPPING)
        self.assertIn("orka", AUTO_QUANTIZATION_CONFIG_MAPPING)

    def test_lm_head_and_embeddings_treated_dense(self):
        # tied-embedding models (Qwen/Llama): lm_head must stay dense, else the
        # tied-weight finalization calls get_parameter on a VQLinear property and fails.
        from orka.integrations.hf_quantizer import _is_embedding

        self.assertTrue(_is_embedding("lm_head.weight"))
        self.assertTrue(_is_embedding("model.embed_tokens.weight"))
        self.assertTrue(_is_embedding("gpt_neox.embed_in.weight"))
        self.assertFalse(_is_embedding("model.layers.0.self_attn.q_proj.weight"))

    def test_config_round_trips(self):
        from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

        import orka.integrations.hf_quantizer  # noqa: F401 (ensures registration)

        OrkaConfig = AUTO_QUANTIZATION_CONFIG_MAPPING["orka"]
        cfg = OrkaConfig(modules={"m.0": {"out_features": 8}})
        d = cfg.to_dict()
        self.assertEqual(d["quant_method"], "orka")
        self.assertIn("m.0", d["modules"])


class OrkaSerializerTest(unittest.TestCase):
    def test_rejects_non_kernel_group_size(self):
        from orka.integrations.hf_quantizer import export_orka_hf_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _pack(root, group_size=4, block=32)
            with self.assertRaises(ValueError) as cm:
                export_orka_hf_repo(artifact, _scaffold(root), root / "repo")
            self.assertIn("group_size", str(cm.exception))

    def test_valid_artifact_emits_quant_config_and_packed_keys(self):
        from orka.integrations.hf_quantizer import export_orka_hf_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _pack(root, group_size=8, block=32)
            out = root / "repo"
            summary = export_orka_hf_repo(artifact, _scaffold(root), out)

            self.assertEqual(summary["vq_linear_modules"], 1)
            self.assertTrue((out / "model.safetensors").exists())

            cfg = json.loads((out / "config.json").read_text())
            qc = cfg["quantization_config"]
            self.assertEqual(qc["quant_method"], "orka")
            module = "model.layers.0.self_attn.q_proj"
            self.assertIn(module, qc["modules"])
            meta = qc["modules"][module]
            self.assertEqual(meta["group_size"], 8)
            self.assertEqual(meta["block_size"], 32)

            from safetensors import safe_open

            with safe_open(str(out / "model.safetensors"), "pt") as f:
                keys = set(f.keys())
            # packed buffers present; empty CSR correction excluded
            self.assertIn(f"{module}.indices_0", keys)
            self.assertIn(f"{module}.scales", keys)
            self.assertFalse(any(k.endswith("corr_col") for k in keys))


if __name__ == "__main__":
    unittest.main()
