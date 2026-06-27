from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import transformers  # noqa: F401
    HAS_HF = True
except ImportError:
    HAS_HF = False

from orka.pipeline.pack import pack_checkpoint
from orka.pipeline.decode import _decode_tensor


@unittest.skipUnless(HAS_HF, "torch + transformers required")
class ExportVllmTest(unittest.TestCase):
    def _build_artifact(self, root: Path, rank: int | None):
        from tests.test_sequential import _build_tiny_llama
        from orka.artifact.correct import correct_artifact

        model_dir = _build_tiny_llama(root)
        source = next(model_dir.glob("*.safetensors"))
        artifact = root / "tiny.orka"
        pack_checkpoint(
            source, artifact, group_size=8, codebook_size=64, iterations=4,
            codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            normalization="slrq-block",
        )
        if rank:
            correct_artifact(artifact, rank=rank, device="cpu")
        return model_dir, artifact

    def test_export_with_adapter_math_is_exact(self) -> None:
        from orka.artifact.export import export_vllm
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, artifact = self._build_artifact(root, rank=2)
            out = root / "export"
            result = export_vllm(artifact, out, model_dir=model_dir, dtype="float32")
            self.assertIsNotNone(result["correction_adapter"])

            base = load_file(str(out / "model.safetensors"))
            adapter = load_file(
                str(out / "correction-adapter" / "adapter_model.safetensors")
            )
            manifest = json.loads((artifact / "manifest.json").read_text())
            checked = 0
            for tm in manifest["tensors"]:
                if not tm.get("lowrank"):
                    continue
                name = tm["name"]
                module = name[: -len(".weight")]
                full = torch.from_numpy(
                    np.asarray(_decode_tensor(artifact, tm), dtype=np.float32)
                ).reshape([int(x) for x in tm["shape"]])
                a = adapter[f"base_model.model.{module}.lora_B.weight"].float()
                b = adapter[f"base_model.model.{module}.lora_A.weight"].float()
                reassembled = base[name].float() + a @ b
                torch.testing.assert_close(reassembled, full, rtol=2e-3, atol=2e-3)
                checked += 1
            self.assertGreater(checked, 0)

    def test_exported_model_loads_and_runs(self) -> None:
        from orka.artifact.export import export_vllm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, artifact = self._build_artifact(root, rank=None)
            out = root / "export"
            result = export_vllm(
                artifact, out, model_dir=model_dir, dtype="float32",
            )
            self.assertIsNone(result["correction_adapter"])

            tok = AutoTokenizer.from_pretrained(str(out), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(out), local_files_only=True, dtype=torch.float32
            )
            enc = tok("hello world the quick", return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            self.assertEqual(logits.shape[-1], model.config.vocab_size)
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_merge_correction_mode(self) -> None:
        from orka.artifact.export import export_vllm
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, artifact = self._build_artifact(root, rank=2)
            out = root / "merged"
            result = export_vllm(
                artifact, out, model_dir=model_dir, dtype="float32",
                correction_adapter=False,
            )
            self.assertIsNone(result["correction_adapter"])
            base = load_file(str(out / "model.safetensors"))
            manifest = json.loads((artifact / "manifest.json").read_text())
            tm = next(t for t in manifest["tensors"] if t.get("lowrank"))
            full = torch.from_numpy(
                np.asarray(_decode_tensor(artifact, tm), dtype=np.float32)
            ).reshape([int(x) for x in tm["shape"]])
            torch.testing.assert_close(base[tm["name"]].float(), full, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
