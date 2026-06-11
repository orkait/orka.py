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
from orka.pipeline.sequential import _block_key, _group_tensors_by_block
from orka.verify import verify_artifact


class BlockGroupingTest(unittest.TestCase):
    def test_block_keys(self) -> None:
        self.assertEqual(_block_key("model.embed_tokens.weight"), -1)
        self.assertEqual(_block_key("model.layers.0.self_attn.q_proj.weight"), 0)
        self.assertEqual(_block_key("model.layers.11.mlp.up_proj.weight"), 11)
        self.assertEqual(_block_key("lm_head.weight"), 1 << 30)

    def test_grouping_is_forward_ordered(self) -> None:
        names = [
            "lm_head.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        blocks = _group_tensors_by_block(names)
        self.assertEqual(blocks[0], ["model.embed_tokens.weight"])
        self.assertEqual(blocks[1], ["model.layers.0.self_attn.q_proj.weight"])
        self.assertEqual(blocks[2], ["model.layers.1.mlp.up_proj.weight"])
        self.assertEqual(blocks[3], ["lm_head.weight"])


class OnlyTensorsPassthroughTest(unittest.TestCase):
    def _pack(self, root: Path, passthrough: bool) -> Path:
        rng = np.random.default_rng(2)
        src = root / "model.json"
        src.write_text(
            json.dumps(
                {
                    "tensors": {
                        "model.layers.0.mlp.up_proj.weight": rng.standard_normal((4, 8)).tolist(),
                        "model.layers.1.mlp.up_proj.weight": rng.standard_normal((4, 8)).tolist(),
                        "model.norm.weight": rng.standard_normal(8).tolist(),
                    }
                }
            )
        )
        artifact = root / f"pt-{passthrough}.orka"
        pack_checkpoint(
            src, artifact, group_size=4, codebook_size=4, iterations=2,
            codebook_mode="per-tensor", backend="numpy", em_aq_passes=0,
            only_tensors=["model.layers.0.mlp.up_proj.weight"],
            only_tensors_passthrough=passthrough,
        )
        return artifact

    def _passthrough_names(self, artifact: Path) -> set:
        from safetensors import safe_open

        path = artifact / "passthrough.safetensors"
        if not path.exists():
            return set()
        with safe_open(str(path), framework="np") as handle:
            return set(handle.keys())

    def test_skip_mode_keeps_norms_drops_unlisted_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._pack(Path(tmp), passthrough=False)
            manifest = json.loads((artifact / "manifest.json").read_text())
            self.assertEqual(
                [t["name"] for t in manifest["tensors"]],
                ["model.layers.0.mlp.up_proj.weight"],
            )
            names = self._passthrough_names(artifact)
            self.assertIn("model.norm.weight", names)
            self.assertNotIn("model.layers.1.mlp.up_proj.weight", names)

    def test_default_mode_passthroughs_unlisted_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._pack(Path(tmp), passthrough=True)
            names = self._passthrough_names(artifact)
            self.assertIn("model.norm.weight", names)
            self.assertIn("model.layers.1.mlp.up_proj.weight", names)


def _build_tiny_llama(root: Path) -> Path:
    """Offline tiny LlamaForCausalLM + word-level tokenizer."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    model_dir = root / "tiny-llama"
    model_dir.mkdir()
    words = ["<unk>", "hello", "world", "the", "quick", "brown", "fox", "jumps",
             "over", "lazy", "dog", "a", "b", "c", "d", "e"]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    fast.save_pretrained(str(model_dir))

    config = LlamaConfig(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return model_dir


@unittest.skipUnless(HAS_HF, "torch + transformers required")
class SequentialPackIntegrationTest(unittest.TestCase):
    def test_sequential_pack_round_trips(self) -> None:
        from orka.pipeline.sequential import pack_checkpoint_sequential

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = _build_tiny_llama(root)
            source = next(model_dir.glob("*.safetensors"))
            prompts = root / "prompts.txt"
            prompts.write_text(
                "the quick brown fox jumps over the lazy dog\n"
                "hello world a b c\n"
                "the lazy dog jumps\n"
            )

            out = root / "seq.orka"
            manifest = pack_checkpoint_sequential(
                source=source,
                out_dir=out,
                model_dir=model_dir,
                prompts_path=prompts,
                model_device="cpu",
                calibration_max_prompts=3,
                calibration_max_length=32,
                calibration_max_samples=256,
                group_size=8,
                codebook_size=16,
                iterations=2,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
            )
            self.assertTrue(manifest["sequential_calibration"])
            self.assertGreater(manifest["tensor_count"], 0)
            self.assertTrue(manifest["hessian_weighted"])

            # No leftover part dirs
            self.assertEqual(list(root.glob("seq.orka.seq-part-*")), [])

            verified = verify_artifact(out)
            self.assertEqual(verified["verified_tensors"], manifest["tensor_count"])
            self.assertLess(verified["max_mse_delta"], 1e-6)


if __name__ == "__main__":
    unittest.main()
