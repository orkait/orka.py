"""--keep-head-fp16: keep the vocab-width head/embedding fp16 for tied models.

Tied models share the embedding with the logit projection, so quantizing it at low bpw
explodes perplexity (measured: Qwen2.5-0.5B 3bpw 2.10x incl-head vs 1.55x head-fp16). 'auto'
turns this on only when the model ties word embeddings; untied models (FalconH1) keep
quantizing the head, which is fine there - so the default does not change their artifact.
"""
import json
import tempfile
import unittest
from pathlib import Path

from orka.pipeline.pack import _head_fp16_skip

HEADS = {"lm_head", "model.embed_tokens"}


def _src(tie=None):
    d = Path(tempfile.mkdtemp())
    if tie is not None:
        (d / "config.json").write_text(json.dumps({"tie_word_embeddings": tie, "vocab_size": 100}))
    return d


class KeepHeadFp16Test(unittest.TestCase):
    def test_off_never_skips(self):
        self.assertEqual(_head_fp16_skip("off", _src(True), HEADS), set())

    def test_on_always_skips(self):
        self.assertEqual(_head_fp16_skip("on", _src(False), HEADS), HEADS)

    def test_auto_skips_when_tied(self):
        self.assertEqual(_head_fp16_skip("auto", _src(True), HEADS), HEADS)

    def test_auto_keeps_quantizing_when_untied(self):
        # untied -> head quantizes fine (FalconH1); artifact unchanged
        self.assertEqual(_head_fp16_skip("auto", _src(False), HEADS), set())

    def test_auto_defaults_off_without_config(self):
        # unknown tie status -> do not change behaviour (off), so untied/unknown is preserved
        self.assertEqual(_head_fp16_skip("auto", _src(None), HEADS), set())


if __name__ == "__main__":
    unittest.main()
