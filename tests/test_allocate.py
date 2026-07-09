from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from orka.eval.verify import verify_artifact
from orka.pipeline.pack import pack_checkpoint
from orka.quant.allocate import allocation_tensor_stages, build_allocation


def _write_source(root: Path) -> Path:
    rng = np.random.default_rng(9)
    src = root / "model.json"
    # 'hard' tensor: high-entropy gaussian. 'easy' tensor: near-constant rows
    # that one centroid captures almost exactly.
    # Big enough that k=256 codebooks are not capped by the vector count.
    hard = rng.standard_normal((64, 64)).round(3)
    easy = np.tile(np.linspace(-0.1, 0.1, 64, dtype=np.float64), (64, 1)).round(3)
    src.write_text(
        json.dumps(
            {
                "tensors": {
                    "model.layers.0.self_attn.q_proj.weight": hard.tolist(),
                    "model.layers.0.mlp.up_proj.weight": easy.tolist(),
                }
            }
        )
    )
    return src


class AllocationTest(unittest.TestCase):
    def test_hard_tensor_gets_more_bits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            allocation = build_allocation(
                src,
                target_bpw=1.0,
                candidate_specs=("vq-2", "vq-4", "vq-8"),
                group_size=8,
                sample_vectors=None,
                iterations=4,
                backend="numpy",
            )
            tensors = allocation["tensors"]
            hard = tensors["model.layers.0.self_attn.q_proj.weight"]
            easy = tensors["model.layers.0.mlp.up_proj.weight"]
            self.assertGreater(hard["bits_per_weight"], easy["bits_per_weight"])
            self.assertLessEqual(allocation["achieved_bpw"], 1.0 + 1e-9)

    def test_allocation_map_drives_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            allocation = build_allocation(
                src,
                target_bpw=1.0,
                candidate_specs=("vq-2", "vq-4", "vq-8"),
                group_size=8,
                sample_vectors=None,
                iterations=4,
                backend="numpy",
            )
            tensor_map = allocation_tensor_stages(allocation)
            artifact = root / "alloc.orka"
            manifest = pack_checkpoint(
                src,
                artifact,
                group_size=8,
                codebook_size=4,
                iterations=4,
                codebook_mode="per-tensor",
                backend="numpy",
                em_aq_passes=0,
                tensor_stages_map=tensor_map,
            )
            self.assertTrue(manifest["tensor_allocation"])
            by_name = {t["name"]: t for t in manifest["tensors"]}
            for name, entry in allocation["tensors"].items():
                stages_bits = sum(
                    s["index_bits"] for s in by_name[name]["stages"]
                )
                expected_bits = round(entry["bits_per_weight"] * 8)
                self.assertEqual(stages_bits, expected_bits)
                # allocation pins the uniform group size (no family override)
                self.assertEqual(by_name[name]["group_size"], 8)

            verified = verify_artifact(artifact)
            self.assertEqual(verified["verified_tensors"], 2)
            self.assertLess(verified["max_mse_delta"], 1e-6)

    def test_requires_per_tensor_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _write_source(root)
            with self.assertRaises(ValueError):
                pack_checkpoint(
                    src, root / "x.orka", group_size=8, codebook_size=4,
                    iterations=2, codebook_mode="global", backend="numpy",
                    em_aq_passes=0,
                    tensor_stages_map={"model.layers.0.mlp.up_proj.weight": [4]},
                )


if __name__ == "__main__":
    unittest.main()


class SizeAwareAllocateTest(unittest.TestCase):
    def test_codebook_bytes_math(self):
        from orka.quant import parse_quant_spec
        from orka.quant.allocate import _spec_codebook_bytes
        # vq-12 = K=4096 entries x group 8 x 2 bytes (fp16).
        self.assertEqual(_spec_codebook_bytes(parse_quant_spec("vq-12"), 8, 2), 4096 * 8 * 2)
        # planar (scalar) stages carry no codebook.
        self.assertEqual(_spec_codebook_bytes(parse_quant_spec("rvq-s8-s8"), 8, 2), 0)

    def test_small_tensor_escapes_to_planar_under_size_aware(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from orka.quant.allocate import build_allocation
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.json"
            # one tiny tensor where the VQ codebook tax dwarfs index savings
            small = (rng.standard_normal((64, 64)) * 0.1).tolist()
            src.write_text(json.dumps({"tensors": {"model.layers.0.small.weight": small}}))
            # Scalar 's8' stages pack at group_size=1 (8 bpw/stage), so planar only fits
            # a generous budget; at 8 bpw the per-tensor codebook tax is what tips a tiny
            # tensor from VQ to planar. (At 1.0 bpw planar is inadmissible - the old test
            # only passed because of the 8x planar-bpw under-count, now fixed.)
            base = build_allocation(src, 8.0, candidate_specs=("vq-4", "vq-8", "vq-12"),
                                    sample_vectors=1024, iterations=2, backend="numpy",
                                    device="cpu", size_aware=False)
            aware = build_allocation(src, 8.0, candidate_specs=("vq-4", "vq-8", "vq-12"),
                                     sample_vectors=1024, iterations=2, backend="numpy",
                                     device="cpu", size_aware=True)
            base_spec = base["tensors"]["model.layers.0.small.weight"]["spec"]
            aware_spec = aware["tensors"]["model.layers.0.small.weight"]["spec"]
            self.assertTrue(base_spec.startswith("vq"))          # naive picks VQ
            self.assertTrue(aware_spec.startswith("rvq-s"))      # size-aware escapes to planar


class HessianWeightedAllocateTest(unittest.TestCase):
    def test_weighted_probe_differs_from_unweighted(self):
        # Mechanism: importance weights change the probe distortion (weighted k-means
        # optimizes high-importance vectors). Uniform weights reduce to unweighted MSE;
        # skewed weights do not. (End-to-end spec flips are shown on real tensors -
        # tiny random tensors quantize losslessly, hiding the effect.)
        import numpy as np

        from orka.quant import parse_quant_spec
        from orka.quant.allocate import _probe_spec_distortion
        rng = np.random.default_rng(0)
        V = rng.standard_normal((4096, 8)).astype(np.float32)
        st = parse_quant_spec("vq-8")  # lossy at this size -> non-degenerate distortion
        raw = _probe_spec_distortion(V, st, 4, "numpy", "cpu", 0)
        uni = _probe_spec_distortion(V, st, 4, "numpy", "cpu", 0,
                                     sample_weights=np.ones(4096, dtype=np.float32))
        skew = np.ones(4096, dtype=np.float32)
        skew[:2048] = 100.0
        wtd = _probe_spec_distortion(V, st, 4, "numpy", "cpu", 0, sample_weights=skew)
        self.assertAlmostEqual(raw, uni, places=4)      # uniform == unweighted
        self.assertNotAlmostEqual(raw, wtd, places=4)   # skewed differs

    def test_uniform_importance_matches_unweighted(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from orka.quant.allocate import build_allocation
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.json"
            W = rng.standard_normal((32, 64))
            src.write_text(json.dumps({"tensors": {"model.layers.0.a.weight": W.tolist()}}))
            # all-ones activation energy -> uniform importance -> same plan as unweighted
            acts = {"model.layers.0.a.weight": np.ones((64, 64), dtype=np.float32)}
            common = dict(candidate_specs=("vq-4", "vq-8", "vq-12"), sample_vectors=512,
                          iterations=3, backend="numpy", device="cpu")
            base = build_allocation(src, 1.0, **common)
            weighted = build_allocation(src, 1.0, awq_activations=acts, **common)
            self.assertEqual(base["tensors"]["model.layers.0.a.weight"]["spec"],
                             weighted["tensors"]["model.layers.0.a.weight"]["spec"])
