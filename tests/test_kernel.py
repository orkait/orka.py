from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernel"


@unittest.skipUnless(shutil.which("gcc") or shutil.which("cc"), "no C compiler")
class OrkaVqKernelTest(unittest.TestCase):
    def test_c_self_test_passes(self) -> None:
        """Build and run the standalone C kernel test (full-chain ordering +
        scalar stage). No artifact or Python decode involved."""
        cc = shutil.which("gcc") or shutil.which("cc")
        binary = KERNEL_DIR / "test_orka_vq_ci"
        try:
            subprocess.run(
                [cc, "-O2", "-std=c99", str(KERNEL_DIR / "orka_vq.c"),
                 str(KERNEL_DIR / "test_orka_vq.c"), "-lm", "-o", str(binary)],
                check=True, capture_output=True, text=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
        finally:
            binary.unlink(missing_ok=True)
        self.assertIn("PASS: full chain ordering correct", result.stdout)
        self.assertIn("PASS: scalar stage correct", result.stdout)


if __name__ == "__main__":
    unittest.main()
