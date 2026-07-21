"""Import-graph invariants.

orka.pipeline and orka.eval import each other at module level (pack_manifest ->
eval.metrics, eval.verify -> pipeline.decode). That resolves today only because
orka/eval/__init__.py never reaches pipeline. eval/hf.py:10 records the cost of
learning this the hard way, so freeze it: each entry point must import from a cold
interpreter, in either order.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

ENTRY_POINTS = [
    "orka",
    "orka.cli",
    "orka.core._checkpoint",
    "orka.eval",
    "orka.eval.metrics",
    "orka.eval.verify",
    "orka.pipeline",
    "orka.pipeline.pack",
    "orka.quant",
    "orka.artifact.reconstruct",
]


def _import_in_fresh_interpreter(*modules: str) -> subprocess.CompletedProcess:
    code = "\n".join(f"import {m}" for m in modules)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )


class ImportHygieneTest(unittest.TestCase):
    def test_every_entry_point_imports_cold(self) -> None:
        for module in ENTRY_POINTS:
            with self.subTest(module=module):
                proc = _import_in_fresh_interpreter(module)
                self.assertEqual(
                    proc.returncode, 0, f"import {module} failed:\n{proc.stderr}"
                )

    def test_pipeline_eval_cycle_resolves_in_both_orders(self) -> None:
        for first, second in (
            ("orka.eval", "orka.pipeline"),
            ("orka.pipeline", "orka.eval"),
            ("orka.eval.verify", "orka.pipeline.pack"),
            ("orka.pipeline.pack", "orka.eval.verify"),
        ):
            with self.subTest(order=f"{first} -> {second}"):
                proc = _import_in_fresh_interpreter(first, second)
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"import {first} then {second} failed:\n{proc.stderr}",
                )

    def test_import_orka_starts_no_background_thread(self) -> None:
        proc = _import_in_fresh_interpreter(
            "orka",
            "threading, sys",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import threading, orka, orka._runtime as r;"
                "print(threading.active_count(), r._BG_WRITER.thread is None)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        count, writer_idle = proc.stdout.split()
        self.assertEqual(count, "1", "import orka spawned a thread")
        self.assertEqual(writer_idle, "True", "BackgroundWriter started at import")


if __name__ == "__main__":
    unittest.main()
