"""CLI entry: parser + dispatch + main()."""

from __future__ import annotations

import sys
from pathlib import Path

from orka.cli.parser import build_parser


def main() -> int:
    """argv -> parse -> cmd_*. Auto-bootstraps Kaggle config when on Kaggle with no args."""
    if Path("/kaggle/working").exists():
        from orka.deploy.kaggle import bootstrap_argv
        bootstrap_argv(sys.argv)
    args = build_parser().parse_args()
    return int(args.func(args))


__all__ = ["build_parser", "main"]
