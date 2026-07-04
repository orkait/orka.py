"""Env-driven config for the analysis-engine backend."""
import os
from pathlib import Path

GPU_MEM_CAP_GB = float(os.environ.get("ORKA_UI_GPU_CAP_GB", "10"))   # orka 10GB cap
HF_CACHE = os.environ.get("HF_HOME", str(Path.home() / "ai-models" / "hf-cache"))
LIVE_PARAM_CEILING = int(os.environ.get("ORKA_UI_LIVE_PARAM_CEILING", str(2_000_000_000)))
HF_TOKEN = os.environ.get("HF_TOKEN")          # never logged
SCHEMA_VERSION = 1
