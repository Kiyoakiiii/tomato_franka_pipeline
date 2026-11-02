"""Shared CLI bootstrap for scripts/: project import path, caches inside the project, config args."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the tree free of __pycache__
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ZED NEURAL depth needs TensorRT 10, installed only in the project env (tensorrt-cu12-libs + unversioned
# symlinks in .venvs/pipeline/trt_links). The SDK dlopens it by name, so the search path must be set before
# the process starts: re-exec once with LD_LIBRARY_PATH extended (this process only).
_TRT_LINKS = PROJECT_DIR / ".venvs" / "pipeline" / "trt_links"
if _TRT_LINKS.is_dir() and str(_TRT_LINKS) not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(_TRT_LINKS), os.environ.get("LD_LIBRARY_PATH", "")]))
    os.execv(sys.executable, [sys.executable] + sys.argv)

# shared machine: cap math-library threads unless the caller sets them explicitly
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

# keep library caches/config inside the project (shared machine)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".config" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))

from seedling.config import load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default configs/pipeline.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override config, e.g. --set gripper.grasp_force_n=15")
    return p


def config_from_args(args) -> dict:
    return load_config(args.config, args.set)
