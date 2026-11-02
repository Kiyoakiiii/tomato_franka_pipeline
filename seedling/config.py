"""YAML config loading with project-relative path resolution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .transforms import make_T

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "pipeline.yaml"


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_DIR / p


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load the YAML config; ``overrides`` are ``section.key=value`` strings (value parsed as YAML)."""
    cfg_path = project_path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value: {item}")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    cfg["_path"] = str(cfg_path)
    return cfg


def flange_to_ee(cfg: dict) -> np.ndarray:
    f = cfg["robot"]["flange_to_ee"]
    return make_T(Rotation.from_euler("xyz", f["rpy_deg"], degrees=True).as_matrix(), f["xyz_m"])


def ee_to_contact(cfg: dict) -> np.ndarray:
    """T_EG: contact frame G has the same orientation as E, offset by a translation."""
    return make_T(None, cfg["robot"]["ee_to_contact_xyz_m"])


def snapshot(cfg: dict) -> dict:
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
