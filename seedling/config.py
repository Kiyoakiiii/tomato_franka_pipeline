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
    from . import kinematics

    kinematics.apply_tool_limits(cfg)  # the tool's joint ranges (robot.side_tool.joint7_range_deg) for every plan and move
    return cfg


def flange_to_ee(cfg: dict) -> np.ndarray:
    f = cfg["robot"]["flange_to_ee"]
    return make_T(Rotation.from_euler("xyz", f["rpy_deg"], degrees=True).as_matrix(), f["xyz_m"])


def tool_frame(cfg: dict) -> np.ndarray | None:
    """T_FA, flange -> the side tool's arm frame (robot.side_tool): x along the arm, z = the flange's z (out of its face:
    down when the face points down), y = z x x; None without a side tool (the gripper on the flange's axis)"""
    t = cfg["robot"].get("side_tool")
    if not t:
        return None
    th = np.radians(float(t["arm_direction_deg"]))
    d = np.array([np.cos(th), np.sin(th), 0.0])
    fz = np.array([0.0, 0.0, 1.0])
    return make_T(np.column_stack([d, np.cross(fz, d), fz]), np.zeros(3))


def ee_to_contact(cfg: dict) -> np.ndarray:
    """T_EG, E -> the contact frame G (the open fingers' centre; z: the approach, y: the closing axis, x: z x y).
    Gripper on the flange's axis: G as E, moved by robot.ee_to_contact_xyz_m. Side tool (robot.side_tool, 2026-11-16:
    the gripper at the end of a horizontal arm under the flange, its fingers along the arm): G reach_m along the arm and
    below_m out of the flange face, approaching along the arm, its x against the flange's z (up when the face is down)."""
    t = cfg["robot"].get("side_tool")
    if not t:
        return make_T(None, cfg["robot"]["ee_to_contact_xyz_m"])
    from .transforms import inv_T

    R_AG = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # columns: G's x = -z_A, y = y_A, z = x_A
    p_A = [float(t["reach_m"]), float(t.get("lateral_m", 0.0)), float(t["below_m"])]
    return inv_T(flange_to_ee(cfg)) @ tool_frame(cfg) @ make_T(R_AG, p_A)


def snapshot(cfg: dict) -> dict:
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
