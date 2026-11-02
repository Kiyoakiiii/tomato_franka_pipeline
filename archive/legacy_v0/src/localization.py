from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .paths import resolve_project_path

if TYPE_CHECKING:
    from .vision import StemDetection


@dataclass
class RobotTarget:
    xy_m: tuple[float, float]
    z_m: float
    yaw_deg: float | None
    method: str
    warning: str | None = None

    @property
    def xyz_m(self) -> list[float]:
        return [float(self.xy_m[0]), float(self.xy_m[1]), float(self.z_m)]


def load_planar_calibration(path: str | Path) -> dict | None:
    path = resolve_project_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_homography(u: float, v: float, matrix: list[list[float]]) -> tuple[float, float]:
    h = np.array(matrix, dtype=np.float64)
    p = h @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-12:
        raise ValueError("Homography produced point at infinity")
    return float(p[0] / p[2]), float(p[1] / p[2])


def check_workspace(xyz: list[float], limits: dict) -> str | None:
    names = ["x", "y", "z"]
    for idx, name in enumerate(names):
        lo, hi = limits[name]
        value = xyz[idx]
        if value < lo or value > hi:
            return f"{name}={value:.3f} m is outside workspace [{lo:.3f}, {hi:.3f}]"
    return None


def detection_to_robot_target(config: dict, detection: "StemDetection") -> RobotTarget:
    loc = config["localization"]
    robot_cfg = config["robot"]
    u, v = detection.grasp_pixel
    mode = loc["mode"]
    warning = None

    if mode == "homography":
        calib = load_planar_calibration(loc["calibration_file"])
        if calib is None:
            raise FileNotFoundError(
                f"Localization mode is homography, but calibration file is missing: {loc['calibration_file']}"
            )
        x, y = apply_homography(u, v, calib["pixel_to_robot_homography"])
        method = f"homography:{loc['calibration_file']}"
    elif mode == "manual_scale":
        cx, cy = loc["workspace_center_pixel"]
        rx, ry = loc["workspace_center_robot_xy_m"]
        sx = float(loc["meters_per_pixel_x"])
        sy = float(loc["meters_per_pixel_y"])
        y_sign = float(loc["image_y_positive_to_robot_y_sign"])
        x = float(rx) + (float(u) - float(cx)) * sx
        y = float(ry) + (float(v) - float(cy)) * sy * y_sign
        method = "manual_scale"
        warning = "manual_scale is approximate; calibrate or use manual correction before real grasp"
    else:
        raise ValueError(f"Unsupported localization mode: {mode}")

    yaw_deg = None
    if bool(loc.get("use_vision_yaw", False)):
        yaw_deg = float(detection.gripper_axis_deg_image) + float(loc.get("yaw_offset_deg", 0.0))
        while yaw_deg < -180.0:
            yaw_deg += 360.0
        while yaw_deg >= 180.0:
            yaw_deg -= 360.0

    z = float(loc["fixed_grasp_z_m"])
    target = RobotTarget(xy_m=(x, y), z_m=z, yaw_deg=yaw_deg, method=method, warning=warning)
    workspace_warning = check_workspace(target.xyz_m, robot_cfg["workspace_limits_m"])
    if workspace_warning:
        target.warning = workspace_warning
    return target


def target_status_lines(target: RobotTarget | None) -> list[str]:
    if target is None:
        return ["target: none"]
    yaw = "fixed" if target.yaw_deg is None else f"{target.yaw_deg:.1f} deg"
    lines = [
        f"robot target xyz: {target.xyz_m[0]:.3f}, {target.xyz_m[1]:.3f}, {target.xyz_m[2]:.3f} m",
        f"localization: {target.method} | yaw: {yaw}",
    ]
    if target.warning:
        lines.append(f"WARNING: {target.warning}")
    return lines
