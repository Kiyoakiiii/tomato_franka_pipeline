"""Text-command jogging in the gripper contact frame, used to teach reference poses.

Commands (mm):  c+2  a-1  s+0.5  x+3 y-2 z+1 (base axes)   p = print pose   done = accept   q = abort
c = closing (G y), a = approach (G z), s = along stem (G x).
"""
from __future__ import annotations

import re

import numpy as np

from .robot import RobotBase

_AX = {"s": ("G", 0), "c": ("G", 1), "a": ("G", 2), "x": ("B", 0), "y": ("B", 1), "z": ("B", 2)}


def jog_session(robot: RobotBase, dynamics: float, max_step_mm: float = 10.0, max_total_mm: float = 60.0, prompt: str = "jog") -> np.ndarray | None:
    start = robot.T_BG()
    step_mode = robot.step_mode
    robot.step_mode = False  # each jog command is itself the operator's confirmation
    try:
        while True:
            cmd = input(f"[{prompt}] c/a/s/x/y/z±mm, p, done, q: ").strip().lower().replace(" ", "")
            if cmd in {"q", "quit"}:
                return None
            if cmd == "done":
                return robot.T_BG()
            if cmd == "p":
                T = robot.T_BG()
                d = (T[:3, 3] - start[:3, 3]) * 1000
                print(f"  G xyz={np.round(T[:3, 3], 4).tolist()} m, moved {np.round(d, 1).tolist()} mm from start")
                continue
            m = re.fullmatch(r"([casxyz])([+-]\d*\.?\d+)", cmd)
            if not m:
                print("  ?")
                continue
            frame, i = _AX[m.group(1)]
            mm = float(m.group(2))
            if abs(mm) > max_step_mm:
                print(f"  step limited to {max_step_mm} mm")
                continue
            T = robot.T_BG()
            delta = (T[:3, i] if frame == "G" else np.eye(3)[i]) * mm / 1000.0
            T[:3, 3] += delta
            if np.linalg.norm(T[:3, 3] - start[:3, 3]) * 1000 > max_total_mm:
                print(f"  refusing: more than {max_total_mm} mm from the jog start")
                continue
            r = robot.move_contact_to(T, dynamics=dynamics, force_limit_n=10.0, description=f"jog {cmd}")
            if not r.ok:
                print(f"  motion failed: {r.reason}")
    finally:
        robot.step_mode = step_mode
