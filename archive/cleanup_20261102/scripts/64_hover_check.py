#!/usr/bin/env python
"""Gripper centre (TCP) check: put the grasp centre (robot.ee_to_contact) hover_mm above a tray cell centre,
fingers pointing straight down, then the operator measures where the finger-gap centre really is (height above
the tray top, offset from the cell centre). Joint motion through the corrected kinematics; returns on request.

    .venvs/pipeline/bin/python scripts/64_hover_check.py --cell B2 --dry-run
    .venvs/pipeline/bin/python scripts/64_hover_check.py --cell B2 --no-step --stay
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--cell", default="B2")
    ap.add_argument("--hover-mm", type=float, default=30.0, help="grasp centre above the tray top")
    ap.add_argument("--min-clearance-m", type=float, default=0.30, help="camera housing / flange / wrist must stay above this z")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--stay", action="store_true", help="stay at the hover pose (default: return after Enter)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from scipy.spatial.transform import Rotation

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T
    from seedling.wrist import body_points_B, path_clearance

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    c = np.array(grid["cells"][args.cell], float)
    target = np.array([c[0], c[1], grid["z_top"] + args.hover_mm / 1000.0])
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    best = None
    for yaw in np.arange(0, 360, 15):
        R = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])  # z (approach) down
        T_BG = make_T(R, target)
        q = kinematics.solve_ik_near(T_BG @ inv_T(G), q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))  # target: the EE pose
        if q is None:
            continue
        # the grasp centre must really come out at the target (guards the frame chain)
        if np.linalg.norm((kinematics.fk(q, F) @ G)[:3, 3] - target) > 0.002:
            continue
        pts = body_points_B(q, T_FC, cfg)
        low = min((float(p[:, 2].min()), k) for k, p in pts.items() if k != "gripper contact point")
        if low[0] < args.min_clearance_m:
            continue
        z, part = path_clearance(q0, q, T_FC, cfg, n=61)
        step = float(np.degrees(np.abs(q - q0)).max())
        if best is None or step < best[0]:
            best = (step, yaw, q, low, z, part)
    if best is None:
        print("no safe hover pose")
        return 1
    step, yaw, q, low, z, part = best
    print(f"cell {args.cell} centre {np.round(c[:2], 4).tolist()}, tray top z {grid['z_top']:.4f}: grasp centre to {np.round(target, 4).tolist()} "
          f"(fingers down, closing axis yaw {yaw:.0f} deg)")
    print(f"max joint change {step:.0f} deg {np.round(np.degrees(q - q0)).astype(int).tolist()}; lowest body part at the hover pose: {low[1]} z={low[0]:.3f}; "
          f"lowest along the way (incl. the grasp centre): {part} z={z:.3f}")
    if args.dry_run:
        return 0
    r = robot.move_joints(q, dynamics=args.dynamics, description=f"hover {args.hover_mm:.0f} mm above {args.cell}")
    print("at the hover pose" if r.ok else f"motion failed: {r.reason}", np.round(robot.q(), 3).tolist())
    print("measure: height of the finger-gap centre above the tray top, and its offset from the cell centre (towards which side)")
    if not args.stay:
        input("Enter to return ")
        robot.move_joints(q0, dynamics=args.dynamics, description="back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
