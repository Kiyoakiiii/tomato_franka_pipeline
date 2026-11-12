#!/usr/bin/env python
"""Small, confirmed robot moves for first hardware checks.

    --home                      joint move to robot.home_joints_rad
    --dx/--dy/--dz MM           translate the EE in the BASE frame (orientation kept)
    --to X Y Z                  move the contact point G to absolute base coordinates (m), orientation kept

Every motion asks for Enter first and is slow (dynamics 0.02) unless --dynamics is given.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--to", type=float, nargs=3, default=None)
    ap.add_argument("--dynamics", type=float, default=None)
    ap.add_argument("--max-step-mm", type=float, default=100.0, help="refuse larger relative moves")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.robot import FrankyRobot

    robot = FrankyRobot(cfg, step_mode=True)
    dyn = args.dynamics if args.dynamics is not None else 0.02
    print("start EE:", np.round(robot.T_BE()[:3, 3], 4).tolist())
    if robot.has_errors():
        print("robot reports errors; run robot_state.py --recover after checking the cause")
        return 1
    if args.home:
        r = robot.move_joints(np.array(cfg["robot"]["home_joints_rad"]), dynamics=dyn, description="go home")
        print(r)
    d = np.array([args.dx, args.dy, args.dz]) / 1000.0
    if np.any(d):
        if np.linalg.norm(d) * 1000 > args.max_step_mm:
            print(f"refusing a {np.linalg.norm(d)*1000:.0f} mm step (> --max-step-mm)")
            return 1
        T = robot.T_BE()
        T[:3, 3] += d
        print(robot.move_to_pose(T, dynamics=dyn, description=f"relative base move {np.round(d*1000, 1).tolist()} mm"))
    if args.to is not None:
        T = robot.T_BG()
        T[:3, 3] = args.to
        print(robot.move_contact_to(T, dynamics=dyn, description=f"contact point -> {args.to}"))
    print("end EE:", np.round(robot.T_BE()[:3, 3], 4).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
