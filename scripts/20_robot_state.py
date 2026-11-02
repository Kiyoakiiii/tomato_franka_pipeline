#!/usr/bin/env python
"""Print FR3 state (pose, joints, external force, errors) and gripper width. No motion.

Error recovery only happens with an explicit --recover.

    .venvs/pipeline/bin/python scripts/20_robot_state.py [--watch] [--recover]
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import time

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--watch", action="store_true", help="print continuously (Ctrl+C to stop)")
    ap.add_argument("--recover", action="store_true", help="call recover_from_errors() once")
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.robot import FrankyRobot
    from seedling import kinematics

    robot = FrankyRobot(cfg, step_mode=True)
    hand = None
    if cfg["gripper"]["type"] == "franka_hand" and not args.no_gripper:
        import franky

        hand = franky.Gripper(str(cfg["robot"]["host"]))
    if args.recover:
        print("recover_from_errors:", robot.recover())
    try:
        while True:
            T = robot.T_BE()
            q = robot.q()
            print(f"EE xyz [m] = {np.round(T[:3, 3], 4).tolist()}   contact G xyz = {np.round(robot.T_BG()[:3, 3], 4).tolist()}")
            print(f"EE R =\n{np.round(T[:3, :3], 3)}")
            print(f"q [rad] = {np.round(q, 3).tolist()}   (margin to limits min {np.min(np.minimum(q - kinematics.Q_MIN, kinematics.Q_MAX - q)):.2f} rad)")
            print(f"F_ext base [N] = {np.round(robot.external_force(), 2).tolist()}   has_errors={robot.has_errors()}")
            if hand is not None:
                print(f"gripper width = {hand.width*1000:.2f} mm  is_grasped={hand.is_grasped}")
            fk_err = np.linalg.norm(kinematics.fk(q, robot.F_T_EE)[:3, 3] - T[:3, 3]) * 1000
            print(f"FK(config flange_to_ee) vs robot EE: {fk_err:.1f} mm")
            if not args.watch:
                break
            print("-" * 60)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
