#!/usr/bin/env python
"""Go to the overview pose (tray.overview_q): first straight up where the arm is (if the grasp centre is below
--up-to), then a joint move; every part is checked along the way (grasp centre above the plants).

    .venvs/pipeline/bin/python scripts/68_goto_overview.py --dry-run
    .venvs/pipeline/bin/python scripts/68_goto_overview.py
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--up-to", type=float, default=0.32, help="lift the grasp centre to at least this z first")
    ap.add_argument("--min-z", type=float, default=0.30, help="lowest allowed z of any checked part after the lift")
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics in the config)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T
    from seedling.wrist import body_points_B, path_self_clearance

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    q_ov = np.array(cfg["tray"]["overview_q"], float)
    robot = FrankyRobot(cfg, step_mode=False)
    q0 = robot.q()

    def lowest(qa, qb, n=61):
        m, part = np.inf, ""
        for s in np.linspace(0, 1, n):
            for k, v in body_points_B(qa + s * (qb - qa), T_FC, cfg).items():
                if float(v[:, 2].min()) < m:
                    m, part = float(v[:, 2].min()), k
        return m, part

    steps = []
    T = kinematics.fk(q0, F) @ G
    q = q0
    if T[2, 3] < args.up_to - 0.01:
        # straight up as high as the arm allows (near the tray's far side the full lift may be out of reach), at
        # least 1 cm (the range was empty when the arm was less than 5 cm below --up-to, e.g. at an observation pose)
        q1 = None
        for zt in np.arange(args.up_to, T[2, 3] + 0.01, -0.02):
            T1 = T.copy()
            T1[2, 3] = zt
            q1 = kinematics.solve_ik_near(T1 @ inv_T(G), q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))
            if q1 is not None:
                break
        if q1 is None:
            print("cannot lift straight up from here")
            return 1
        if zt < args.up_to:
            print(f"straight lift only to z={zt:.2f} (not {args.up_to:.2f}); the move on is checked for height")
        steps.append(("lift in place", q0, q1))
        q = q1
    # if the straight lift was limited, the move on may stay at that height, but never closer than 15 cm to the
    # tray top (the turn above the pre-grasp keeps 12 cm)
    min_z_on = args.min_z
    if steps and zt < args.up_to:
        import json

        from seedling.config import PROJECT_DIR

        gf = PROJECT_DIR / cfg["tray"]["grid_file"]
        tray_top = float(json.loads(gf.read_text())["z_top"]) if gf.exists() else None
        if tray_top is not None:
            min_z_on = max(min(args.min_z, zt - 0.005), tray_top + 0.15)
    steps.append(("to the overview", q, q_ov))
    ok = True
    for name, qa, qb in steps:
        m, part = lowest(qa, qb)
        floor = min(args.min_z, float(lowest(qa, qa, 1)[0]) - 1e-3) if name == "lift in place" else min_z_on
        sc = path_self_clearance(qa, qb, T_FC, cfg)
        bad = m < floor or sc < 0.090  # hand from the arm itself: see --self-clearance-mm of 65
        ok &= not bad
        print(f"{name}: max joint change {np.degrees(np.abs(qb - qa)).max():.0f} deg, lowest {part} z={m:.3f}, hand from the arm itself "
              f"{sc * 1000:.0f} mm" + ("  <-- NOT SAFE" if bad else ""))
    if args.dry_run or not ok:
        return 0 if ok else 1
    for name, qa, qb in steps:
        r = robot.move_joints(qb, dynamics=args.dynamics, description=name)
        print(name, "ok" if r.ok else "FAILED " + r.reason)
        if not r.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
