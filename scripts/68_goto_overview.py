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
    # a flat hand (pitch 0) carries the wrist below the grasp centre: lift until its lowest part is at --up-to
    up_to = args.up_to + max(0.0, float(T[2, 3]) - lowest(q0, q0, 1)[0])
    if T[2, 3] < up_to - 0.01:
        # straight up as high as the arm allows (near the tray's far side the full lift may be out of reach), at
        # least 1 cm (the range was empty when the arm was less than 5 cm below --up-to, e.g. at an observation pose)
        q1 = None
        for zt in np.arange(up_to, T[2, 3] + 0.01, -0.02):
            T1 = T.copy()
            T1[2, 3] = zt
            q1 = kinematics.solve_ik_near(T1 @ inv_T(G), q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))
            if q1 is not None:
                break
        if q1 is None:
            print("cannot lift straight up from here")
            return 1
        if zt < up_to:
            print(f"straight lift only to z={zt:.2f} (not {up_to:.2f}); the move on is checked for height")
        steps.append(("lift in place", q0, q1))
        q = q1
    # if the straight lift was limited, the move on may stay at that height, but never closer than 15 cm to the
    # tray top (the turn above the pre-grasp keeps 12 cm)
    min_z_on = args.min_z
    if steps and zt < up_to:
        import json

        from seedling.config import PROJECT_DIR

        gf = PROJECT_DIR / cfg["tray"]["grid_file"]
        tray_top = float(json.loads(gf.read_text())["z_top"]) if gf.exists() else None
        if tray_top is not None:
            min_z_on = max(min(args.min_z, zt - 0.005), tray_top + 0.15)
    def check(name, qa, qb):
        m, part = lowest(qa, qb)
        floor = min(args.min_z, float(lowest(qa, qa, 1)[0]) - 1e-3) if name == "lift in place" else min_z_on
        sc = path_self_clearance(qa, qb, T_FC, cfg)
        bad = m < floor or sc < 0.090  # hand from the arm itself: see --self-clearance-mm of 65
        return not bad, (f"{name}: max joint change {np.degrees(np.abs(qb - qa)).max():.0f} deg, lowest {part} z={m:.3f}, hand from the arm "
                         f"itself {sc * 1000:.0f} mm" + ("  <-- NOT SAFE" if bad else ""))

    ok, lines = True, []
    for name, qa, qb in steps:
        ok_, ln = check(name, qa, qb)
        ok &= ok_
        lines.append(ln)
    ok_, ln = check("to the overview", q, q_ov)
    if ok_:
        steps.append(("to the overview", q, q_ov))
    else:
        # from a flat hand (joint 6 near 250 deg) the straight turn to the overview swings the hand low: through a via
        # pose instead, the arm joints 1-4 first (lifts the hand) or the wrist joints 5-7 first
        import itertools

        subsets = [(0, 1, 2, 3), (4, 5, 6)] + [c_ for n_ in (1, 2, 3, 4, 5) for c_ in itertools.combinations(range(7), n_)
                                              if c_ not in ((0, 1, 2, 3), (4, 5, 6))]
        vias = []
        for sub in subsets:  # these joints go to their overview angles first, the others after
            qv = q.copy()
            qv[list(sub)] = q_ov[list(sub)]
            vias.append(("joints " + ",".join(str(j + 1) for j in sub) + " first", qv))
        for vname, qv in vias:
            ok1, l1 = check(f"to a via pose ({vname})", q, qv)
            ok2, l2 = check("via pose to the overview", qv, q_ov) if ok1 else (False, "")
            if ok1 and ok2:
                steps += [(f"to a via pose ({vname})", q, qv), ("via pose to the overview", qv, q_ov)]
                ln, ok_ = l1 + "\n" + l2, True
                break
    ok &= ok_
    lines.append(ln)
    print("\n".join(lines))
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
