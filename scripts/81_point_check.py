#!/usr/bin/env python
"""WARNING (2026-11-06): do NOT keep the pin-brake gripper closed for this: closed on nothing for ~20 min its motor nearly
burned out. Use it open (or a pointer the user holds) and keep any close to under a minute.

Point the closed fingers at a disc hole centre or a tray cell centre (no plant held), so one can see how far the
hand's closing point lands from where the program aims: the closing point (where the pins meet) is put above the
target, the hand flat with the heading the program uses there, then the script stops and the arm stays put.

    .venvs/pipeline/bin/python scripts/81_point_check.py H2 --dry-run
    .venvs/pipeline/bin/python scripts/81_point_check.py H2                 # above hole 2, 25 mm over the disc top
    .venvs/pipeline/bin/python scripts/81_point_check.py C9 --height-mm 40   # above cell C9, 40 mm over the tray top
    .venvs/pipeline/bin/python scripts/81_point_check.py back                # straight up, then to the overview

Uses calibration/tray_grid.json and calibration/disc.json as the last locate left them (the run locates first).
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("target", help="centre (the disc centre), H1..H6 (a disc hole), a cell name (e.g. C9), or 'back'")
    ap.add_argument("--height-mm", type=float, default=80.0, help="closing point this far above the disc top / tray top")
    ap.add_argument("--heading-deg", type=float, default=30.0, help="at a hole: hand heading from straight out (as the program)")
    ap.add_argument("--down", action="store_true", help="the hand pointing straight down, the gripper OPEN: the open gap "
                    "centre over the target (its xy does not depend on the finger length); --height-mm is then the fingertips' height")
    ap.add_argument("--wrist-deg", type=float, default=0.0, help="with --down: the hand turned this much about the vertical")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import cartesian, kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.planner import lift_then_plan
    from seedling.program import flat_R
    from seedling.robot import FrankyRobot
    from seedling.transforms import make_T
    from seedling.world import World

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    from seedling.disc import taken_holes

    world = World(cfg, T_FC, grid, disc, tray_plants=list(grid.get("planted") or grid.get("occupied", [])),
                  taken_holes=taken_holes(disc))
    robot = FrankyRobot(cfg, step_mode=False)
    q0 = robot.q()
    fwd = float(cfg["gripper"]["pin_brake"].get("close_forward_m", 0.0))

    def G_of(q):
        return kinematics.fk(np.asarray(q, float), F) @ G

    def run(steps):
        for name, qa, qb in steps:
            m, part = world.worst(qa, qb)
            print(f"{name}: max joint change {np.degrees(np.abs(qb - qa)).max():.0f} deg, smallest margin {m * 1000:+.0f} mm ({part})")
        if args.dry_run:
            return 0
        for name, qa, qb in steps:
            r = robot.move_joints(qb, dynamics="fine" if name.startswith("down") else None, description=name)
            print(name, "ok" if r.ok else "FAILED " + r.reason)
            if not r.ok:
                return 1
        return 0

    if args.target == "back":
        q_ov = np.array(cfg["tray"]["overview_q"], float)
        steps, msg = lift_then_plan(q0, q_ov, world, cfg, lifts_m=(0.05, 0.10, 0.15), time_s=10.0)
        if steps is None:
            print("back: NOT SAFE:", msg)
            return 1
        return run(steps)

    t = args.target.upper()
    if t in ("CENTRE", "CENTER"):
        xy = np.array(disc["centre"][:2], float)
        surface = float(disc["z_top"])
        yaw = float(np.arctan2(xy[1], xy[0]) + np.radians(args.heading_deg))
        what = f"disc centre {np.round(xy, 4).tolist()}, disc top {surface:.3f}"
    elif t.startswith("H") and t[1:].isdigit():
        h = int(t[1:])
        xy = np.array(disc["holes"][h - 1][:2], float)
        surface = float(disc["z_top"])
        yaw = float(np.arctan2(xy[1], xy[0]) + np.radians(args.heading_deg))
        what = f"hole H{h} centre {np.round(xy, 4).tolist()}, disc top {surface:.3f}"
    elif t in grid["cells"]:
        xy = np.array(grid["cells"][t][:2], float)
        surface = float(grid["z_top"])
        cells = np.array([v[:2] for v in grid["cells"].values()], float)
        ctr = cells.mean(0)
        yaw = float(np.arctan2(ctr[1], ctr[0]) + np.radians(15.0))  # the program's approach direction
        what = f"cell {t} centre {np.round(grid['cells'][t][:2], 4).tolist()}, tray top {surface:.3f}"
    else:
        print("target: H1..H6, a cell name, or back")
        return 1
    if args.down:
        tip = float(cfg["gripper"].get("fingertip_ahead_of_contact_m", 0.013))
        R = (kinematics.fk(np.array(cfg["robot"]["home_joints_rad"], float), F) @ G)[:3, :3]  # pointing down, as at home
        from scipy.spatial.transform import Rotation as _Rot

        R = _Rot.from_euler("z", args.wrist_deg, degrees=True).as_matrix() @ R
        a = R[:, 2]
        fwd = 0.0
        close = np.array([xy[0], xy[1], surface + args.height_mm / 1000.0 + tip])  # the open gap centre; fingertips tip lower
    else:
        R, a = flat_R(yaw)
        close = np.array([xy[0], xy[1], surface + args.height_mm / 1000.0])
    if not args.down:
        fwd = 0.029  # (measured 2026-11-06) closed on nothing the pins turn fully and meet 27 mm ahead of the open gap centre (the pointer)
    T_at = make_T(R, close - a * fwd)
    T_above = make_T(R, T_at[:3, 3] + [0, 0, 0.08])
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    from seedling.transforms import inv_T

    # seeds: where the arm is, the overview, and the joints the last runs used near the disc and over the tray
    seeds = [q0, np.array(cfg["tray"]["overview_q"], float)]
    for f in sorted((PROJECT_DIR / "experiments" / "grasp").glob("*_p*_plan.json")) + \
            sorted((PROJECT_DIR / "experiments" / "grasp").glob("*_side_plan.json")):
        try:
            qd = json.loads(f.read_text()).get("q")
        except Exception:  # noqa: BLE001
            continue
        for v in (qd.values() if isinstance(qd, dict) else [qd]):
            if isinstance(v, list) and len(v) == 7 and all(isinstance(x, (int, float)) for x in v):
                seeds.append(np.array(v, float))
    q_above, why = None, "no joint solution"
    for s_ in sorted(seeds, key=lambda s: np.linalg.norm(G_of(s)[:2, 3] - T_above[:2, 3])):
        q_ = kinematics.solve_ik_near(T_above @ inv_T(G), s_, F, margin)
        if q_ is None or np.linalg.norm(G_of(q_)[:3, 3] - T_above[:3, 3]) > 0.0015:
            continue
        m_, part_ = world.worst(q_, q_)
        if m_ <= 0:
            why = f"in collision ({part_}, {m_ * 1000:+.0f} mm)"
            continue
        q_above = q_
        break
    if q_above is None:
        print("above the target:", why)
        print("no joint solution above the target")
        return 1
    steps, msg = lift_then_plan(q0, q_above, world, cfg, time_s=10.0)
    if steps is None:
        print("to above the target: NOT SAFE:", msg)
        return 1
    pcs, why = cartesian.chain(q_above, [("down", T_at)], F, G, margin, max_tilt=None if args.down else 1.0)
    if pcs is None:
        print("no straight way down to the target:", why)
        return 1
    wps = list(pcs[0][2])
    q_at = wps[-1]
    if np.linalg.norm(G_of(q_at)[:3, 3] - T_at[:3, 3]) > 0.0015:
        print("no joint solution at the target")
        return 1
    prev = q_above
    for i_, w in enumerate(wps):
        steps.append((f"down to the target ({i_ + 1}/{len(wps)})", prev, w))
        prev = w
    print(f"target: {what}; the pointer (where the pins meet, closed on nothing) aimed at {np.round(close, 4).tolist()}, "
          f"hand heading {np.degrees(yaw):.0f} deg")
    return run(steps)


if __name__ == "__main__":
    raise SystemExit(main())
