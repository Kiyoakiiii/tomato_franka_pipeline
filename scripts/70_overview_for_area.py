#!/usr/bin/env python
"""Overview pose that sees a whole work area (tray and transplant disc wherever they are put in it), looking
down at an angle from the robot side (an oblique view covers more than a straight-down one from the same
height). The area is a rectangle in the base frame at the table-object height; every corner (plus a margin)
must be in the picture, and the far edge must still have enough pixels per cell to locate the tray. Checks:
reachable, calibrated joint envelope, workspace, lowest part on the way. The arm goes there (lift in place
first) and the pose is stored in the config (tray.overview_q).

    .venvs/pipeline/bin/python scripts/70_overview_for_area.py --dry-run
    .venvs/pipeline/bin/python scripts/70_overview_for_area.py
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import re

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--area", type=float, nargs=4, default=[0.35, 0.85, -0.45, 0.45], metavar=("X0", "X1", "Y0", "Y1"),
                    help="work area in the base frame (m)")
    ap.add_argument("--area-z", type=float, nargs=2, default=[0.0, 0.20], help="height range of what lies in it (m): a tray "
                    "raised up to 20 cm (2026-11-04: from [0.02, 0.12]; one view covers the whole default area only up to 0.20)")
    ap.add_argument("--elevations", type=float, nargs="+", default=[70, 65, 60, 75, 55, 80],
                    help="view angles down from horizontal, preferred first")
    ap.add_argument("--margin-px", type=float, default=40.0)
    ap.add_argument("--min-px-per-cell", type=float, default=30.0, help="tray pitch in pixels at the far edge, at least")
    ap.add_argument("--min-z", type=float, default=0.30, help="lowest z of any checked part on the way")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import apply_T, inv_T
    from seedling.wrist import body_points_B, look_from

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    pitch = float(cfg["tray"]["model"]["pitch_m"])
    ws = cfg["robot"]["workspace_m"]
    jr = {int(k): v for k, v in (cfg["tray"].get("observe", {}).get("joint_range_rad") or {}).items()}
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    robot = FrankyRobot(cfg, step_mode=False)
    q0 = robot.q()
    # ZED 2i left camera, HD1080 (the intrinsics of the saved pictures)
    K = np.array([[1064.5, 0, 945.9], [0, 1064.5, 536.4], [0, 0, 1.0]])
    Ww, Hh = 1920, 1080

    x0, x1, y0, y1 = args.area
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in args.area_z])
    edge = np.array([[x, y, args.area_z[0]] for x in np.linspace(x0, x1, 6) for y in np.linspace(y0, y1, 9)])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, float(np.mean(args.area_z))])
    print(f"area x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f} m ({x1 - x0:.2f} x {y1 - y0:.2f} m), centre {np.round(centre, 3).tolist()}")

    best = None
    for ie, el in enumerate(args.elevations):
        for aim_dx in (-0.10, -0.05, 0.0, 0.05):  # aim a little nearer or farther than the area centre
            aim = centre + np.array([aim_dx, 0.0, 0.0])
            for d in np.arange(0.45, 1.01, 0.025):
                for roll in (0.0, 180.0):  # image long side across the reach (the area is wider than deep)
                    for rr in (90.0, -90.0, 0.0):
                        T = look_from(aim, 0.0, el, d, roll + rr)
                        pc = apply_T(inv_T(T), corners)
                        if (pc[:, 2] < 0.25).any():
                            continue
                        uv = (K @ (pc / pc[:, 2:3]).T).T
                        if (uv[:, 0] < args.margin_px).any() or (uv[:, 1] < args.margin_px).any() \
                                or (uv[:, 0] > Ww - args.margin_px).any() or (uv[:, 1] > Hh - args.margin_px).any():
                            continue
                        # resolution at the farthest point: one pitch step along the area seen in pixels
                        pe = apply_T(inv_T(T), edge)
                        px_cell = float((K[0, 0] * pitch / pe[:, 2]).min())
                        if px_cell < args.min_px_per_cell:
                            continue
                        q = kinematics.solve_ik_near(T @ inv_T(T_FC) @ F, q0, F, margin)
                        if q is None or any(not (lo <= q[j - 1] <= hi) for j, (lo, hi) in jr.items()):
                            continue
                        pg = (kinematics.fk(q, F) @ G)[:3, 3]
                        if any(not (ws[a][0] <= pg[i] <= ws[a][1]) for i, a in enumerate("xyz")):
                            continue
                        step = float(np.degrees(np.abs(q - q0)).max())
                        cost = ie * 3.0 + step * 0.2 - px_cell * 0.3
                        if best is None or cost < best[0]:
                            best = (cost, el, d, roll + rr, q, T, px_cell, step)
    if best is None:
        print("no reachable view shows the whole area: make it smaller (--area) or allow other elevations")
        return 1
    _, el, d, roll, q1, T, px_cell, step = best

    def lowest(qa, qb, n=61):
        lo, part = np.inf, ""
        for s in np.linspace(0, 1, n):
            for k, pts in body_points_B(qa + s * (qb - qa), T_FC, cfg).items():
                if float(pts[:, 2].min()) < lo:
                    lo, part = float(pts[:, 2].min()), k
        return lo, part

    steps, q = [], q0
    Tg = kinematics.fk(q0, F) @ G
    if Tg[2, 3] < args.min_z + 0.02:
        Tu = Tg.copy()
        Tu[2, 3] = args.min_z + 0.02
        qu = kinematics.solve_ik_near(Tu @ inv_T(G), q0, F, margin)
        if qu is None:
            print("cannot lift in place")
            return 1
        steps.append(("lift in place", q0, qu))
        q = qu
    steps.append(("to the new overview", q, q1))
    ok = True
    for name, qa, qb in steps:
        lo, part = lowest(qa, qb)
        bad = lo < args.min_z and name != "lift in place"
        ok &= not bad
        print(f"{name}: max joint change {np.degrees(np.abs(qb - qa)).max():.0f} deg, lowest {part} z={lo:.3f}" + ("  <-- TOO LOW" if bad else ""))
    print("joint change (deg):", np.round(np.degrees(q1 - q0)).astype(int).tolist())
    print("new overview joints:", "[" + ", ".join(f"{x:.3f}" for x in q1) + "]")
    print(f"new overview: camera {np.round(T[:3, 3], 3).tolist()}, looking down {el:g} deg from {d:.2f} m, "
          f"far edge {px_cell:.0f} px per cell")
    if args.dry_run or not ok:
        return 0 if ok else 1
    for name, qa, qb in steps:
        r = robot.move_joints(qb, dynamics=args.dynamics, description=name)
        print(name, "ok" if r.ok else "FAILED " + r.reason)
        if not r.ok:
            return 1
    p = PROJECT_DIR / "configs" / "pipeline.yaml"
    txt = p.read_text()
    txt = re.sub(r"(\n  overview_q: )\[[^\]]*\]", lambda mm: mm.group(1) + "[" + ", ".join(f"{x:.3f}" for x in robot.q()) + "]", txt, count=1)
    p.write_text(txt)
    print("overview pose stored in the config (tray.overview_q)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
