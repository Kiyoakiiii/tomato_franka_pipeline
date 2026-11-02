#!/usr/bin/env python
"""New overview pose when the tray is not fully in the picture. The rough tray centre comes from the plants (they
only grow in the tray; on a cut-off tray the grid fit can be whole cells off or even turned), or from the grid
fit if no plants are seen. A top view is planned that keeps a circle of half the tray diagonal plus a margin
around that centre inside the picture, so the whole tray is in view whatever its yaw. The arm goes there (lift
in place first) and the pose is stored in the config (tray.overview_q). Run 60_tray_locate.py afterwards.

    .venvs/pipeline/bin/python scripts/67_overview_for_tray.py --dry-run
    .venvs/pipeline/bin/python scripts/67_overview_for_tray.py
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--frame", default=None, help="use a saved picture (frame.npz with its joints) instead of a new one")
    ap.add_argument("--margin-px", type=float, default=60.0)
    ap.add_argument("--margin-cells", type=float, default=1.5, help="margin (cells) beyond half the tray diagonal")
    ap.add_argument("--min-z", type=float, default=0.30, help="lowest z of any checked part on the way")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--joint-weight", type=float, default=0.2, help="cost per degree of the largest joint change")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.camera import Frame
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import apply_T, inv_T
    from seedling.traygrid import locate_tray, locate_tray_outline
    from seedling.wrist import body_points_B, look_from

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    m = cfg["tray"]["model"]
    rows, cols, pitch = int(m["rows"]), int(m["cols"]), float(m["pitch_m"])
    robot = FrankyRobot(cfg, step_mode=False)
    q0 = robot.q()
    if args.frame:
        f = Frame.load(args.frame)
        q_pic = np.array(f.meta["q"])
    else:
        from seedling.camera import ZedCamera

        cam = ZedCamera(cfg)
        try:
            f = cam.grab(n_average=3)
        finally:
            cam.close()
        q_pic = q0
    T_BC = kinematics.fk_flange(q_pic) @ T_FC
    # rough centre: the plants (they only grow in the tray; the grid fit on a cut-off tray can be off by whole
    # cells or even turned); an empty tray falls back to the grid fit. Planned in view: a circle around it of
    # half the tray diagonal plus a margin, so the whole tray fits whatever its yaw
    from seedling.stemline import _mask

    ok = np.isfinite(f.xyz).all(-1)
    Pd = apply_T(T_BC, f.xyz[ok])
    hz, ez = np.histogram(Pd[:, 2], bins=np.arange(Pd[:, 2].min(), Pd[:, 2].max() + 0.004, 0.004))
    table_z = float(ez[int(np.argmax(hz))] + 0.002)
    green = Pd[_mask(f.rgb, cfg["stemline"])[ok] & (Pd[:, 2] > table_z + 0.05) & (Pd[:, 2] < table_z + 0.40)]
    if len(green) > 500:
        c2 = np.median(green[:, :2], 0)
        for _ in range(3):  # drop far-off green (other plants or objects on the table)
            near = green[np.hypot(*(green[:, :2] - c2).T) < 0.25]
            c2 = np.median(near[:, :2], 0)
        centre = np.array([c2[0], c2[1], float(np.percentile(near[:, 2], 20))])
        src = f"{len(near)} plant points"
    else:
        g = locate_tray(f, T_BC, cfg)
        if not g.cells:
            print("no plants and no tray pattern found:", g.status)
            return 1
        centre = np.array([v for v in g.cells.values()]).mean(0)
        src = "grid fit: " + g.status[:50]
    R = 0.5 * np.hypot(cols * pitch, rows * pitch) + args.margin_cells * pitch
    outline = [centre + np.array([R * np.cos(a), R * np.sin(a), 0.0]) for a in np.radians(np.arange(0, 360, 15))]
    print(f"rough tray centre {np.round(centre, 3).tolist()} ({src}), planned in view: circle of radius {R:.2f} m around it")
    K = f.intr.K()
    Hh, Ww = f.rgb.shape[:2]
    # better, when the outline of the visible part is found (a tray cut off by the picture edge is too short
    # along the cut direction): the sides whose corners lie at or beyond the picture edge are the cut ones; the
    # rectangle is extended there to the full tray size (plus a cell of margin) and planned in view. Its yaw is
    # known, so this view can be lower than the circle one (which needs the whole diagonal across the short
    # picture side)
    o = locate_tray_outline(f, T_BC, rows, cols, cfg, pitch=pitch)
    Lt, Wt = cols * pitch + 0.02, rows * pitch + 0.02  # outer size: cells plus rims
    if o.get("corners") is not None:
        c4 = np.array(o["corners"])[:, :2]
        zr = float(o.get("rim_z") or centre[2])
        pc = apply_T(inv_T(T_BC), np.column_stack([c4, np.full(4, zr)]))
        uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
        at_edge = (uv[:, 0] < 15) | (uv[:, 1] < 15) | (uv[:, 0] > Ww - 15) | (uv[:, 1] > Hh - 15)
        e01, e12 = np.linalg.norm(c4[1] - c4[0]), np.linalg.norm(c4[2] - c4[1])
        ia = (0, 1) if e01 >= e12 else (1, 2)  # one long side
        la, lb = max(e01, e12), min(e01, e12)
        if Wt + 0.045 < la < Lt + 0.045 and 0.5 * Wt < lb < Wt + 0.045:
            u = (c4[ia[1]] - c4[ia[0]]) / la
            vv = np.array([-u[1], u[0]])
            mid = c4.mean(0)
            s = (c4 - mid) @ u
            lo_cut = at_edge[s < 0].any()
            hi_cut = at_edge[s > 0].any()
            ext = Lt - la
            shift = (0.5 * ext if hi_cut and not lo_cut else -0.5 * ext if lo_cut and not hi_cut else 0.0)
            hl = 0.5 * Lt + args.margin_cells * pitch + (0.5 * ext if lo_cut == hi_cut else 0.0)
            hw = 0.5 * Wt + pitch
            c2 = mid + shift * u
            centre = np.array([c2[0], c2[1], zr])
            outline = [centre + np.append(sx * hl * u + sy * hw * vv, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
            print(f"visible outline {la * 1000:.0f} x {lb * 1000:.0f} mm, cut at the {'both/neither' if lo_cut == hi_cut else 'one'} "
                  f"end(s): tray centre {np.round(centre, 3).tolist()}, long side along "
                  f"{np.degrees(np.arctan2(u[1], u[0])):.0f} deg; planned in view: {2 * hl:.2f} x {2 * hw:.2f} m around it")
    radial = np.arctan2(centre[1], centre[0])
    jr = {int(k): v for k, v in (cfg["tray"].get("observe", {}).get("joint_range_rad") or {}).items()}  # calibrated envelope
    ws = cfg["robot"]["workspace_m"]
    best = None
    for el in (88, 84, 80, 75, 70):
        for d in np.arange(0.55, 1.01, 0.05):
            for roll in np.arange(-180, 180, 15):
                T = look_from(centre, radial, el, d, roll)
                pc = apply_T(inv_T(T), np.array(outline))
                uv = (K @ (pc / pc[:, 2:3]).T).T
                if (pc[:, 2] < 0.2).any() or (uv[:, 0] < args.margin_px).any() or (uv[:, 1] < args.margin_px).any() \
                        or (uv[:, 0] > Ww - args.margin_px).any() or (uv[:, 1] > Hh - args.margin_px).any():
                    continue
                q = kinematics.solve_ik_near(T @ inv_T(T_FC) @ F, q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))
                if q is None or any(not (lo_ <= q[j - 1] <= hi_) for j, (lo_, hi_) in jr.items()):
                    continue
                pg = (kinematics.fk(q, F) @ G)[:3, 3]  # the robot refuses contact points outside the workspace
                if any(not (ws[a][0] <= pg[i] <= ws[a][1]) for i, a in enumerate("xyz")):
                    continue
                cost = (90 - el) * 0.5 + d * 10 + float(np.degrees(np.abs(q - q0)).max()) * args.joint_weight
                if best is None or cost < best[0]:
                    best = (cost, el, d, roll, q, T)
    if best is None:
        print("no reachable view shows the whole tray")
        return 1
    _, el, d, roll, q1, T = best

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
        qu = kinematics.solve_ik_near(Tu @ inv_T(G), q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))
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
    print("joint change (deg):", np.round(np.degrees(q1 - q0)).astype(int).tolist(), f"roll {roll}")
    print(f"new overview: camera {np.round(T[:3, 3], 3).tolist()}, looking down {el} deg from {d:.2f} m")
    if args.dry_run or not ok:
        return 0 if ok else 1
    for name, qa, qb in steps:
        r = robot.move_joints(qb, dynamics=args.dynamics, description=name)
        print(name, "ok" if r.ok else "FAILED " + r.reason)
        if not r.ok:
            return 1
    # remember it as the overview pose
    p = PROJECT_DIR / "configs" / "pipeline.yaml"
    txt = p.read_text()
    import re

    txt = re.sub(r"(\n  overview_q: )\[[^\]]*\]", lambda mm: mm.group(1) + "[" + ", ".join(f"{x:.3f}" for x in robot.q()) + "]", txt, count=1)
    p.write_text(txt)
    print("overview pose stored in the config (tray.overview_q)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
