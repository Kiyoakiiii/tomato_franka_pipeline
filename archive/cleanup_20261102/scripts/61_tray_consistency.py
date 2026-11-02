#!/usr/bin/env python
"""End-to-end check of the wrist-camera calibration in the working area: the tray (fixed, cell layout known) is
located from several robot poses that differ in wrist roll, viewing angle and position. With a correct
calibration every pose gives the same 72 cell centres; their spread is the error the picking will see.
Only poses that show the whole tray are used. Returns to the start pose at the end.

    .venvs/pipeline/bin/python scripts/61_tray_consistency.py --dry-run
    .venvs/pipeline/bin/python scripts/61_tray_consistency.py --no-step
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import time

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--distance", type=float, nargs=2, default=[0.60, 0.75])
    ap.add_argument("--elevation", type=float, nargs=2, default=[55, 80])
    ap.add_argument("--azimuth", type=float, nargs=2, default=[-25, 25])
    ap.add_argument("--rolls", type=float, nargs="+", default=[-60, -30, 0, 30, 60])
    ap.add_argument("--margin-px", type=float, default=40)
    ap.add_argument("--close", type=int, default=0, help="also this many closer views of part of the tray (located near the "
                    "saved grid), so that a mounting fit sees the working distance too")
    ap.add_argument("--close-distance", type=float, nargs=2, default=[0.38, 0.46])
    ap.add_argument("--max-joint-step-deg", type=float, default=45.0)
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start-q", default=None, help="with --dry-run: plan from these joints (JSON) instead of the current ones")
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--camera-always-on", dest="camera_per_view", action="store_false",
                    help="keep the camera streaming during the moves (default: open it only at each pose)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC, now_id
    from seedling.config import flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import apply_T, inv_T
    from seedling.traygrid import draw_cells, locate_tray
    from seedling.wrist import look_from, lowest_body_z, path_clearance

    T_FC, meta = load_T_FC(cfg)
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    cells = {k: np.array(v) for k, v in grid["cells"].items()}
    P = np.array(list(cells.values()))
    centre = P.mean(0)
    half = 0.5 * float(cfg["tray"]["model"]["pitch_m"]) + 0.01
    corners = [P[np.argmin(P[:, 0] + P[:, 1])], P[np.argmin(P[:, 0] - P[:, 1])], P[np.argmax(P[:, 0] + P[:, 1])], P[np.argmax(P[:, 0] - P[:, 1])]]
    outline = [c + [sx * half, sy * half, 0] for c in corners for sx in (-1, 1) for sy in (-1, 1)]
    F = flange_to_ee(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = np.array(json.loads(args.start_q)) if (args.dry_run and args.start_q) else robot.q()
    floor = float(centre[2]) + float(cfg["tray"]["observe"]["min_clearance_m"])

    # image size / intrinsics from the last saved frame of the grid (same camera mode)
    W, H = 1920, 1080
    K = np.array(grid.get("K")) if grid.get("K") else None

    rng = np.random.default_rng(args.seed)
    radial = np.arctan2(centre[1], centre[0])
    fx = K[0, 0] if K is not None else 1065.0  # pinhole, nominal ZED HD1080 intrinsics if none saved
    cx, cy = (K[0, 2], K[1, 2]) if K is not None else (W / 2, H / 2)

    def in_picture(T_BC, pts, margin_px):
        pc = apply_T(inv_T(T_BC), np.array(pts))
        u, v = fx * pc[:, 0] / pc[:, 2] + cx, fx * pc[:, 1] / pc[:, 2] + cy
        return (u > margin_px) & (v > margin_px) & (u < W - margin_px) & (v < H - margin_px) & (pc[:, 2] > 0.1)

    def candidates(n, dist, close):
        out = []
        for _ in range(60 * n):
            el, d = rng.uniform(*args.elevation), rng.uniform(*dist)
            az = radial + np.radians(rng.uniform(*args.azimuth))
            roll = float(rng.choice(args.rolls))
            target = centre if not close else P[rng.integers(len(P))]
            T_BC = look_from(target, az, el, d, roll)
            q = kinematics.solve_ik_near(T_BC @ inv_T(T_FC) @ F, q0, F, margin)
            if q is None or lowest_body_z(q, T_FC, cfg)[0] < floor:
                continue
            if not close and not in_picture(T_BC, outline, args.margin_px).all():
                continue  # the whole tray in the picture
            if close and in_picture(T_BC, P, 60).sum() < 15:
                continue  # enough cells for a located piece of grid
            out.append((q, T_BC, el, d, roll, np.degrees(az - radial), close))
        out.sort(key=lambda c: c[0][6])  # spread over joint 7
        return [out[i] for i in np.unique(np.linspace(0, len(out) - 1, n).round().astype(int))] if out else []

    pick = candidates(args.n, args.distance, False) + (candidates(args.close, args.close_distance, True) if args.close else [])
    pick.sort(key=lambda c: c[0][6])  # sweep in joint-7 order
    plan, kinds, q = [], [], q0
    for qc, T_BC, el, d, roll, az, close in pick:
        qn = kinematics.solve_ik_near(T_BC @ inv_T(T_FC) @ F, q, F, margin)
        qn = qc if qn is None else qn
        dq = np.degrees(np.abs(qn - q))
        z, part = path_clearance(q, qn, T_FC, cfg)
        ok = dq[:6].max() <= args.max_joint_step_deg and dq[6] <= 90 and z >= min(floor, lowest_body_z(q, T_FC, cfg)[0] - 1e-3)
        print(f"  {'ok  ' if ok else 'skip'} {'close' if close else 'whole'} elev {el:4.1f}, azimuth {az:+5.1f}, dist {d:.2f} m, roll {roll:+4.0f} -> joint 6 {qn[5]:+.2f}, "
              f"joint 7 {qn[6]:+.2f} rad; max change joints 1-6 {dq[:6].max():3.0f} deg, joint 7 {dq[6]:3.0f} deg; lowest {part} z={z:.3f}")
        if ok:
            plan.append(qn)
            kinds.append(close)
            q = qn
    if not plan:
        print("no pose")
        return 1
    Q = np.array(plan)
    print(f"\n{len(plan)} poses; joint 6 {Q[:, 5].min():+.2f}..{Q[:, 5].max():+.2f}, joint 7 {Q[:, 6].min():+.2f}..{Q[:, 6].max():+.2f} rad; "
          f"return {np.degrees(np.abs(plan[-1] - q0)).max():.0f} deg")
    if args.dry_run:
        return 0

    from seedling.camera import ZedCamera

    out = PROJECT_DIR / "experiments" / "tray_consistency" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    # the camera streams only while the arm stands still: opened at each pose, closed before the next move (two
    # runs in a row lost the whole USB controller, keyboard included, while streaming during a move)
    cam = None if args.camera_per_view else ZedCamera(cfg)
    res = []
    try:
        for i, qn in enumerate(plan, 1):
            r = robot.move_joints(qn, dynamics=args.dynamics, description=f"tray view {i}")
            if not r.ok:
                print("motion failed:", r.reason)
                break
            time.sleep(1.0)
            if args.camera_per_view:
                c = ZedCamera(cfg)
                try:
                    f = c.grab(n_average=3)
                finally:
                    c.close()
            else:
                f = cam.grab(n_average=3)
            qm = robot.q()
            f.meta["q"] = qm.tolist()
            T_BC = kinematics.fk_flange(qm) @ T_FC
            f.save(out / f"view{i:02d}", save_png=False)
            g = locate_tray(f, T_BC, cfg, prior=grid["cells"] if kinds[i - 1] else None)
            if g.cells:
                cv2.imwrite(str(out / f"view{i:02d}.jpg"), cv2.resize(draw_cells(f, T_BC, g.cells), (960, 540)))
            res.append({"view": i, "close": bool(kinds[i - 1]), "q": qm.tolist(), **g.summary(), "cells": g.cells})
            (out / "views.json").write_text(json.dumps(res, indent=1))
            print(f"  view {i}: joint 7 {qm[6]:+.2f} rad, {g.status}, A1 {g.cells['A1'] if g.cells else None}", flush=True)
    finally:
        if cam is not None:
            cam.close()
        dq = np.degrees(np.abs(robot.q() - q0))
        if dq[:6].max() <= 60:
            robot.move_joints(q0, dynamics=args.dynamics, description="back to the start pose")

    good = [r for r in res if r["ok"]]
    if len(good) >= 2:
        names = list(good[0]["cells"])
        C = np.array([[r["cells"][k][:2] for k in names] for r in good])  # views x cells x 2
        mean = C.mean(0)
        dev = np.linalg.norm(C - mean, axis=2) * 1000  # mm
        print(f"\nCONSISTENCY over {len(good)} views: cell centre deviation from the mean: median {np.median(dev):.2f} mm, "
              f"95% {np.percentile(dev, 95):.2f} mm, max {dev.max():.2f} mm")
        for r, d in zip(good, dev):
            print(f"  view {r['view']}: joint 7 {r['q'][6]:+.2f}, mean dev {d.mean():.2f} mm, max {d.max():.2f} mm")
        (out / "summary.json").write_text(json.dumps({"views": len(good), "median_mm": float(np.median(dev)), "p95_mm": float(np.percentile(dev, 95)),
                                                      "max_mm": float(dev.max()), "calibration": meta.get("calibration_id")}, indent=1))
    print("saved", out.relative_to(PROJECT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
