#!/usr/bin/env python
"""Locate the tray from one picture at the current robot pose (no motion): every cell centre in the base frame.
The cell layout is config tray.model; the result goes to tray.grid_file (and an overlay into experiments/tray/).
The whole tray must be in the picture. The work itself is seedling/overview.py (the offline replay runs it too).

    .venvs/pipeline/bin/python scripts/locate.py
    .venvs/pipeline/bin/python scripts/locate.py --frame experiments/tray/<id>/frame.npz --no-save

With --frame the camera pose is the one saved with that picture (its tray_grid.json: the calibration of that day;
the camera was remounted on 2026-11-06, the current calibration puts older pictures 15-20 mm off); --pose current
computes it from the frame's joint angles with today's calibration instead.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
from pathlib import Path

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--q", default=None, help="with --frame: joint angles (JSON) if the frame has none")
    ap.add_argument("--pose", choices=["auto", "saved", "current"], default="auto",
                    help="with --frame: the camera pose saved with the picture (saved), from its joint angles and today's "
                         "calibration (current), or saved when there is one (auto)")
    ap.add_argument("--no-save", action="store_true", help="do not overwrite tray.grid_file")
    ap.add_argument("--method", choices=["auto", "grid", "outline"], default="auto",
                    help="grid: match the cell walls (needs some empty cells in view); outline: the tray's outline + rows x "
                         "cols x pitch from tray.model (works when every cell holds a plant); auto: grid, else outline")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC, now_id
    from seedling.camera import Frame
    from seedling.overview import locate_overview
    from seedling.traygrid import draw_cells

    T_FC, meta = load_T_FC(cfg)
    calib_id = meta.get("calibration_id")
    T_BC = None
    if args.frame:
        frame = Frame.load(args.frame)
        q = np.array(json.loads(args.q)) if args.q else np.array(frame.meta["q"])
        saved = Path(args.frame).with_suffix("").parent / "tray_grid.json"
        if args.pose != "current" and not args.q and saved.exists():
            s = json.loads(saved.read_text())
            if "T_BC" in s:
                T_BC, calib_id = np.array(s["T_BC"]), s.get("calibration")
                print(f"camera pose: saved with the picture (calibration {calib_id})")
        if T_BC is None and args.pose == "saved":
            print(f"no saved camera pose next to {args.frame}")
            return 1
    else:
        from seedling.camera import ZedCamera
        from seedling.robot import FrankyRobot

        robot = FrankyRobot(cfg, step_mode=True)
        cam = ZedCamera(cfg)
        try:
            frame = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        finally:
            cam.close()
        q = robot.q()
        frame.meta["q"] = q.tolist()
    if T_BC is None:
        T_BC = kinematics.fk_flange(q) @ T_FC
    # the tray top found last time, as a hint (searched near it first; the full search if no clear fit there)
    gf = PROJECT_DIR / cfg["tray"]["grid_file"]
    try:
        last = json.loads(gf.read_text())
        z_hint = float(last["z_top"]) if last.get("ok") else None
    except Exception:  # noqa: BLE001
        z_hint = None
    # the previous disc position as a prior: used only when too few clean hole rims are left (a filling disc)
    dp = PROJECT_DIR / "calibration" / "disc.json"
    prev = json.loads(dp.read_text()) if dp.exists() else None

    res = locate_overview(frame, T_BC, cfg, z_hint=z_hint, prev_disc=prev, method=args.method)
    g, d = res["grid"], res["disc"]
    rec = {**res["rec"], "q": q.tolist(), "T_BC": T_BC.tolist(), "K": frame.intr.K().tolist(), "calibration": calib_id}

    out = PROJECT_DIR / "experiments" / "tray" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    if not args.frame:
        from seedling.camera import save_in_background

        save_in_background([(frame, out / "frame", False)])  # a record only: written without waiting
    img = draw_cells(frame, T_BC, g.cells) if g.cells else cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if d.get("centre") is not None:
        from seedling.transforms import apply_T, inv_T

        K = frame.intr.K()
        for i, (hx, hy) in enumerate(d["holes"], 1):
            ring = np.array([[hx + 0.02475 * np.cos(t), hy + 0.02475 * np.sin(t), d["z_top"]] for t in np.linspace(0, 2 * np.pi, 40)])
            pc = apply_T(inv_T(T_BC), ring)
            uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.int32)
            cv2.polylines(img, [uv], True, (255, 255, 0) if d["ok"] else (0, 0, 255), 2)
            c0 = uv.mean(0).astype(int)
            cv2.putText(img, f"H{i}", (int(c0[0]) - 12, int(c0[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if g.cells or d.get("centre") is not None:
        cv2.imwrite(str(out / "cells.jpg"), img)

    (out / "tray_grid.json").write_text(json.dumps(rec, indent=1))
    if g.ok and not args.no_save:
        gf.write_text(json.dumps(rec, indent=1))
        print("saved", gf.relative_to(PROJECT_DIR))
    if d["ok"] and not args.no_save:
        dp.write_text(json.dumps({**d, "q": q.tolist()}, indent=1))
        print("saved", dp.relative_to(PROJECT_DIR))
    if not d["ok"] and not args.no_save:
        # mark the saved disc as not found in this picture (the transplant loop must not use an old free-hole list); the
        # position is kept as the prior for the next picture
        if prev and prev.get("ok"):
            dp.write_text(json.dumps({**prev, "ok_last_picture": False, "last_status": d["status"]}, indent=1))
    if (out / "cells.jpg").exists():
        print("overlay", (out / "cells.jpg").relative_to(PROJECT_DIR))
    return 0 if g.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
