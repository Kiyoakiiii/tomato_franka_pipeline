#!/usr/bin/env python
"""Locate the tray from one picture at the current robot pose (no motion): every cell centre in the base frame.
The cell layout is config tray.model; the result goes to tray.grid_file (and an overlay into experiments/tray/).
The whole tray must be in the picture.

    .venvs/pipeline/bin/python scripts/60_tray_locate.py
    .venvs/pipeline/bin/python scripts/60_tray_locate.py --frame experiments/stemline/<id>/frame.npz --q '[...]'
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--q", default=None, help="with --frame: joint angles (JSON) if the frame has none")
    ap.add_argument("--no-save", action="store_true", help="do not overwrite tray.grid_file")
    ap.add_argument("--method", choices=["auto", "grid", "outline"], default="auto",
                    help="grid: match the cell walls (needs some empty cells in view); outline: the tray's outline + rows x "
                         "cols x pitch from tray.model (works when every cell holds a plant); auto: grid, else outline")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC, now_id
    from seedling.camera import Frame
    from seedling.traygrid import TrayGrid, draw_cells, locate_tray, locate_tray_outline, occupied_cells

    T_FC, meta = load_T_FC(cfg)
    if args.frame:
        frame = Frame.load(args.frame)
        q = np.array(json.loads(args.q)) if args.q else np.array(frame.meta["q"])
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
    T_BC = kinematics.fk_flange(q) @ T_FC
    m = cfg["tray"]["model"]
    # the tray top found last time, as a hint (searched near it first; the full search if no clear fit there)
    gf = PROJECT_DIR / cfg["tray"]["grid_file"]
    try:
        last = json.loads(gf.read_text())
        z_hint = float(last["z_top"]) if last.get("ok") else None
    except Exception:  # noqa: BLE001
        z_hint = None
    g = locate_tray(frame, T_BC, cfg, z_hint=z_hint) if args.method in ("auto", "grid") else None
    if g is not None:
        print("grid:", json.dumps(g.summary()))
    if g is not None and g.ok and args.method == "auto":
        # cross-check: the outline (+ known pitch) is good to a few mm and cannot slip by whole cells; a grid
        # fit that disagrees by more than a third of a cell has locked onto the wrong cells (2026-10-22: one column)
        o = locate_tray_outline(frame, T_BC, int(m["rows"]), int(m["cols"]), cfg, pitch=float(m["pitch_m"]), top_z=g.z_top)
        if o["ok"]:
            d = np.array([np.linalg.norm(np.array(g.cells[k][:2]) - np.array(o["cells"][k][:2])) for k in g.cells])
            print(f"cross-check grid vs outline: median {np.median(d) * 1000:.1f} mm, max {d.max() * 1000:.1f} mm")
            pm = float(m["pitch_m"])
            if np.median(d) > pm / 3:
                # the grid is good to a mm but can slip by whole cells (it is periodic); the outline is good to ~1 cm and
                # cannot: the grid moved by the whole number of cells that brings it onto the outline
                vr = (np.array(g.cells["B1"][:2]) - np.array(g.cells["A1"][:2]))  # one row
                vc = (np.array(g.cells["A2"][:2]) - np.array(g.cells["A1"][:2]))  # one column
                off = np.mean([np.array(o["cells"][k][:2]) - np.array(g.cells[k][:2]) for k in g.cells], axis=0)
                nr, nc = np.round(np.linalg.solve(np.column_stack([vr, vc]), off)).astype(int)
                moved = {k: [round(v[0] + nr * vr[0] + nc * vc[0], 4), round(v[1] + nr * vr[1] + nc * vc[1], 4), v[2]] for k, v in g.cells.items()}
                d2 = np.array([np.linalg.norm(np.array(moved[k][:2]) - np.array(o["cells"][k][:2])) for k in moved])
                if (nr or nc) and np.median(d2) <= pm / 3:
                    g.cells = moved
                    g.status = f"ok (grid moved by {nr:+d} rows, {nc:+d} columns onto the tray outline)"
                    print(f"CORRECTED: the grid fit was {abs(nr)} row(s), {abs(nc)} column(s) off; moved onto the outline "
                          f"(now median {np.median(d2) * 1000:.1f} mm from it)")
                    d = d2
            if np.median(d) > pm / 3:
                g.ok, g.status = False, (f"grid fit and tray outline disagree by {np.median(d) * 1000:.0f} mm (about "
                                         f"{np.median(d) / float(m['pitch_m']):.1f} cells): grid locked onto the wrong cells")
                print("REJECTED:", g.status)
        else:
            print("cross-check not possible:", o["status"])
    if g is None or (not g.ok and args.method == "auto"):  # outline as the fallback (printed as such)
        o = locate_tray_outline(frame, T_BC, int(m["rows"]), int(m["cols"]), cfg, pitch=float(m["pitch_m"]))
        print("outline:", json.dumps({k: v for k, v in o.items() if k != "cells"}, default=float))
        g = TrayGrid(o["ok"], "outline: " + o["status"], o.get("cells"), o.get("rim_z"), o.get("yaw_deg"), 0.0, 0.0,
                     {"method": "outline", "corners": o.get("corners")})
    out = PROJECT_DIR / "experiments" / "tray" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    if not args.frame:
        from seedling.camera import save_in_background

        save_in_background([(frame, out / "frame", False)])  # a record only: written without waiting
    if g.cells:
        cv2.imwrite(str(out / "cells.jpg"), draw_cells(frame, T_BC, g.cells))
    occ = occupied_cells(frame, T_BC, g.cells, g.z_top, cfg) if g.ok else {}
    soil = {}
    if g.ok:
        print(f"{len(occ)} cells with a plug: {sorted(occ)}")
        from seedling.plugsense import measure_plug

        for k in sorted(occ):  # each plug's own soil surface (it may stand above, level with or below the tray top)
            r = measure_plug(frame, T_BC, g.cells[k], cfg)
            soil[k] = r["soil_z"] if r["ok"] else None
        print("soil surface (overview), mm vs the tray top:",
              {k: (None if v is None else round((v - g.z_top) * 1000)) for k, v in soil.items()})
    # cells with a plant over them (green within 15 mm of the centre, seen from the overview): the peat test misses
    # plugs hidden under leaves; the grasp planner keeps its path clear of these too
    planted, green_cells = set(occ), set()
    if g.ok:
        from seedling.stemline import _mask
        from seedling.transforms import apply_T as _apply, inv_T as _inv

        green = _mask(frame.rgb, cfg["stemline"])
        Kc = frame.intr.K()
        for k, (x, y, zc) in g.cells.items():
            ring = np.array([[x + rr * np.cos(t), y + rr * np.sin(t), zc + 0.02] for rr in (0.0, 0.007, 0.014) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
            pc = _apply(_inv(T_BC), ring)
            uv = np.round((Kc @ (pc / pc[:, 2:3]).T).T[:, :2]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < green.shape[1]) & (uv[:, 1] < green.shape[0])
            if inside.sum() > 10 and green[uv[inside, 1], uv[inside, 0]].mean() > 0.3:
                planted.add(k)
                green_cells.add(k)
        print("cells with a plant over them (path obstacles):", sorted(planted))
    rec = {**g.summary(), "occupied": sorted(occ), "planted": sorted(planted), "green": sorted(green_cells), "soil_z_overview": soil, "cells": g.cells, "q": q.tolist(), "T_BC": T_BC.tolist(), "K": frame.intr.K().tolist(), "calibration": meta.get("calibration_id")}
    (out / "tray_grid.json").write_text(json.dumps(rec, indent=1))
    if g.ok and not args.no_save:
        p = PROJECT_DIR / cfg["tray"]["grid_file"]
        p.write_text(json.dumps(rec, indent=1))
        print("saved", p.relative_to(PROJECT_DIR))
    # the transplant disc, if one is in view (seedling/disc.py): saved to calibration/disc.json, drawn on the overlay
    from seedling.disc import locate_disc
    from seedling.transforms import apply_T, inv_T

    # the previous position as a prior: used only when too few clean hole rims are left (a filling disc)
    dp = PROJECT_DIR / "calibration" / "disc.json"
    prev = json.loads(dp.read_text()) if dp.exists() else None
    d = locate_disc(frame, T_BC, prior=prev if prev and prev.get("ok") else None)
    if d.get("centre") is not None:
        print(f"disc: {d['status']}; centre {np.round(d['centre'], 4).tolist()}, top z {d['z_top']:.3f}, "
              f"{len(d.get('holes_seen', []))} holes seen (rim radius mm {d.get('hole_radius_mm')}, hole circle {d.get('hole_circle_r_mm')} mm)")
        img = cv2.imread(str(out / "cells.jpg")) if (out / "cells.jpg").exists() else cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        K = frame.intr.K()
        for i, (hx, hy) in enumerate(d["holes"], 1):
            ring = np.array([[hx + 0.02475 * np.cos(t), hy + 0.02475 * np.sin(t), d["z_top"]] for t in np.linspace(0, 2 * np.pi, 40)])
            pc = apply_T(inv_T(T_BC), ring)
            uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.int32)
            cv2.polylines(img, [uv], True, (255, 255, 0) if d["ok"] else (0, 0, 255), 2)
            c0 = uv.mean(0).astype(int)
            cv2.putText(img, f"H{i}", (int(c0[0]) - 12, int(c0[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imwrite(str(out / "cells.jpg"), img)
        if d["ok"] and not args.no_save:
            dp.write_text(json.dumps({**d, "q": q.tolist()}, indent=1))
            print("saved", dp.relative_to(PROJECT_DIR))
    else:
        print("disc:", d["status"])
    if not d["ok"] and not args.no_save:
        # mark the saved disc as not found in this picture (the transplant loop must not use an old free-hole list); the
        # position is kept as the prior for the next picture
        if prev and prev.get("ok"):
            dp.write_text(json.dumps({**prev, "ok_last_picture": False, "last_status": d["status"]}, indent=1))
    print("overlay", (out / "cells.jpg").relative_to(PROJECT_DIR))
    return 0 if g.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
