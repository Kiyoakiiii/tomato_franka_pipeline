#!/usr/bin/env python
"""Live: for every given cell, the stem where it leaves the soil (seedling/stembase.py), seen from two
directions (azimuth -A and +A around the plant, camera roll 0). Gives the root (soil/stem junction), the lower
stem direction and a grasp point per plant, and how well the two views agree. Returns at the end.

    .venvs/pipeline/bin/python scripts/62_stem_base_live.py --cells calibration/tray_grid.json --dry-run
    .venvs/pipeline/bin/python scripts/62_stem_base_live.py --cells calibration/tray_grid.json --no-step
(--cells: a tray grid file (its "occupied" cells, plug top = tray top + --plug-dz) or {cell: [x, y, z]})
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import copy
import json
import time

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--plug-dz", type=float, default=0.013, help="plug top above the tray top (grid files)")
    ap.add_argument("--distance", type=float, default=0.30)
    ap.add_argument("--elevation", type=float, nargs="+", default=[55, 65, 45])
    ap.add_argument("--azimuth", type=float, default=20.0, help="the two views at -A and +A")
    ap.add_argument("--grasp-height-m", type=float, default=0.015, help="grasp point this high above the soil, on the stem line")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--return-q", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-step", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import flange_pose, load_T_FC, now_id
    from seedling.robot import FrankyRobot
    from seedling.stembase import find_stem_base
    from seedling.transforms import apply_T, inv_T
    from seedling.wrist import path_clearance, plan_observation

    T_FC, meta = load_T_FC(cfg)
    src = json.loads(open(args.cells).read())
    if "cells" in src and "occupied" in src:
        cells = {k: np.array([*src["cells"][k][:2], src["z_top"] + args.plug_dz]) for k in src["occupied"]}
    else:
        cells = {k: np.array(v, float) for k, v in src.items()}
    if args.only:
        cells = {k: v for k, v in cells.items() if k in args.only}
    clearance = float(cfg["tray"]["observe"]["min_clearance_m"])

    def plan(target, q, az):
        c = copy.deepcopy(cfg)
        o = c["tray"]["observe"]
        o.update(distance_m=args.distance, azimuth_offsets_deg=[az], elevation_deg=list(args.elevation), roll_deg=[0, 180])
        return plan_observation(target, q, T_FC, float(target[2]) - 0.01 + clearance, c)

    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    q_back = np.array(json.loads(args.return_q)) if args.return_q else q0
    # whole tour planned first; nothing moves if a view is impossible
    tour, q = [], q0
    # all cells from one side, then all from the other: one big wrist change instead of one per cell
    for k, az in enumerate((-args.azimuth, args.azimuth), 1):
        for name, p in (cells.items() if k == 1 else reversed(list(cells.items()))):
            tgt = p + [0, 0, 0.01]
            v = plan(tgt, q, az)
            print(f"{name} view {k}: " + (f"{v.desc}, camera {np.round(v.T_BC[:3, 3], 3).tolist()}, max joint change {v.max_joint_step_deg:.0f} deg, "
                                          f"lowest {v.lowest_part} z={v.lowest_z:.3f}" if v.ok else "NO SAFE POSE " + str(v.tried[-2:])))
            if v.ok:
                tour.append((name, k, p, v))
                q = v.q
    z, part = path_clearance(q, q_back, T_FC, cfg)
    print(f"return: max joint change {np.degrees(np.abs(q_back - q)).max():.0f} deg, lowest {part} z={z:.3f}")
    if args.dry_run:
        return 0

    from seedling.camera import ZedCamera

    out = PROJECT_DIR / "experiments" / "stem_base" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    cam = ZedCamera(cfg)
    rows = []
    try:
        for name, k, p, v in tour:
            print(f"{name} view {k}: {v.desc}", flush=True)
            r = robot.move_joints(v.q, dynamics=args.dynamics, description=f"{name} view {k}")
            if not r.ok:
                print("motion failed:", r.reason, flush=True)
                break
            time.sleep(float(cfg["tray"]["observe"].get("settle_s", 0.5)))
            f = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
            qm = robot.q()
            f.meta["q"] = qm.tolist()
            T_BC = flange_pose(robot) @ T_FC
            tag = f"{name}_v{k}"
            f.save(out / tag, save_png=False)
            e = find_stem_base(f, T_BC, cfg, p)
            rows.append({"cell": name, "view": k, "pose": v.desc, "plug": p.tolist(), "q": qm.tolist(), "T_BC": T_BC.tolist(), **e.summary(),
                         "grasp_B": None if not e.ok else np.round(e.at_height(args.grasp_height_m), 4).tolist()})
            (out / "base.json").write_text(json.dumps(rows, indent=1, default=float))
            img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            if e.left is not None:
                b = e.left.band
                img[b[:, 1], b[:, 0]] = (255, 0, 255) if e.ok else (0, 255, 255)
            T_CB = inv_T(T_BC)
            cv2.drawMarker(img, tuple(int(a) for a in f.intr.project(apply_T(T_CB, p))), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
            if e.ok:
                cv2.drawMarker(img, tuple(int(a) for a in f.intr.project(apply_T(T_CB, e.root_B))), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.imwrite(str(out / f"{tag}.jpg"), img)
            s = e.summary()
            print(f"  {'FOUND' if e.ok else 'not found'}: {s['status'][:90]}" + (f"; root {s['root_B']} ({s['root_off_centre_mm']} mm from the cell centre), "
                  f"tilt {s['tilt_deg']} deg, score {s['match_score']}, resid {s['disparity_resid_px']}" if e.ok else ""), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("stopped:", exc, flush=True)
    finally:
        cam.close()
        z, part = path_clearance(robot.q(), q_back, T_FC, cfg)
        if z >= 0.2 and np.degrees(np.abs(robot.q() - q_back))[:6].max() <= 60:
            robot.move_joints(q_back, dynamics=args.dynamics, description="back to the start pose")
        else:
            print(f"not returning automatically (lowest {part} z={z:.3f} or large joint change)", flush=True)

    print("\nSUMMARY", flush=True)
    summ = {}
    for name in dict.fromkeys(r["cell"] for r in rows):
        es = [r for r in rows if r["cell"] == name and r["ok"]]
        item = {"found": len(es), "views": sum(r["cell"] == name for r in rows)}
        line = f"{name}: found in {item['found']}/{item['views']} views"
        if es:
            root = np.mean([e["root_B"] for e in es], axis=0)
            d = np.mean([e["dir_B"] for e in es], axis=0)
            d /= np.linalg.norm(d)
            item.update(root_B=np.round(root, 4).tolist(), dir_B=np.round(d, 3).tolist(), tilt_deg=round(float(np.degrees(np.arccos(d[2]))), 1),
                        grasp_B=np.round(root + d * args.grasp_height_m / d[2], 4).tolist(),
                        root_off_centre_mm=round(float(np.hypot(*(root[:2] - np.array(es[0]["plug"][:2]))) * 1000), 1))
            line += f"; root {item['root_B']} ({item['root_off_centre_mm']} mm from the centre), tilt {item['tilt_deg']} deg, grasp {item['grasp_B']}"
        if len(es) == 2:
            dr = (np.array(es[1]["root_B"]) - es[0]["root_B"]) * 1000
            ang = float(np.degrees(np.arccos(np.clip(np.dot(es[0]["dir_B"], es[1]["dir_B"]), -1, 1))))
            g = [np.array(e["grasp_B"]) for e in es]
            item.update(root_diff_mm=round(float(np.linalg.norm(dr[:2])), 1), dir_diff_deg=round(ang, 1), grasp_diff_mm=round(float(np.linalg.norm(g[1] - g[0]) * 1000), 1))
            line += f"; two views: root {item['root_diff_mm']} mm, direction {item['dir_diff_deg']} deg, grasp point {item['grasp_diff_mm']} mm apart"
        summ[name] = item
        print(line, flush=True)
    (out / "summary.json").write_text(json.dumps({"cells": summ, "calibration": meta.get("calibration_id")}, indent=1))
    print("saved", out.relative_to(PROJECT_DIR), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
