#!/usr/bin/env python
"""Live stem localisation cell by cell: the tray grid gives each plug; for each cell the wrist camera moves to an
oblique view of that plug, takes a picture and looks only for the stem whose root is on that plug (stem-line
stereo). A stem that runs horizontally in the image cannot be triangulated (parallel to the stereo baseline):
then the camera is rolled about its optical axis until the stem is vertical in the image and a second picture
is taken. Optionally one more view 30 deg around the plant to measure how well two estimates agree.
Returns to the start pose at the end.

    .venvs/pipeline/bin/python scripts/59_tray_cells_live.py --cells experiments/survey/plugs_20261020_moved.json --dry-run
    .venvs/pipeline/bin/python scripts/59_tray_cells_live.py --cells ... --views 2 --no-step
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


def plan(target, q, T_FC, floor, cfg, distance, azimuths, elevations, rolls=None):
    from seedling.wrist import plan_observation

    c = copy.deepcopy(cfg)
    o = c["tray"]["observe"]
    o["distance_m"], o["azimuth_offsets_deg"], o["elevation_deg"] = distance, list(azimuths), list(elevations)
    if rolls is not None:
        o["roll_deg"] = list(rolls)
    return plan_observation(target, q, T_FC, floor, c)


def parse(desc: str) -> dict:
    """'elev 65 deg, azimuth +15 deg, roll +45 deg' -> {'elev': 65, 'azimuth': 15, 'roll': 45}"""
    return {k.strip().split()[0]: float(k.strip().split()[1]) for k in desc.split(",")}


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--cells", required=True, help="JSON {cell: [x, y, z]} of plug tops (base frame, m)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--distance", type=float, default=0.34)
    ap.add_argument("--azimuths", type=float, nargs="+", default=[0, -15, 15])
    ap.add_argument("--elevation", type=float, nargs="+", default=[65, 55])
    ap.add_argument("--rolls", type=float, nargs="+", default=[0, 180],
                    help="camera roll of the first view: 0 shows an upright stem vertical in the image (what the stereo needs)")
    ap.add_argument("--return-q", default=None, help="joints (JSON) to return to at the end (default: where it started)")
    ap.add_argument("--views", type=int, default=1, choices=(1, 2), help="2: after a success, one more view 30 deg around")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true", help="plan the first view of every cell only (no motion, no camera)")
    ap.add_argument("--no-step", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import flange_pose, load_T_FC, now_id
    from seedling.robot import FrankyRobot
    from seedling.stemline import estimate_cell_stem_line
    from seedling.transforms import apply_T, inv_T

    T_FC, meta = load_T_FC(cfg)
    cells = {k: np.array(v, float) for k, v in json.loads(open(args.cells).read()).items() if v is not None}
    if args.only:
        cells = {k: v for k, v in cells.items() if k in args.only}
    clearance = float(cfg["tray"]["observe"]["min_clearance_m"])
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    q_back = np.array(json.loads(args.return_q)) if args.return_q else q0

    if args.dry_run:
        q = q0
        for name, p in cells.items():
            v = plan(p + [0, 0, 0.02], q, T_FC, float(p[2]) + clearance, cfg, args.distance, args.azimuths, args.elevation, args.rolls)
            print(f"{name}: " + (f"{v.desc}, camera {np.round(v.T_BC[:3, 3], 3).tolist()}, max joint change {v.max_joint_step_deg:.0f} deg, "
                                 f"lowest {v.lowest_part} z={v.lowest_z:.3f}" if v.ok else "NO SAFE POSE"))
            q = v.q if v.ok else q
        print("(a rolled view, when needed, is planned live with the same checks)")
        return 0

    from seedling.camera import ZedCamera

    out = PROJECT_DIR / "experiments" / "cells_live" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    cam = ZedCamera(cfg)
    rows = []

    def look(name, k, p, v):
        print(f"{name} view {k}: {v.desc}, max joint change {v.max_joint_step_deg:.0f} deg, lowest {v.lowest_part} z={v.lowest_z:.3f}", flush=True)
        r = robot.move_joints(v.q, dynamics=args.dynamics, description=f"{name} view {k}")
        if not r.ok:
            raise RuntimeError(f"motion failed: {r.reason}")
        time.sleep(float(cfg["tray"]["observe"].get("settle_s", 0.5)))
        f = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        qn = robot.q()
        f.meta["q"] = qn.tolist()
        T_BC = flange_pose(robot) @ T_FC
        tag = f"{name}_v{k}"
        f.save(out / tag, save_png=False)
        e, tried = estimate_cell_stem_line(f, T_BC, cfg, p)
        row = {"cell": name, "view": k, "pose": v.desc, "plug": p.tolist(), "T_BC": T_BC.tolist(), "q": qn.tolist(),
               "found": e is not None, "estimate": e.summary() if e else None,
               "tried": [[r_.tolist(), s] for r_, s, _ in tried]}
        rows.append(row)
        (out / "cells.json").write_text(json.dumps(rows, indent=1, default=float))
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        for r_, _, left in tried:
            cv2.circle(img, tuple(int(x) for x in r_), 5, (0, 0, 255), 1)
            if left is not None and (e is None or left is not e.left):
                b = left.band
                img[b[:, 1], b[:, 0]] = (255, 255, 0)
        if e:
            b = e.left.band
            img[b[:, 1], b[:, 0]] = (255, 0, 255)
            cv2.circle(img, tuple(int(x) for x in e.base_px), 6, (0, 255, 255), 2)
        uv = f.intr.project(apply_T(inv_T(T_BC), p)).astype(int)
        cv2.drawMarker(img, tuple(int(x) for x in uv), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.imwrite(str(out / f"{tag}.jpg"), img)
        if e:
            s = e.summary()
            print(f"  FOUND root {np.round(e.root_B, 4).tolist()} ({np.linalg.norm((e.root_B - p)[:2]) * 1000:.0f} mm from the cell centre), "
                  f"tilt {s['tilt_deg']}, grasp {np.round(e.grasp_B, 4).tolist()}, score {s['match_score']}, "
                  f"window resid {s['disparity_resid_px']} px, root vs soil {s['root_vs_soil_mm']} mm", flush=True)
        else:
            from collections import Counter
            print("  not found: " + ", ".join(f"{n}x {t}" for t, n in Counter(s.split("(")[0].strip() for _, s, _ in tried).items()), flush=True)
        return e, tried, f, T_BC

    try:
        for name, p in cells.items():
            tgt, floor = p + [0, 0, 0.02], float(p[2]) + clearance
            v1 = plan(tgt, robot.q(), T_FC, floor, cfg, args.distance, args.azimuths, args.elevation, args.rolls)
            if not v1.ok:
                print(f"{name}: no safe observation pose", flush=True)
                continue
            e, tried, f, T_BC = look(name, 1, p, v1)
            used = parse(v1.desc)
            if e is None:
                # a stem lying across the image: roll the camera so that it runs vertically in the image
                horiz = [(r_, left) for r_, s, left in tried if s.startswith("stem nearly horizontal") and left is not None]
                if horiz:
                    ctr = f.intr.project(apply_T(inv_T(T_BC), p))
                    _, left = min(horiz, key=lambda h: float(np.linalg.norm(h[0] - ctr)))
                    phi = float(np.degrees(np.arctan2(left.direction[0], left.direction[1])))  # angle from the image vertical
                    phi = (phi + 90.0) % 180.0 - 90.0  # a line: +-90
                    # at most 60 deg (a horizontal stem then runs 30 deg from the horizontal: enough for the stereo
                    # match), to spare the camera cable big wrist spins
                    step = float(np.clip(phi, -60.0, 60.0))
                    for roll in (wrap(used["roll"] + step), wrap(used["roll"] - step)):
                        v2 = plan(tgt, robot.q(), T_FC, floor, cfg, args.distance, [used["azimuth"]], [used["elev"]], [round(roll)])
                        if v2.ok:
                            break
                    if v2.ok:
                        print(f"  stem at {phi:+.0f} deg from the image vertical: roll the camera", flush=True)
                        e, tried, f, T_BC = look(name, 2, p, v2)
                        used = parse(v2.desc)
                    else:
                        print("  rolled view not reachable", flush=True)
            if e is not None and args.views == 2:
                for daz in (30, -30):
                    v3 = plan(tgt, robot.q(), T_FC, floor, cfg, args.distance, [used["azimuth"] + daz], [used["elev"]], [used["roll"]])
                    if v3.ok:
                        look(name, 3, p, v3)
                        break
    except Exception as exc:  # noqa: BLE001
        print("stopped:", exc, flush=True)
    finally:
        cam.close()
        from seedling.wrist import path_clearance

        z, part = path_clearance(robot.q(), q_back, T_FC, cfg)
        if z >= 0.2 and np.degrees(np.abs(robot.q() - q_back))[:6].max() <= 60:
            robot.move_joints(q_back, dynamics=args.dynamics, description="back to the start pose")
        else:
            print(f"not returning automatically (lowest {part} z={z:.3f} or large joint change)", flush=True)

    print("\nSUMMARY", flush=True)
    summary = {}
    for name in dict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == name]
        es = [r["estimate"] for r in rs if r["found"]]
        line = f"{name}: found in {len(es)}/{len(rs)} views"
        item = {"views": len(rs), "found": len(es)}
        if es:
            item["root_B"], item["grasp_B"], item["tilt_deg"] = es[0]["root_B"], es[0]["grasp_B"], es[0]["tilt_deg"]
        if len(es) >= 2:
            d_root = (np.array(es[-1]["root_B"]) - es[0]["root_B"]) * 1000
            d_grasp = (np.array(es[-1]["grasp_B"]) - es[0]["grasp_B"]) * 1000
            ang = float(np.degrees(np.arccos(np.clip(np.dot(es[0]["dir_B"], es[-1]["dir_B"]), -1, 1))))
            item.update(root_diff_mm=np.round(d_root, 1).tolist(), grasp_diff_mm=round(float(np.linalg.norm(d_grasp)), 1), dir_diff_deg=round(ang, 1))
            line += f"; two views: root diff {np.round(d_root, 1).tolist()} mm, grasp point diff {np.linalg.norm(d_grasp):.1f} mm, direction {ang:.1f} deg"
        summary[name] = item
        print(line, flush=True)
    (out / "summary.json").write_text(json.dumps({"cells": summary, "calibration": meta.get("calibration_id")}, indent=1))
    print("saved", out.relative_to(PROJECT_DIR), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
