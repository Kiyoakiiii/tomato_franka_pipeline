#!/usr/bin/env python
"""Wrist camera survey of several plants: for each plant, move to an oblique observation pose close to it,
take a picture, find the stem whose root is on that plant's plug (stem-line stereo); optionally a second view
from another direction to measure how well the two estimates agree. Returns to the start pose at the end.

Targets: plug-top points in the base frame (m), from a JSON file {"name": [x, y, z], ...}.

    .venvs/pipeline/bin/python scripts/58_survey_plants.py --targets plugs.json --dry-run      # plan only
    .venvs/pipeline/bin/python scripts/58_survey_plants.py --targets plugs.json --views 2      # every move confirmed
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


def plan_view(target, q, T_FC, floor, cfg, distance, azimuths, elevations):
    from seedling.wrist import plan_observation

    c = copy.deepcopy(cfg)
    c["tray"]["observe"]["distance_m"] = distance
    c["tray"]["observe"]["azimuth_offsets_deg"] = azimuths
    c["tray"]["observe"]["elevation_deg"] = elevations
    return plan_observation(target, q, T_FC, floor, c)


def az_of(desc: str) -> float:
    return float(desc.split("azimuth")[1].split("deg")[0])


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--targets", required=True, help="JSON {name: [x, y, z]} of plug tops (base frame, m)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of target names")
    ap.add_argument("--distance", type=float, default=0.34, help="camera to the point 2 cm above the plug (ZED min 0.3 m)")
    ap.add_argument("--views", type=int, default=1, choices=(1, 2), help="2: a second view 30 deg around the plant")
    ap.add_argument("--azimuths", type=float, nargs="+", default=[0, -15, 15],
                    help="first view, relative to base->plant (0: camera between robot and plant). Large values look "
                         "along the row of plants (they hide each other)")
    ap.add_argument("--elevation", type=float, nargs="+", default=[65, 55], help="down-looking angle, tried in order")
    ap.add_argument("--match-radius-mm", type=float, default=25.0, help="a stem belongs to the plant if its root is this close (xy)")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true", help="plan only (reads the robot state, no motion, no camera)")
    ap.add_argument("--no-step", action="store_true", help="do not ask before each move")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import flange_pose, load_T_FC, now_id
    from seedling.robot import FrankyRobot
    from seedling.stemline import estimate_all_stems
    from seedling.transforms import apply_T, inv_T

    T_FC, meta = load_T_FC(cfg)
    targets = {k: np.array(v, float) for k, v in json.loads(open(args.targets).read()).items() if v is not None}
    if args.only:
        targets = {k: v for k, v in targets.items() if k in args.only}
    clearance = float(cfg["tray"]["observe"]["min_clearance_m"])

    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    # plan the whole tour first (each view from the previous one), refuse to start if any view is impossible
    tour, q = [], q0
    for name, p in targets.items():
        tgt, floor = p + [0, 0, 0.02], float(p[2]) + clearance
        pl = plan_view(tgt, q, T_FC, floor, cfg, args.distance, args.azimuths, args.elevation)
        views = [pl]
        if pl.ok and args.views == 2:
            a1 = az_of(pl.desc)
            views.append(plan_view(tgt, pl.q, T_FC, floor, cfg, args.distance, [a1 + 30, a1 - 30], args.elevation))
        for k, v in enumerate(views):
            print(f"{name} view {k + 1}: " + (f"{v.desc}, camera {np.round(v.T_BC[:3, 3], 3).tolist()}, max joint change "
                                              f"{v.max_joint_step_deg:.0f} deg, lowest {v.lowest_part} z={v.lowest_z:.3f}"
                                              if v.ok else "NO SAFE POSE"))
            if v.ok:
                tour.append((name, k + 1, p, v))
                q = v.q
    if args.dry_run:
        return 0
    if not tour:
        return 1

    from seedling.camera import ZedCamera

    out = PROJECT_DIR / "experiments" / "survey" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    cam = ZedCamera(cfg)
    rows = []
    try:
        for name, k, p, v in tour:
            r = robot.move_joints(v.q, dynamics=args.dynamics, description=f"{name} view {k}")
            if not r.ok:
                print("motion failed:", r.reason)
                break
            time.sleep(float(cfg["tray"]["observe"].get("settle_s", 0.5)))
            f = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
            qn = robot.q()
            f.meta["q"] = qn.tolist()
            T_BC = flange_pose(robot) @ T_FC
            tag = f"{name}_v{k}"
            f.save(out / tag, save_png=False)
            found, rejected = estimate_all_stems(f, T_BC, cfg)
            near = [e for e in found if np.linalg.norm((e.root_B - p)[:2]) * 1000 <= args.match_radius_mm]
            best = min(near, key=lambda e: np.linalg.norm((e.root_B - p)[:2])) if near else None
            row = {"plant": name, "view": k, "pose": v.desc, "plug": p.tolist(), "T_BC": T_BC.tolist(), "q": qn.tolist(),
                   "found": best is not None, "estimate": best.summary() if best else None,
                   "other_stems": len(found) - (best is not None), "rejected": [[r_.tolist(), s] for r_, s in rejected]}
            rows.append(row)
            print(f"{tag}: " + (f"root {np.round(best.root_B, 4).tolist()} tilt {best.tilt_deg:.1f} grasp {np.round(best.grasp_B, 4).tolist()} "
                                f"score {best.match_score:.2f} resid {best.disparity_resid_px} soil {best.root_vs_soil_mm}"
                                if best else f"no stem at this plug ({len(found)} other stems, {len(rejected)} rejected candidates)"))
            img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            T_CB = inv_T(T_BC)
            for e in found:
                b = e.left.band
                img[b[:, 1], b[:, 0]] = (255, 0, 255) if e is best else (255, 255, 0)
                cv2.circle(img, tuple(int(x) for x in e.base_px), 6, (0, 255, 255), 2)
            for r_, _ in rejected:
                cv2.circle(img, tuple(int(x) for x in r_), 4, (0, 0, 255), 1)
            uv = f.intr.project(apply_T(T_CB, p)).astype(int)
            cv2.drawMarker(img, tuple(int(x) for x in uv), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.imwrite(str(out / f"{tag}.jpg"), img)
            (out / "survey.json").write_text(json.dumps(rows, indent=1, default=float))
    finally:
        cam.close()
        robot.move_joints(q0, dynamics=args.dynamics, description="back to the start pose")

    # summary: found per plant, and view-to-view agreement
    print("\nSUMMARY")
    agree = {}
    for name in dict.fromkeys(r["plant"] for r in rows):
        es = [r["estimate"] for r in rows if r["plant"] == name and r["found"]]
        line = f"{name}: found in {len(es)}/{sum(r['plant'] == name for r in rows)} views"
        if len(es) == 2:
            d_root = (np.array(es[1]["root_B"]) - es[0]["root_B"]) * 1000
            d_grasp = (np.array(es[1]["grasp_B"]) - es[0]["grasp_B"]) * 1000
            ang = float(np.degrees(np.arccos(np.clip(np.dot(es[0]["dir_B"], es[1]["dir_B"]), -1, 1))))
            agree[name] = {"root_diff_mm": np.round(d_root, 1).tolist(), "grasp_diff_mm": np.round(d_grasp, 1).tolist(), "dir_diff_deg": round(ang, 1)}
            line += f"; root diff {np.round(d_root, 1).tolist()} mm, grasp diff {np.linalg.norm(d_grasp):.1f} mm, direction {ang:.1f} deg"
        print(line)
    (out / "summary.json").write_text(json.dumps({"agreement": agree, "calibration": meta.get("calibration_id")}, indent=1))
    print("saved", out.relative_to(PROJECT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
