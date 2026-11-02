#!/usr/bin/env python
"""Wrist camera: move to the observation pose of one target, take a picture, run the per-cell stem estimate.

No gripper needed. Targets:
  --cell A1            a cell of the taught tray (tray.file)
  --marker-id 1        a marker of the calibration sheet (its centre, from T_flange_cam.yaml) as the "cell":
                       also checks the camera pose at an oblique view: the marker is detected and its position
                       compared with the calibration (put a fake stem next to it to test the stem estimate)
  --point X Y Z        substrate point in the base frame (m)

    .venvs/pipeline/bin/python scripts/56_wrist_observe.py --marker-id 1 --dry-run     # plan only, no motion
    .venvs/pipeline/bin/python scripts/56_wrist_observe.py --marker-id 1               # every move confirmed
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell")
    g.add_argument("--marker-id", type=int)
    g.add_argument("--point", type=float, nargs=3)
    ap.add_argument("--grasp-height-mm", type=float, default=30.0, help="for --marker-id/--point: grasp height above the substrate")
    ap.add_argument("--marker-mm", type=float, default=None, help="marker side (default: from the calibration)")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--elevation", type=float, nargs="*", default=None, help="override tray.observe.elevation_deg")
    ap.add_argument("--dynamics", type=float, default=None, help="default robot.dynamics")
    ap.add_argument("--dry-run", action="store_true", help="plan only (reads the robot state, no motion, no camera)")
    ap.add_argument("--no-step", action="store_true", help="do not ask before each move")
    ap.add_argument("--no-return", action="store_true", help="stay at the observation pose")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.elevation:
        cfg["tray"]["observe"]["elevation_deg"] = args.elevation

    from seedling.calibration import detect_markers, flange_pose, load_T_FC, now_id
    from seedling.robot import FrankyRobot
    from seedling.tray import Tray, estimate_cell_stem
    from seedling.transforms import apply_T, inv_T
    from seedling.wrist import lowest_body_z, plan_observation

    T_FC, meta = load_T_FC(cfg)
    gh = args.grasp_height_mm / 1000.0
    marker_ref = None
    if args.cell:
        tray = Tray.load(cfg["tray"]["file"])
        nominal, soil_z, gh = tray.grasp_point(args.cell), tray.soil_z(args.cell), tray.grasp_height_m
        name = args.cell
    elif args.marker_id is not None:
        refs = meta.get("T_base_marker_at_calibration") or {}
        if str(args.marker_id) not in refs:
            raise SystemExit(f"marker {args.marker_id} not in the calibration ({sorted(refs)})")
        marker_ref = np.array(refs[str(args.marker_id)])[:3, 3]
        soil_z = float(marker_ref[2])
        nominal = marker_ref + [0, 0, gh]
        name = f"marker{args.marker_id}"
    else:
        soil_z = args.point[2]
        nominal = np.array(args.point) + [0, 0, gh]
        name = "point"

    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    floor = soil_z + float(cfg["tray"]["observe"]["min_clearance_m"])
    plan = plan_observation(nominal, q0, T_FC, floor, cfg)
    z0, part0 = lowest_body_z(q0, T_FC, cfg)
    print(f"target {name}: grasp point {np.round(nominal, 4).tolist()} m, substrate z {soil_z:.3f}, floor z {floor:.3f}")
    print(f"now: lowest part {part0} at z={z0:.3f}")
    for desc, why in plan.tried or []:
        print(f"  skipped: {desc}: {why}")
    if not plan.ok:
        print("no safe observation pose")
        return 1
    cam_pos = plan.T_BC[:3, 3]
    print(f"observation pose: {plan.desc}; camera at {np.round(cam_pos, 3).tolist()} m; "
          f"max joint change {plan.max_joint_step_deg:.0f} deg; lowest part on the way: {plan.lowest_part} z={plan.lowest_z:.3f}")
    print(f"  q now    {np.round(q0, 3).tolist()}\n  q target {np.round(plan.q, 3).tolist()}")
    if args.dry_run:
        return 0

    from seedling.camera import ZedCamera
    from seedling.viz import save_rgb

    out = PROJECT_DIR / "experiments" / "wrist_observe" / f"{now_id()}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    dyn = float(args.dynamics if args.dynamics is not None else cfg["robot"]["dynamics"])
    cam = ZedCamera(cfg)
    moved = False
    try:
        r = robot.move_joints(plan.q, dynamics=dyn, description=f"observation pose for {name}")
        moved = True
        if not r.ok:
            print("motion failed:", r.reason, "(robot left where it stopped)")
            return 1
        time.sleep(float(cfg["tray"]["observe"].get("settle_s", 0.5)))
        f = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        T_BC = flange_pose(robot) @ T_FC
        f.save(out / "frame", save_png=False)
        est = estimate_cell_stem(f, T_BC, nominal, soil_z, gh, cfg)
        res = {"target": name, "nominal": nominal.tolist(), "observation": plan.desc, "T_BC": T_BC.tolist(), "estimate": est.summary()}
        print(f"\nstem estimate: {est.status}  tilt {est.tilt_deg}  offset {est.offset_mm} mm  "
              f"(points in cone {est.n_roi}, green {est.n_green}, inliers {est.n_inliers})")
        img = f.rgb.copy()
        T_CB = inv_T(T_BC)
        if marker_ref is not None:
            L = float(args.marker_mm / 1000.0 if args.marker_mm else meta.get("marker_length_m", 0.04))
            obs = {o.marker_id: o for o in detect_markers(f, L, args.dict)}
            refs = meta.get("T_base_marker_at_calibration") or {}
            rows = []
            for k, o in sorted(obs.items()):
                if str(k) in refs:
                    m = apply_T(T_BC @ o.T_CM, np.zeros(3))
                    # lateral check from the pixel ray alone (no PnP depth): where the ray meets the sheet plane
                    uv = o.corners_px.mean(0)
                    ray = T_BC[:3, :3] @ np.array([(uv[0] - f.intr.cx) / f.intr.fx, (uv[1] - f.intr.cy) / f.intr.fy, 1.0])
                    ref = np.array(refs[str(k)])[:3, 3]
                    hit = T_BC[:3, 3] + ray * (ref[2] - T_BC[2, 3]) / ray[2]
                    rows.append({"id": k, "pnp_diff_mm": np.round((m - ref) * 1000, 1).tolist(),
                                 "ray_on_plane_diff_mm": np.round((hit - ref)[:2] * 1000, 1).tolist()})
                    print(f"marker {k}: PnP position vs calibration {rows[-1]['pnp_diff_mm']} mm; "
                          f"pixel ray on the sheet plane vs calibration (x, y) {rows[-1]['ray_on_plane_diff_mm']} mm")
            res["marker_check"] = rows
        for p, colr in ((nominal, (255, 0, 0)), (est.p_B, (0, 255, 0))):
            if p is not None:
                uv = f.intr.project(apply_T(T_CB, p)).astype(int)
                cv2.drawMarker(img, tuple(int(x) for x in uv), colr, cv2.MARKER_CROSS, 30, 3)
        if est.pixel_box:
            u0, v0, u1, v1 = est.pixel_box
            cv2.rectangle(img, (u0, v0), (u1, v1), (255, 255, 0), 2)
        save_rgb(out / "overlay.jpg", img)
        (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
        print("saved", out.relative_to(PROJECT_DIR))
    finally:
        cam.close()
        if moved and not args.no_return:
            try:
                robot.move_joints(q0, dynamics=dyn, description="back to the start pose")
            except Exception as exc:  # noqa: BLE001
                print("could not return:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
