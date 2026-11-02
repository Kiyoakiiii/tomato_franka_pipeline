#!/usr/bin/env python
"""Stem centre line from the ZED left/right images at the current robot pose (no motion), or from a saved frame.

    .venvs/pipeline/bin/python scripts/57_stem_triangulate.py                    # live: capture here
    .venvs/pipeline/bin/python scripts/57_stem_triangulate.py --seed 1100 520    # pick the plant nearest this pixel
    .venvs/pipeline/bin/python scripts/57_stem_triangulate.py --frame experiments/stemline/<id>/frame.npz --q '[...]'

Prints the root (on the substrate), the stem direction/tilt, the grasp point (base frame) and two independent
consistency checks; saves overlays of both images.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def overlay(frame, est, T_BC):
    from seedling.transforms import apply_T, inv_T

    L = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    R = cv2.cvtColor(frame.rgb_right, cv2.COLOR_RGB2BGR) if frame.rgb_right is not None else None
    if est.left is not None:
        b = est.left.band
        L[b[:, 1], b[:, 0]] = (0.4 * L[b[:, 1], b[:, 0]] + [0, 0, 150]).astype(np.uint8)
        for p in est.left.path.astype(int):
            L[p[1], p[0]] = (255, 255, 0)
        c, d = est.left.centre, est.left.direction
        cv2.line(L, tuple((c - 80 * d).astype(int)), tuple((c + 80 * d).astype(int)), (0, 255, 255), 1)
    if est.base_px is not None:
        cv2.circle(L, tuple(int(x) for x in est.base_px), 4, (0, 128, 255), 1)
    if R is not None and est.right_centre is not None:
        c, d = est.right_centre, est.right_direction
        cv2.line(R, tuple((c - 80 * d).astype(int)), tuple((c + 80 * d).astype(int)), (0, 255, 255), 1)
    T_CB = inv_T(T_BC)
    for p, col in ((est.root_B, (255, 0, 255)), (est.grasp_B, (0, 255, 0)), (est.top_B, (255, 128, 0))):
        if p is not None:
            uv = frame.intr.project(apply_T(T_CB, p)).astype(int)
            cv2.drawMarker(L, tuple(int(x) for x in uv), col, cv2.MARKER_CROSS, 22, 2)
    cv2.putText(L, f"{est.status} tilt={est.tilt_deg and round(est.tilt_deg, 1)} score={est.match_score:.2f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return L, R


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--frame", default=None, help="saved frame.npz instead of a live capture")
    ap.add_argument("--q", default=None, help="with --frame: joint angles (JSON list) at capture")
    ap.add_argument("--seed", type=float, nargs=2, default=None, help="pixel (u v) near the plant to use")
    ap.add_argument("--orbit-deg", type=float, default=None,
                    help="validation: also move the camera this far around the stem (vertical axis through the grasp point), "
                         "measure again, return; both results must agree in the base frame")
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--no-step", action="store_true", help="do not ask before each move")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import T_base_flange, load_T_FC, now_id
    from seedling.camera import Frame
    from seedling.config import flange_to_ee
    from seedling.stemline import estimate_stem_line

    T_FC, meta = load_T_FC(cfg)
    out = PROJECT_DIR / "experiments" / "stemline" / now_id()
    out.mkdir(parents=True, exist_ok=True)
    if args.frame:
        frame = Frame.load(args.frame)
        q = np.array(json.loads(args.q)) if args.q else np.array(frame.meta["q"])
    else:
        from seedling.camera import ZedCamera
        from seedling.robot import FrankyRobot

        robot = FrankyRobot(cfg, step_mode=True)
        cam = ZedCamera(cfg)  # opened once per process: re-opening the ZED in the same process crashes the SDK
        try:
            frame = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        except Exception:
            cam.close()
            raise
        q = robot.q()
        frame.meta["q"] = q.tolist()
        frame.save(out / "frame", save_png=False)
    T_BC = T_base_flange(kinematics.fk(q, flange_to_ee(cfg)), cfg) @ T_FC
    est = estimate_stem_line(frame, T_BC, cfg, None if args.seed is None else np.array(args.seed))
    s = est.summary()
    print(json.dumps(s, indent=1))
    if est.ok:
        print(f"\nroot {np.round(est.root_B, 4).tolist()} m, tilt {est.tilt_deg:.1f} deg, grasp point {np.round(est.grasp_B, 4).tolist()} m, "
              f"stem diameter ~{est.diameter_m * 1000:.1f} mm")
        print(f"root from: {est.root_source} ({est.extra['base_below_green_px']} px below the lowest green pixel, "
              f"{est.extra['root_off_line_mm']} mm off the straight stem line)")
        f1 = lambda x: "n/a" if x is None else f"{x:.1f}"  # noqa: E731
        print(f"checks: root (stem stereo) vs the same ray on the substrate (dense depth) {f1(est.root_vs_soil_mm)} mm; "
              f"row windows matched alone: {f1(est.disparity_resid_px)} px RMS = {f1(est.depth_resid_mm)} mm in depth")
    L, R = overlay(frame, est, T_BC)
    cv2.imwrite(str(out / "left.jpg"), L)
    if R is not None:
        cv2.imwrite(str(out / "right.jpg"), R)
    (out / "result.json").write_text(json.dumps({**s, "q": q.tolist(), "T_BC": T_BC.tolist(), "calibration": meta.get("calibration_id")}, indent=1))
    print("saved", out.relative_to(PROJECT_DIR))
    if args.frame:
        return 0
    try:
        if args.orbit_deg and est.ok:
            return orbit_check(args, cfg, robot, cam, T_FC, T_BC, est, out)
    finally:
        cam.close()
    return 0


def orbit_check(args, cfg, robot, cam, T_FC, T_BC, est, out) -> int:
    """Second view rotated about the vertical axis through the grasp point; compare the two stem lines."""
    import time

    from scipy.spatial.transform import Rotation

    from seedling import kinematics
    from seedling.calibration import T_base_flange
    from seedling.config import flange_to_ee
    from seedling.stemline import estimate_stem_line
    from seedling.transforms import inv_T, make_T
    from seedling.wrist import path_clearance

    F = flange_to_ee(cfg)
    q0 = robot.q()
    p = est.grasp_B
    Rz = make_T(Rotation.from_euler("z", args.orbit_deg, degrees=True).as_matrix())
    T_BC2 = make_T(np.eye(3), p) @ Rz @ make_T(np.eye(3), -p) @ T_BC
    q1 = kinematics.solve_ik_near(T_BC2 @ inv_T(T_FC) @ F, q0, F, float(cfg["robot"]["ik_joint_margin_rad"]))
    if q1 is None:
        print("second view not reachable")
        return 1
    z, part = path_clearance(q0, q1, T_FC, cfg)
    step = float(np.degrees(np.abs(q1 - q0)).max())
    print(f"\nsecond view: camera {args.orbit_deg:+.0f} deg around the stem, max joint change {step:.0f} deg, lowest part {part} z={z:.3f}")
    if step > 40 or z < float(est.soil_z or 0) + 0.10:
        print("refused: large joint change or too low")
        return 1
    robot.step_mode = not args.no_step
    r = robot.move_joints(q1, dynamics=args.dynamics, description=f"second view ({args.orbit_deg:+.0f} deg)")
    try:
        if not r.ok:
            print("motion failed:", r.reason)
            return 1
        time.sleep(0.8)
        f2 = cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        q = robot.q()
        f2.meta["q"] = q.tolist()
        f2.save(out / "frame_view2", save_png=False)
        T_BC2m = T_base_flange(kinematics.fk(q, F), cfg) @ T_FC
        e2 = estimate_stem_line(f2, T_BC2m, cfg)
        L, R = overlay(f2, e2, T_BC2m)
        cv2.imwrite(str(out / "left_view2.jpg"), L)
        if R is not None:
            cv2.imwrite(str(out / "right_view2.jpg"), R)
        print("view 2:", json.dumps(e2.summary()))
        if e2.ok:
            ang = float(np.degrees(np.arccos(np.clip(est.dir_B @ e2.dir_B, -1, 1))))
            dz = est.grasp_B[2] - e2.root_B[2]
            g2 = e2.root_B + e2.dir_B * dz / e2.dir_B[2]  # view-2 line at the height of view-1's grasp point
            res = {"root_diff_mm": np.round((e2.root_B - est.root_B) * 1000, 1).tolist(),
                   "line_at_grasp_height_diff_mm": np.round((g2 - est.grasp_B)[:2] * 1000, 1).tolist(),
                   "direction_diff_deg": round(ang, 1), "tilt_deg": [round(est.tilt_deg, 1), round(e2.tilt_deg, 1)]}
            print("\nCONSISTENCY between the two views (base frame):", json.dumps(res))
            (out / "orbit_check.json").write_text(json.dumps({**res, "view1": est.summary(), "view2": e2.summary()}, indent=1, default=float))
    finally:
        robot.move_joints(q0, dynamics=args.dynamics, description="back to the first view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
