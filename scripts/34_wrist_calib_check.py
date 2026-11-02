#!/usr/bin/env python
"""Wrist-camera calibration, step 3: quick check (use before each session, ~10 s).

Looks at the marker(s) on the table and reports where the calibration puts them in the robot base frame.
If the sheet has not moved since calibration, this is compared with the positions stored then.
With --point, the gripper is moved (gripper pointing down, every move confirmed) to --height-mm above
the centre of --marker-id so you can see by eye whether the fingertip centre is over the marker centre.

    .venvs/pipeline/bin/python scripts/34_wrist_calib_check.py --marker-mm 40 [--point --marker-id 0 --height-mm 30]
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--marker-mm", type=float, required=True)
    ap.add_argument("--marker-id", type=int, default=None, help="marker for --point (default: lowest id seen)")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--point", action="store_true", help="move the gripper above the marker centre")
    ap.add_argument("--height-mm", type=float, default=30.0)
    ap.add_argument("--camera-clearance-mm", type=float, default=80.0,
                    help="the wrist camera hangs below the fingertips: its optical centre must stay this high above the marker")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import camera_pose_in_base, detect_markers, load_T_FC
    from seedling.camera import ZedCamera
    from seedling.robot import FrankyRobot
    from seedling.transforms import apply_T

    T_FC, meta = load_T_FC(cfg)
    robot = FrankyRobot(cfg, step_mode=True)
    cam = ZedCamera(cfg)
    try:
        f = cam.grab(n_average=3)
        T_BC = camera_pose_in_base(cfg, robot)
    finally:
        cam.close()
    obs = detect_markers(f, args.marker_mm / 1000.0, args.dict)
    if not obs:
        print("no marker visible from this pose")
        return 1
    ref = meta.get("T_base_marker_at_calibration") or {}
    if ref and not isinstance(ref, dict):  # single-marker calibration files
        ref = {"0": ref}
    pos = {}
    for o in obs:
        m = apply_T(T_BC @ o.T_CM, np.zeros(3))
        pos[o.marker_id] = m
        line = (f"marker {o.marker_id}: centre in base {np.round(m, 4).tolist()} m "
                f"(seen from {np.linalg.norm(o.T_CM[:3, 3]):.3f} m, reproj {o.reproj_px:.2f} px)")
        if str(o.marker_id) in ref:
            d = (m - np.array(ref[str(o.marker_id)])[:3, 3]) * 1000
            line += f"  vs calibration {np.round(d, 1).tolist()} mm (|d| = {np.linalg.norm(d):.1f})"
        print(line)
    print(f"calibration {meta.get('calibration_id')}; differences are only meaningful if the sheet has not moved")
    # stereo bias check: dense depth at each marker vs its PnP distance -> disparity offset (camera.disparity_offset_px)
    fB = f.intr.fx * float(f.meta.get("baseline_m", 0.12))
    cur = float(f.meta.get("disparity_offset_px", 0.0))
    offs = [cur + fB / o.depth_center_C[2] - fB / o.T_CM[2, 3] for o in obs if o.depth_center_C is not None]
    if offs:
        print(f"stereo disparity offset estimate {np.median(offs):.1f} px (config uses {cur:.1f}; "
              f"update camera.disparity_offset_px if it differs by more than ~3 px)")
    mid = args.marker_id if args.marker_id is not None else min(pos)
    if mid not in pos:
        print(f"marker {mid} not visible")
        return 1
    m = pos[mid]
    if not args.point:
        return 0
    T = robot.T_BG()
    if T[2, 2] > -0.85:  # gripper not pointing down: use the home orientation
        T[:3, :3] = (kinematics.fk(np.array(cfg["robot"]["home_joints_rad"]), robot.F_T_EE) @ robot.T_EG)[:3, :3]
    target = T.copy()
    target[:3, 3] = m + [0, 0, args.height_mm / 1000.0]
    # camera height at that pose (contact point -> EE -> flange -> camera)
    cam_z = (target @ np.linalg.inv(robot.T_EG) @ np.linalg.inv(robot.F_T_EE) @ T_FC)[2, 3]
    lift = max(0.0, m[2] + args.camera_clearance_mm / 1000.0 - cam_z)
    if lift > 0:
        target[2, 3] += lift
        print(f"camera clearance: fingertips stop {(target[2, 3] - m[2]) * 1000:.0f} mm above the marker instead of {args.height_mm:.0f} mm")
    above = target.copy()
    above[2, 3] += 0.05
    q = kinematics.solve_ik_near(above @ np.linalg.inv(robot.T_EG), robot.q(), robot.F_T_EE, float(cfg["robot"]["ik_joint_margin_rad"]))
    if q is None or np.degrees(np.abs(q - robot.q())).max() > 45:
        print("cannot reach a pose above the marker without a large joint change; move the robot closer first")
        return 1
    robot.move_joints(q, dynamics=float(cfg["robot"]["dynamics"]), description=f"{(above[2, 3] - m[2]) * 1000:.0f} mm above the marker")
    r = robot.move_contact_to(target, dynamics=float(cfg["robot"]["fine_dynamics"]), force_limit_n=10.0,
                              description=f"descend to {(target[2, 3] - m[2]) * 1000:.0f} mm above the marker centre")
    print(r)
    print("看一下：夹爪两指的中心是否在标记中心正上方？偏差大概多少毫米、往哪个方向？")
    input("按 Enter 抬起夹爪 ...")
    robot.move_contact_to(above, dynamics=float(cfg["robot"]["fine_dynamics"]), description="lift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
