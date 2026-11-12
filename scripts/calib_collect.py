#!/usr/bin/env python
"""Wrist-camera calibration, step 1: collect samples (camera on the robot, marker sheet fixed on the table).

Works with one marker (calibration/aruco_marker_A4.pdf, 100 mm) or a sheet of several markers (e.g. the lab's
"ZED / FRANKA fixed view calibration" page: DICT_4X4_50 ids 0-3, 40 mm). All visible markers are recorded.

Preparation
  1. Tape the sheet flat on the table near the tray. It must not move during the whole collection.
  2. Bring the robot (hand guiding or robot_jog.py) so the camera looks at the sheet from ~0.3-0.5 m.
     This start pose is also the lowest the camera will go.

What the script does
  - checks the markers are visible from the start pose
  - phase 1: --probe small moves (rotations about the flange, translations upward only) -> rough estimate
  - phase 2: --n poses orbiting the sheet (camera never lower nor closer than at the start), nearest-first order
  - every move: IK from the current joints (no branch jumps; a pose needing > --max-joint-step-deg on any joint
    is skipped), joint path checked for height, slow joint motion, settle, 3-frame capture
  - returns to the start pose at the end. Ctrl+C stops where the robot is (no return move).

    .venvs/pipeline/bin/python scripts/calib_collect.py --marker-mm 40

A second session under the working conditions (sheet on the raised tray next to the seedlings, oblique start
pose) to test / extend an existing calibration: no probe phase, the sheet is located with the current one.
    .venvs/pipeline/bin/python scripts/calib_collect.py --marker-mm 40 --use-calib --max-angle-deg 25 --dry-run
    .venvs/pipeline/bin/python scripts/calib_collect.py --marker-mm 40 --use-calib --max-angle-deg 25
    .venvs/pipeline/bin/python scripts/calib_eval.py calibration/sessions/wrist_<new>
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
    ap.add_argument("--marker-mm", type=float, required=True, help="measured side of the black squares")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="marker ids to use (default: all detected)")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--probe", type=int, default=6, help="small moves for the rough estimate")
    ap.add_argument("--n", type=int, default=18, help="orbit poses")
    ap.add_argument("--max-angle-deg", type=float, default=20.0)
    ap.add_argument("--dist-scale", type=float, nargs=2, default=[1.05, 1.25], help="orbit distance relative to the start")
    ap.add_argument("--roll-deg", type=float, default=10.0, help="orbit rotation about the optical axis")
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--dynamics", type=float, default=0.03, help="relative speed of every move (slow)")
    ap.add_argument("--max-joint-step-deg", type=float, default=30.0)
    ap.add_argument("--min-height-mm", type=float, default=150.0, help="gripper contact point must stay this high above the sheet")
    ap.add_argument("--no-step", action="store_true", help="do not ask before each move (operator at the e-stop)")
    ap.add_argument("--use-calib", action="store_true", help="skip the probe phase: locate the sheet with the current calibration")
    ap.add_argument("--dry-run", action="store_true", help="look from the start pose and plan every move; no motion, nothing saved")
    args = ap.parse_args()
    if args.dry_run and not args.use_calib:
        ap.error("--dry-run needs --use-calib (the probe phase needs real moves)")
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import (T_base_flange, detect_markers, flange_pose, load_T_FC, now_id, order_poses, orbit_poses,
                                      perturbation_poses, solve_eye_in_hand_markers, view_elevation_deg)
    from seedling.camera import ZedCamera
    from seedling.config import flange_to_ee
    from seedling.robot import FrankyRobot, UnsafeTarget
    from seedling.transforms import apply_T, inv_T

    F_T_EE = flange_to_ee(cfg)
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    cam = ZedCamera(cfg)
    out = PROJECT_DIR / ("scratch" if args.dry_run else "calibration") / "sessions" / f"wrist_{now_id()}"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "collect_args.json").write_text(json.dumps({**vars(args), "flange_to_ee": F_T_EE.tolist()}, indent=1))
    L = args.marker_mm / 1000.0
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    samples: list[dict] = []
    floor = {"contact": None, "camera": None}  # lowest allowed z (base) for the contact point and the camera
    T_FC_rough = None

    def record(tag: str) -> bool:
        time.sleep(args.settle_s)
        f = cam.grab(n_average=3)
        obs = detect_markers(f, L, args.dict, args.ids)
        T_BE = robot.T_BE()
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        if not obs:
            print(f"  {tag}: no marker detected (skipped)")
            if not args.dry_run:
                cv2.imwrite(str(out / f"{tag}_nomarker.jpg"), cv2.resize(img, (960, 540)))
            return False
        for o in obs:
            cv2.aruco.drawDetectedMarkers(img, [o.corners_px.reshape(1, 4, 2).astype(np.float32)], np.array([[o.marker_id]]))
            cv2.drawFrameAxes(img, f.intr.K(), None, cv2.Rodrigues(o.T_CM[:3, :3])[0], o.T_CM[:3, 3], L)
        if not args.dry_run:
            cv2.imwrite(str(out / f"{tag}.jpg"), cv2.resize(img, (960, 540)))
        rec = {"tag": tag, "T_BE": T_BE.tolist(), "T_BF": T_base_flange(T_BE, cfg).tolist(), "q": robot.q().tolist(),
               "K": f.intr.K().tolist(), "marker_length_m": L, "camera_meta": f.meta,
               "markers": {str(o.marker_id): {"corners_px": o.corners_px.tolist(), "T_CM": o.T_CM.tolist(), "reproj_px": o.reproj_px,
                                              "depth_center_C": None if o.depth_center_C is None else o.depth_center_C.tolist()}
                           for o in obs}}
        samples.append(rec)
        if not args.dry_run:
            with (out / "samples.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        dz = [1000 * (o.depth_center_C[2] - o.T_CM[2, 3]) for o in obs if o.depth_center_C is not None]
        print(f"  {tag}: ids {[o.marker_id for o in obs]}, distance {np.mean([np.linalg.norm(o.T_CM[:3, 3]) for o in obs]):.3f} m, "
              f"reproj max {max(o.reproj_px for o in obs):.2f} px" + (f", stereo-PnP depth {np.median(dz):+.1f} mm" if dz else ""))
        return True

    def path_ok(q_from: np.ndarray, q_to: np.ndarray) -> str | None:
        if np.degrees(np.abs(q_to - q_from)).max() > args.max_joint_step_deg:
            return f"needs {np.degrees(np.abs(q_to - q_from)).max():.0f} deg on one joint"
        def heights(q):
            T_BE = kinematics.fk(q, F_T_EE)
            c = (T_BE @ inv_T(F_T_EE) @ T_FC_rough)[2, 3] if T_FC_rough is not None else np.inf
            return apply_T(T_BE @ robot.T_EG, np.zeros(3))[2], c

        # never below the floor; if already below it (e.g. after a probe rotation), never lower than now
        g0, c0 = heights(q_from)
        g_floor, c_floor = min(floor["contact"], g0) - 1e-3, (min(floor["camera"], c0) - 1e-3) if T_FC_rough is not None else -np.inf
        for s in np.linspace(0, 1, 21):
            g, c = heights(q_from + s * (q_to - q_from))
            if g < g_floor:
                return "gripper would go too low"
            if c < c_floor:
                return "camera would go too low"
        return None

    sim = {"q": None, "planned": []}  # dry run: joints the robot would be at

    def go(T_BF: np.ndarray, desc: str) -> bool:
        q_now = sim["q"] if args.dry_run else robot.q()
        q = kinematics.solve_ik_near(T_BF @ F_T_EE, q_now, F_T_EE, margin)
        if q is None:
            print(f"  {desc}: skipped (no IK near the current joints)")
            return False
        why = path_ok(q_now, q)
        if why:
            print(f"  {desc}: skipped ({why})")
            return False
        if args.dry_run:
            T_BC = T_BF @ T_FC_rough
            print(f"  {desc}: ok, max joint change {np.degrees(np.abs(q - q_now)).max():.0f} deg, camera looking down "
                  f"{view_elevation_deg(T_BC):.0f} deg at z={T_BC[2, 3]:.3f}")
            sim["q"] = q
            sim["planned"].append(T_BC)
            return True
        try:
            r = robot.move_joints(q, dynamics=args.dynamics, description=desc)
        except UnsafeTarget as exc:
            print(f"  {desc}: skipped ({exc})")
            return False
        if not r.ok:
            raise RuntimeError(f"motion failed: {r.reason} (robot left where it stopped; check Desk, recover manually)")
        return True

    q0 = robot.q()
    if args.use_calib:
        load_T_FC(cfg)  # sets the joint offsets of the current calibration before any pose is computed
    T_BF0 = flange_pose(robot)
    floor["contact"] = float(robot.T_BG()[2, 3]) - 0.01
    finished = False
    try:
        print("start pose:")
        if not record("p000_start"):
            print("\nThe camera does not see a marker. Move the robot so the sheet is in view (~0.3-0.5 m) and retry.")
            return 1
        if args.use_calib:
            T_FC_rough, _meta = load_T_FC(cfg)
            s0 = samples[0]
            T_BC0 = T_BF0 @ T_FC_rough
            centre = np.mean([apply_T(T_BC0, np.array(v["T_CM"])[:3, 3]) for v in s0["markers"].values()], 0)
            print(f"sheet located with the current calibration ({_meta.get('calibration_id')})")
        else:
            print(f"\nphase 1: {args.probe} small moves (upward / rotations only)")
            for i, T in enumerate(order_poses(T_BF0, perturbation_poses(T_BF0, args.probe, up_only=True)), 1):
                if go(T, f"probe {i}/{args.probe}"):
                    record(f"p{i:03d}_probe")
            if len(samples) < 5:
                print("too few detections for a rough estimate; check lighting / marker size")
                return 1
            rough = solve_eye_in_hand_markers([np.array(s["T_BF"]) for s in samples],
                                              [{int(k): np.array(v["corners_px"]) for k, v in s["markers"].items()} for s in samples],
                                              [np.array(s["K"]) for s in samples], L)
            T_FC_rough = rough["T_FC"]
            centre = np.mean([T[:3, 3] for T in rough["T_BM"].values()], 0)
            print(f"rough estimate fit rms {rough['fit_rms_px']:.2f} px")
        cam_z0 = float((T_BF0 @ T_FC_rough)[2, 3])
        floor["camera"] = cam_z0 - 0.01
        floor["contact"] = max(floor["contact"], float(centre[2]) + args.min_height_mm / 1000.0)
        print(f"sheet centre at {np.round(centre, 3).tolist()} m (base), camera {cam_z0 - centre[2]:.3f} m above it, "
              f"looking down {view_elevation_deg(T_BF0 @ T_FC_rough):.0f} deg")
        if not args.use_calib:
            print("\nback to the start pose before phase 2")
            if np.degrees(np.abs(q0 - robot.q())).max() > args.max_joint_step_deg:
                raise RuntimeError("start pose too far away; not moving")
            r = robot.move_joints(q0, dynamics=args.dynamics, description="back to the start pose")
            if not r.ok:
                raise RuntimeError(f"motion failed: {r.reason}")
        sim["q"] = q0
        print(f"\nphase 2: {args.n} orbit poses around the sheet")
        poses = orbit_poses(T_BF0, T_FC_rough, centre, 3 * args.n, args.max_angle_deg, roll_deg=args.roll_deg,
                            dist_scale=tuple(args.dist_scale), min_height_m=cam_z0 - float(centre[2]))
        main = order_poses(T_BF0, poses[:args.n])  # the planned set; the rest only replaces skipped ones
        got = 0
        for i, T in enumerate(main + order_poses(main[-1], poses[args.n:]), 1):
            if got >= args.n:
                break
            if go(T, f"orbit candidate {i} ({got}/{args.n} collected)") and (args.dry_run or record(f"p{100 + i:03d}_orbit")):
                got += 1
        if args.dry_run:
            el = [view_elevation_deg(T) for T in sim["planned"]]
            if el:
                print(f"\nDRY RUN: {len(el)} poses planned, camera looking down {min(el):.0f}-{max(el):.0f} deg, "
                      f"lowest camera z={min(T[2, 3] for T in sim['planned']):.3f} (start {cam_z0:.3f}); no motion")
            return 0
        finished = True
    except KeyboardInterrupt:
        print("\ninterrupted: robot left where it is")
    finally:
        if finished:
            q_now = robot.q()
            if np.degrees(np.abs(q0 - q_now)).max() <= 45:
                robot.move_joints(q0, dynamics=args.dynamics, description="back to the start pose")
            else:
                print("not returning to the start automatically (large joint change)")
        cam.close()
    if args.dry_run:
        return 0
    print(f"\n{len(samples)} samples saved in {out}")
    print(f"next: .venvs/pipeline/bin/python scripts/calib_solve.py {out.relative_to(PROJECT_DIR)} --save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
