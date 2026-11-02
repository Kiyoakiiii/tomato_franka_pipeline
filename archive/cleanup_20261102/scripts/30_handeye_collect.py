#!/usr/bin/env python
"""Collect eye-to-hand calibration samples: robot EE pose + ArUco marker pose seen by the fixed ZED.

Preparation: print an ArUco marker (default DICT_4X4_50 id 0, side --marker-mm, measure the printed
size!) and fix it rigidly to the hand so it faces the camera. Move the robot so the marker is
clearly visible in the middle of the tray area; that pose is the centre of the automatic pattern.

    --auto N    generate N poses around the current pose (+-translation, +-tilt), each move confirmed
    (default)   manual: move the robot yourself (guiding), press Enter to record, q to finish

Output: calibration/sessions/<id>/samples.jsonl + images. Then run 31_handeye_solve.py.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def auto_poses(T0: np.ndarray, n: int, trans_m: float, tilt_deg: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        T = T0.copy()
        T[:3, 3] += rng.uniform(-trans_m, trans_m, 3) * np.array([1, 1, 0.6])
        rv = np.radians(rng.uniform(-tilt_deg, tilt_deg, 3))
        T[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(rv).as_matrix()  # rotate about the EE's own axes
        out.append(T)
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--auto", type=int, default=0)
    ap.add_argument("--marker-mm", type=float, required=True, help="measured printed marker side length")
    ap.add_argument("--marker-id", type=int, default=None)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--trans-mm", type=float, default=70.0)
    ap.add_argument("--tilt-deg", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--no-step", action="store_true", help="do not ask before each auto move (only after a dry check)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import detect_marker, now_id
    from seedling.camera import ZedCamera
    from seedling.robot import FrankyRobot, UnsafeTarget

    cam = ZedCamera(cfg)
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    sid = now_id()
    out = PROJECT_DIR / "calibration" / "sessions" / sid
    out.mkdir(parents=True, exist_ok=True)
    (out / "collect_args.json").write_text(json.dumps(vars(args), indent=1))
    L = args.marker_mm / 1000.0
    n_ok = 0

    def record(tag: str) -> bool:
        nonlocal n_ok
        time.sleep(args.settle_s)
        f = cam.grab(n_average=int(cfg["camera"].get("average_frames", 3)))
        obs = detect_marker(f, L, args.dict, args.marker_id)
        T_BE = robot.T_BE()
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        if obs is None:
            print(f"  {tag}: marker NOT detected")
            cv2.imwrite(str(out / f"{tag}_nomarker.jpg"), img)
            return False
        cv2.aruco.drawDetectedMarkers(img, [obs.corners_px.reshape(1, 4, 2).astype(np.float32)])
        cv2.drawFrameAxes(img, f.intr.K(), None, cv2.Rodrigues(obs.T_CM[:3, :3])[0], obs.T_CM[:3, 3], L)
        cv2.imwrite(str(out / f"{tag}.jpg"), img)
        rec = {"tag": tag, "T_BE": T_BE.tolist(), "q": robot.q().tolist(), "T_CM": obs.T_CM.tolist(), "reproj_px": obs.reproj_px,
               "marker_id": obs.marker_id, "depth_center_C": None if obs.depth_center_C is None else obs.depth_center_C.tolist(),
               "camera_meta": f.meta, "intr": [f.intr.fx, f.intr.fy, f.intr.cx, f.intr.cy]}
        with (out / "samples.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        n_ok += 1
        pnp_z = obs.T_CM[2, 3]
        dz = "" if obs.depth_center_C is None else f" stereo-PnP z diff {(obs.depth_center_C[2]-pnp_z)*1000:+.1f} mm"
        print(f"  {tag}: ok (reproj {obs.reproj_px:.2f}px, marker at {np.round(obs.T_CM[:3,3],3).tolist()} m){dz}")
        return True

    try:
        if args.auto:
            T0 = robot.T_BE()
            poses = auto_poses(T0, args.auto, args.trans_mm / 1000.0, args.tilt_deg)
            record("p000_center")
            for i, T in enumerate(poses, 1):
                try:
                    r = robot.move_to_pose(T, dynamics=float(cfg["robot"]["dynamics"]), force_limit_n=float(cfg["task"]["guard_force_limit_n"]),
                                           description=f"calib pose {i}/{len(poses)}")
                except UnsafeTarget as exc:
                    print(f"  pose {i}: skipped ({exc})")
                    continue
                if not r.ok:
                    print(f"  pose {i}: motion failed: {r.reason}; stopping")
                    break
                record(f"p{i:03d}")
            robot.move_to_pose(T0, dynamics=float(cfg["robot"]["dynamics"]), description="back to centre pose")
        else:
            i = 0
            while True:
                ans = input("移动机械臂到新姿态（标记朝向相机），Enter 记录，q 结束: ").strip().lower()
                if ans == "q":
                    break
                i += 1
                record(f"m{i:03d}")
    finally:
        cam.close()
    print(f"\n{n_ok} samples in {out}\nnext: .venvs/pipeline/bin/python scripts/31_handeye_solve.py {out.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
