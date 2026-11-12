#!/usr/bin/env python
"""Wrist-camera calibration, independent validation samples: the marker sheet seen from working-style views
(oblique, 50-70 deg down, the wrist turned far: joint 7 in --q7-range), poses that the calibration samples do
not contain. Samples are saved like calib_collect.py, so calib_eval.py tests the current
calibration on them without refitting.

The sheet must lie where the calibration found it (T_base_marker_at_calibration of T_flange_cam.yaml).

    .venvs/pipeline/bin/python scripts/calib_validate.py --dry-run
    .venvs/pipeline/bin/python scripts/calib_validate.py --no-step
    .venvs/pipeline/bin/python scripts/calib_eval.py calibration/sessions/wristval_<id>
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
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--elevation", type=float, nargs=2, default=[50, 70])
    ap.add_argument("--distance", type=float, nargs=2, default=[0.40, 0.55])
    ap.add_argument("--azimuth", type=float, nargs=2, default=[-45, 45], help="relative to base->sheet")
    ap.add_argument("--q7-range", type=float, nargs=2, default=[1.5, 2.75], help="joint 7 of most poses (rad)")
    ap.add_argument("--q7-share", type=float, default=0.7, help="fraction of the poses with joint 7 in --q7-range")
    ap.add_argument("--min-height-mm", type=float, default=150.0, help="gripper contact point above the sheet")
    ap.add_argument("--max-joint-step-deg", type=float, default=45.0)
    ap.add_argument("--dynamics", type=float, default=0.03)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-step", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import T_base_flange, detect_markers, load_T_FC, now_id
    from seedling.config import flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T
    from seedling.wrist import look_from, lowest_body_z, path_clearance

    T_FC, meta = load_T_FC(cfg)
    refs = {k: np.array(v) for k, v in (meta.get("T_base_marker_at_calibration") or {}).items()}
    if not refs:
        raise SystemExit("calibration has no T_base_marker_at_calibration")
    centre = np.mean([T[:3, 3] for T in refs.values()], axis=0)
    L = float(meta.get("marker_length_m", 0.04))
    F = flange_to_ee(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    robot = FrankyRobot(cfg, step_mode=not args.no_step)
    q0 = robot.q()
    floor = float(centre[2]) + args.min_height_mm / 1000.0

    # candidates: random oblique views, every roll in 30 deg steps; keep the IK solution's joint 7
    rng = np.random.default_rng(args.seed)
    radial = np.arctan2(centre[1], centre[0])
    cands = []
    for _ in range(40 * args.n):
        el, d = rng.uniform(*args.elevation), rng.uniform(*args.distance)
        az = radial + np.radians(rng.uniform(*args.azimuth))
        roll = float(rng.choice(np.arange(-180, 180, 30)))
        T_BC = look_from(centre, az, el, d, roll)
        T_BC = T_BC @ np.eye(4)
        q = kinematics.solve_ik(T_BC @ inv_T(T_FC) @ F, q0, F, margin)
        if q is None:
            continue
        if lowest_body_z(q, T_FC, cfg)[0] < floor:
            continue
        cands.append((T_BC, q, el, d, roll))
    hi = [c for c in cands if args.q7_range[0] <= c[1][6] <= args.q7_range[1]]
    lo = [c for c in cands if not args.q7_range[0] <= c[1][6] <= args.q7_range[1]]
    n_hi = int(round(args.q7_share * args.n))
    lo = [c for c in lo if c[1][6] >= q0[6] - 0.6]
    pick = [hi[i] for i in rng.permutation(len(hi))[:n_hi]] + [lo[i] for i in rng.permutation(len(lo))[:args.n - n_hi]]
    # order: sweep joint 7 upward from the start (small steps), each move re-solved near the current joints
    # (joint 7 below the start's is already covered by the calibration sessions: skip those far below)
    order = sorted([c for c in pick if c[1][6] >= q0[6] - 0.6], key=lambda c: c[1][6])
    plan, q = [], q0
    for T_BC, qc, el, d, roll in order:
        qn = kinematics.solve_ik_near(T_BC @ inv_T(T_FC) @ F, q, F, margin)
        qn = qc if qn is None else qn
        dq = np.degrees(np.abs(qn - q))
        z, part = path_clearance(q, qn, T_FC, cfg)
        # joints 1-6 move the arm: <= max step; joint 7 only spins the hand: <= 90 deg
        ok = dq[:6].max() <= args.max_joint_step_deg and dq[6] <= 90 and z >= min(floor, lowest_body_z(q, T_FC, cfg)[0] - 1e-3)
        print(f"  {'ok  ' if ok else 'skip'} elev {el:4.1f} deg, dist {d:.2f} m, roll {roll:+5.0f}, joint 7 {qn[6]:+.2f} rad, "
              f"max change joints 1-6 {dq[:6].max():3.0f} deg, joint 7 {dq[6]:3.0f} deg, lowest {part} z={z:.3f}")
        if ok:
            plan.append((qn, el, d, roll))
            q = qn
    q7 = [p[0][6] for p in plan]
    print(f"\n{len(plan)} poses; joint 7 {min(q7):+.2f}..{max(q7):+.2f} rad ({sum(args.q7_range[0] <= v <= args.q7_range[1] for v in q7)} in "
          f"{args.q7_range}); elevation {min(p[1] for p in plan):.0f}-{max(p[1] for p in plan):.0f} deg; return to the start "
          f"{np.degrees(np.abs(plan[-1][0] - q0)).max():.0f} deg")
    if args.dry_run:
        return 0

    from seedling.camera import ZedCamera

    out = PROJECT_DIR / "calibration" / "sessions" / f"wristval_{now_id()}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "collect_args.json").write_text(json.dumps({**vars(args), "flange_to_ee": F.tolist(), "validation_of": meta.get("calibration_id")}, indent=1))
    cam = ZedCamera(cfg)
    n = 0
    try:
        for i, (qn, el, d, roll) in enumerate(plan, 1):
            r = robot.move_joints(qn, dynamics=args.dynamics, description=f"validation pose {i}")
            if not r.ok:
                print("motion failed:", r.reason)
                break
            time.sleep(1.0)
            f = cam.grab(n_average=3)
            obs = detect_markers(f, L, args.dict, None)
            tag = f"v{i:03d}"
            img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / f"{tag}.jpg"), cv2.resize(img, (960, 540)))
            if not obs:
                print(f"  {tag}: no marker detected")
                continue
            T_BE = robot.T_BE()
            rec = {"tag": tag, "T_BE": T_BE.tolist(), "T_BF": T_base_flange(T_BE, cfg).tolist(), "q": robot.q().tolist(),
                   "K": f.intr.K().tolist(), "marker_length_m": L, "camera_meta": f.meta,
                   "markers": {str(o.marker_id): {"corners_px": o.corners_px.tolist(), "T_CM": o.T_CM.tolist(), "reproj_px": o.reproj_px,
                                                  "depth_center_C": None if o.depth_center_C is None else o.depth_center_C.tolist()}
                               for o in obs}}
            with (out / "samples.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            print(f"  {tag}: ids {[o.marker_id for o in obs]}, elev {el:.0f} deg, joint 7 {rec['q'][6]:+.2f} rad, "
                  f"distance {np.mean([np.linalg.norm(o.T_CM[:3, 3]) for o in obs]):.3f} m", flush=True)
    finally:
        cam.close()
        dq = np.degrees(np.abs(robot.q() - q0))
        if dq[:6].max() <= 60 and dq[6] <= 180:
            robot.move_joints(q0, dynamics=args.dynamics, description="back to the start pose")
    print(f"\n{n} validation samples in {out.relative_to(PROJECT_DIR)}")
    print(f"next: .venvs/pipeline/bin/python scripts/calib_eval.py {out.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
