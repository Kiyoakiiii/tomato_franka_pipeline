#!/usr/bin/env python
"""Compare the vision stem estimate with an independent reference (proposal section 6).

For each target: observe -> best grasp candidate (estimate) -> robot moves to the pre-grasp and then
to the estimated contact pose with the gripper OPEN (no closing) -> operator jogs the gripper until the
stem is truly centred between the pads -> 'done'. The taught contact position is the reference.
Error = estimate - reference, projected on the candidate's c (closing), a (approach), s (stem) axes.

The reference is only as good as the operator's centring (state its uncertainty in the notes); this
measures the full chain (segmentation + depth + extrinsics + TCP offset), which is what the gripper sees.

    .venvs/pipeline/bin/python scripts/41_localization_check.py --plant-id P03 --role held_out_validation
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--plant-id", default="")
    ap.add_argument("--role", default="held_out_validation", choices=["calibration_fit", "held_out_validation"])
    ap.add_argument("--reference-id", default="taught_centre_v1")
    ap.add_argument("--standoff-mm", type=float, default=0.0, help="stop this far before the estimated pose (along -a) first")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import load_T_BC
    from seedling.camera import ZedCamera
    from seedling.gripper import make_gripper
    from seedling.jog import jog_session
    from seedling.perception import observe, save_observation
    from seedling.robot import FrankyRobot, _console_confirm
    from seedling.segmentation import StemSegmenter
    from seedling.trial_log import Session

    T_BC, calib = load_T_BC(cfg)
    cam = ZedCamera(cfg)
    seg = StemSegmenter(cfg)
    robot = FrankyRobot(cfg, step_mode=True)
    grip = make_gripper(cfg, sim=False, step_mode=True, confirm_fn=_console_confirm)
    session = Session(PROJECT_DIR / "experiments" / "localization", args.role, cfg, vars(args))
    fine = float(cfg["robot"]["fine_dynamics"])
    k = 0
    try:
        while True:
            k += 1
            if input(f"\n目标 {k}: 摆好幼苗后 Enter 观察（q 结束）: ").strip().lower() == "q":
                break
            robot.move_joints(np.array(cfg["robot"]["home_joints_rad"]), dynamics=float(cfg["robot"]["dynamics"]), description="home")
            obs = observe(cam, seg, cfg, T_BC, q_seed=robot.q())
            sid = f"s{k:03d}"
            save_observation(obs, session.dir, sid, cfg, T_BC)
            best = obs.best()
            row = {"sample_id": sid, "session_id": session.dir.name, "plant_id": args.plant_id, "frame_id": f"{sid}_frame.npz",
                   "model_id": cfg["perception"]["model"], "calibration_id": calib.get("calibration_id", ""), "reference_id": args.reference_id,
                   "reference_role": args.role}
            if best is None:
                row.update(rejected=1, rejection_reason=obs.rejection_summary())
                session.append_csv("localization_checks.csv", row)
                print("  rejected:", row["rejection_reason"])
                continue
            segm = next(s for g in obs.geoms for s in g.segments if g.instance == best.instance and s.seg_index == best.seg_index)
            row.update(estimate_x_m=round(best.p_B[0], 5), estimate_y_m=round(best.p_B[1], 5), estimate_z_m=round(best.p_B[2], 5),
                       valid_depth_fraction=round(segm.valid_fraction, 3), fit_residual_mm=round(segm.rms_m * 1000, 3), rejected=0)
            grip.open()
            robot.move_contact_to(best.T_BG_pre, dynamics=float(cfg["robot"]["dynamics"]), description="pre-grasp (gripper open)")
            T_est = best.T_BG.copy()
            T_est[:3, 3] -= T_est[:3, 2] * args.standoff_mm / 1000.0
            r = robot.move_contact_to(T_est, dynamics=fine, force_limit_n=10.0, description="estimated contact pose (no closing)")
            if not r.ok:
                print("  approach stopped:", r.reason)
            print("  现在把夹爪中心调到真实茎中心（c=闭合方向，a=接近方向，s=沿茎）。")
            T_ref = jog_session(robot, fine, prompt="centre on stem")
            if T_ref is None:
                row["notes"] = "reference not taught"
            else:
                e = best.p_B - T_ref[:3, 3]
                R = best.T_BG[:3, :3]
                ec = e @ R[:, 1] * 1000
                ea = e @ R[:, 2] * 1000
                es = e @ R[:, 0] * 1000
                row.update(reference_x_m=round(T_ref[0, 3], 5), reference_y_m=round(T_ref[1, 3], 5), reference_z_m=round(T_ref[2, 3], 5),
                           error_c_mm=round(ec, 2), error_a_mm=round(ea, 2), error_s_mm=round(es, 2))
                print(f"  error (estimate - reference): c={ec:+.1f}  a={ea:+.1f}  s={es:+.1f} mm")
                row["notes"] = input("  备注（参照不确定度等，可空）: ").strip()
            T_back = robot.T_BG()
            T_back[:3, 3] -= T_back[:3, 2] * float(cfg["grasp"]["pregrasp_distance_m"])
            robot.move_contact_to(T_back, dynamics=fine, description="retreat")
            session.append_csv("localization_checks.csv", row)
    finally:
        cam.close()
    print(f"results: {session.dir / 'localization_checks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
