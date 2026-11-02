#!/usr/bin/env python
"""Run end-to-end transplanting trials (observe -> grasp -> extract -> transport -> place).

Offline, no hardware (synthetic scene + simulated robot/gripper):
    .venvs/pipeline/bin/python scripts/50_run_transplant.py --sim --trials 3

Real hardware, every motion waits for Enter (recommended until the flow is trusted):
    .venvs/pipeline/bin/python scripts/50_run_transplant.py --plant-id P01 --step --ask
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import sys

from _common import base_parser, config_from_args

from seedling.trial_log import Session


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--sim", action="store_true", help="synthetic camera + simulated robot and gripper")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--sim-stems", type=int, default=2)
    ap.add_argument("--sim-extraction-force", type=float, default=None, help="override simulated rooting force (N)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--plant-id", default="")
    ap.add_argument("--condition-id", default="")
    ap.add_argument("--setting-id", default="")
    ap.add_argument("--step", action="store_true", help="ask for Enter before every robot/gripper motion")
    ap.add_argument("--ask", action="store_true", help="ask the operator for placement/damage labels after each trial")
    ap.add_argument("--no-reobserve", action="store_true")
    ap.add_argument("--out", default="experiments/runs")
    args = ap.parse_args()
    if args.no_reobserve:
        args.set.append("task.reobserve_before_acquire=false")
    cfg = config_from_args(args)

    from seedling.config import project_path

    if args.sim:
        from seedling.gripper import SimGripper
        from seedling.robot import SimRobot
        from seedling.synthetic import GTSegmenter, SyntheticCamera, default_sim_camera, default_world

        cfg["grasp"]["table_z_m"] = 0.05  # synthetic tray top
        world = default_world(args.sim_seed, args.sim_stems)
        if args.sim_extraction_force is not None:
            for s in world.stems:
                s.extraction_force_n = args.sim_extraction_force
        world.open_width = float(cfg["gripper"]["open_width_m"])
        world.empty_width = float(cfg["gripper"]["empty_close_width_m"])
        T_BC, intr = default_sim_camera()
        camera = SyntheticCamera(world, T_BC, intr)
        segmenter = GTSegmenter()
        robot = SimRobot(cfg, step_mode=args.step, world=world)
        gripper = SimGripper(cfg, world=world)
        calib_id, model_id = "synthetic", "ground_truth_masks"
    else:
        from seedling.calibration import load_T_BC
        from seedling.camera import ZedCamera
        from seedling.gripper import make_gripper
        from seedling.robot import FrankyRobot, _console_confirm
        from seedling.segmentation import StemSegmenter

        T_BC, calib = load_T_BC(cfg)
        calib_id = str(calib.get("calibration_id", cfg["camera"]["T_base_cam_file"]))
        model_id = str(project_path(cfg["perception"]["model"]).name)
        segmenter = StemSegmenter(cfg)
        camera = ZedCamera(cfg)
        robot = FrankyRobot(cfg, step_mode=args.step)
        gripper = make_gripper(cfg, sim=False, step_mode=args.step, confirm_fn=_console_confirm)

    from seedling.task import TransplantTask, TrialAbort

    session = Session(project_path(args.out), "sim" if args.sim else "real", cfg,
                      {"argv": sys.argv, "calibration_id": calib_id, "model_id": model_id, "T_BC": T_BC.tolist()})
    print(f"session: {session.dir}")
    task = TransplantTask(cfg, camera, segmenter, robot, gripper, T_BC, session, ask_operator=args.ask,
                          ids={"condition_id": args.condition_id, "setting_id": args.setting_id, "model_id": model_id, "calibration_id": calib_id})
    rows = []
    try:
        for i in range(args.trials):
            tid = f"trial_{i+1:03d}"
            if args.sim and i > 0:  # fresh synthetic tray for each trial
                world.stems = default_world(args.sim_seed + i, args.sim_stems).stems
                if args.sim_extraction_force is not None:
                    for s in world.stems:
                        s.extraction_force_n = args.sim_extraction_force
            print(f"\n=== {tid} ===")
            rows.append(task.run_trial(tid, plant_id=args.plant_id or (f"sim_{i+1}" if args.sim else "")))
    except (KeyboardInterrupt, TrialAbort) as exc:
        print(f"\n中止: {exc}")
    finally:
        for dev in (gripper, robot, camera):
            try:
                dev.close()
            except Exception:
                pass
    ok = sum(int(r.get("end_to_end_success") or 0) for r in rows)
    print(f"\n{len(rows)} trials, end-to-end success {ok}/{len(rows)}. Results: {session.dir / 'end_to_end_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
