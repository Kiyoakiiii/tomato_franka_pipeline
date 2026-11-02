#!/usr/bin/env python
"""Pick seedlings cell by cell from a taught tray and place them in the receiving cup.

Modes:  blind      grasp at the taught cell position, stem assumed upright
        vision     per-cell correction (no trained model; see seedling/tray.py), fallback per tray.vision.fallback
        alternate  blind/vision assigned to cells in a balanced random order (comparison experiment)

Real hardware (confirm every motion at first):
    .venvs/pipeline/bin/python scripts/55_run_tray.py --cells A1,A2 --mode blind --step --ask
Simulation (synthetic tray, tilted seedlings):
    .venvs/pipeline/bin/python scripts/55_run_tray.py --sim --mode vision --sim-tilt 20
    .venvs/pipeline/bin/python scripts/55_run_tray.py --sim --study 0,10,20,30    # blind vs vision table + plot
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import csv
import json
import random
import sys

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import project_path
from seedling.trial_log import Session

TRAY_COLS = ["trial_id", "cell", "mode", "target_source", "est_status", "est_tilt_deg", "est_offset_x_mm", "est_offset_y_mm",
             "gt_tilt_deg", "gt_offset_mm", "acquisition_success", "extraction_success", "placement_success", "end_to_end_success",
             "failure_stage", "failure_reason", "rejection_reason", "cycle_time_s"]


def build_sim(cfg, tilt, seed, rows, cols):
    """Returns world, tray, camera, robot, gripper, T_BC, T_FC (T_BC for a fixed camera, T_FC for camera.mount=wrist)."""
    from seedling.gripper import SimGripper
    from seedling.robot import SimRobot
    from seedling.synthetic import SyntheticCamera, WristSyntheticCamera, sim_wrist_T_FC, tray_sim_camera, tray_world

    world, tray = tray_world(rows, cols, tilt_deg=tilt, seed=seed)
    cfg["grasp"]["table_z_m"] = float(world.tray_top_z) - 0.005  # substrate level is the obstacle in the sim
    world.open_width = float(cfg["gripper"]["open_width_m"])
    world.empty_width = float(cfg["gripper"]["empty_close_width_m"])
    T_BC, intr = tray_sim_camera()
    robot = SimRobot(cfg, world=world)
    if cfg["camera"].get("mount") == "wrist":
        T_FC = sim_wrist_T_FC()
        return world, tray, WristSyntheticCamera(world, robot, T_FC, intr, cfg), robot, SimGripper(cfg, world=world), None, T_FC
    return world, tray, SyntheticCamera(world, T_BC, intr), robot, SimGripper(cfg, world=world), T_BC, None


def run_session(cfg, args, tray, cam, robot, gripper, T_BC, session, gt=None, quiet=False, T_FC=None) -> list[dict]:
    """gt: simulation only, per-cell (stem base, stem direction) in tray.cells() order."""
    from seedling.task import TransplantTask, TrialAbort
    from seedling.tray import TrayTargeter

    cells = tray.cells() if args.cells == "all" else [c.strip().upper() for c in args.cells.split(",")]
    cells = tray.order_for_approach(cells, cfg) if not args.keep_order else cells
    if args.mode == "alternate":
        modes = (["blind", "vision"] * len(cells))[: len(cells)]
        random.Random(args.seed).shuffle(modes)
    else:
        modes = [args.mode] * len(cells)
    task = TransplantTask(cfg, cam, None, robot, gripper, T_BC, session, ask_operator=args.ask, T_FC=T_FC,
                          ids={"setting_id": args.setting_id, "calibration_id": "synthetic" if args.sim else args.calibration_id,
                               "model_id": "none (tray)"})
    out = []
    for i, (cell, mode) in enumerate(zip(cells, modes), 1):
        if not quiet:
            print(f"\n=== {i}/{len(cells)} cell {cell} mode={mode} ===")
        if args.pause and not args.sim and input("  Enter 开始（s 跳过，q 结束）: ").strip().lower() in {"q", "s"}:
            continue
        tg = TrayTargeter(tray, cell, mode)
        task.targeter = tg
        task.ids["condition_id"] = f"tray_{mode}"
        try:
            row = task.run_trial(f"{i:03d}_{cell}", plant_id=cell, notes=f"tray={tray.name}")
        except (KeyboardInterrupt, TrialAbort) as exc:
            print("中止:", exc)
            break
        est = tg.last_estimate
        rec = {"trial_id": row["trial_id"], "cell": cell, "mode": mode, "target_source": tg.used,
               "est_status": est.status if est else "", "est_tilt_deg": None if not est or est.tilt_deg is None else round(est.tilt_deg, 1),
               "est_offset_x_mm": None if not est or not est.offset_mm else round(est.offset_mm[0], 1),
               "est_offset_y_mm": None if not est or not est.offset_mm else round(est.offset_mm[1], 1),
               **{k: row.get(k) for k in TRAY_COLS if k in row}}
        if gt is not None:
            base, d = gt[tray.cells().index(cell)]
            gp = tray.grasp_point(cell)
            at_h = base + d * (gp[2] - base[2]) / d[2]
            rec["gt_tilt_deg"] = round(float(np.degrees(np.arccos(d[2]))), 1)
            rec["gt_offset_mm"] = round(float(np.linalg.norm(at_h[:2] - gp[:2]) * 1000), 1)
        out.append(rec)
        path = session.dir / "tray_trials.csv"
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TRAY_COLS)
            if new:
                w.writeheader()
            w.writerow({k: ("" if rec.get(k) is None else rec.get(k)) for k in TRAY_COLS})
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--tray", default=None, help="tray yaml (default tray.file)")
    ap.add_argument("--cells", default="all", help="'all' or e.g. A1,A2,B1")
    ap.add_argument("--keep-order", action="store_true", help="do not reorder cells (default: approach side first)")
    ap.add_argument("--mode", choices=["blind", "vision", "alternate"], default="blind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--setting-id", default="")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before each cell (real runs)")
    ap.add_argument("--out", default="experiments/tray_runs")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-tilt", type=float, default=0.0, help="tilt of every synthetic seedling (deg)")
    ap.add_argument("--sim-rows", type=int, default=3)
    ap.add_argument("--sim-cols", type=int, default=4)
    ap.add_argument("--study", default=None, help="sim only: comma-separated tilts; runs blind and vision for each")
    ap.add_argument("--study-seeds", type=int, default=1)
    args = ap.parse_args()
    args.calibration_id = ""
    cfg = config_from_args(args)
    cfg["task"]["reobserve_before_acquire"] = False  # handled per targeter; tray cells are not re-observed

    if args.study:
        if not args.sim:
            raise SystemExit("--study requires --sim")
        return study(cfg, args)

    if args.sim:
        world, tray, cam, robot, gripper, T_BC, T_FC = build_sim(cfg, args.sim_tilt, args.seed, args.sim_rows, args.sim_cols)
        gt = [(s.base_B.copy(), s.dir_B.copy()) for s in world.stems]
    else:
        from seedling.calibration import load_T_BC, load_T_FC
        from seedling.camera import ZedCamera
        from seedling.gripper import make_gripper
        from seedling.robot import FrankyRobot, _console_confirm
        from seedling.tray import Tray

        gt = None
        tray = Tray.load(args.tray or cfg["tray"]["file"])
        T_BC = T_FC = cam = None
        wrist = cfg["camera"].get("mount") == "wrist"
        try:
            if wrist:
                T_FC, meta = load_T_FC(cfg)
            else:
                T_BC, meta = load_T_BC(cfg)
            args.calibration_id = str(meta.get("calibration_id", ""))
            if args.mode != "blind":
                cam = ZedCamera(cfg)
        except FileNotFoundError:
            if args.mode != "blind":
                raise SystemExit("vision mode needs the camera calibration (scripts/33_wrist_calib_solve.py or 31_handeye_solve.py --save)")
            print("no camera calibration: blind mode without visual extraction check")
        if cam is None or wrist:
            cfg["task"]["verify_extraction_visual"] = False
        robot = FrankyRobot(cfg, step_mode=args.step)
        gripper = make_gripper(cfg, sim=False, step_mode=args.step, confirm_fn=_console_confirm)

    session = Session(project_path(args.out), f"{'sim' if args.sim else 'real'}_{args.mode}", cfg, {"argv": sys.argv, "tray": tray.name})
    print("session:", session.dir)
    try:
        rows = run_session(cfg, args, tray, cam, robot, gripper, T_BC, session, gt, T_FC=T_FC)
    finally:
        for dev in (gripper, robot, cam):
            try:
                dev.close()
            except Exception:
                pass
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        print(f"{mode}: end-to-end {sum(int(r.get('end_to_end_success') or 0) for r in rs)}/{len(rs)}")
    print("results:", session.dir / "tray_trials.csv")
    return 0


def study(cfg, args) -> int:
    import contextlib
    import copy
    import io

    tilts = [float(t) for t in args.study.split(",")]
    cfg["tray"]["vision"]["save_frames"] = False  # overlays/CSVs only; raw synthetic frames are large
    root = Session(project_path(args.out), "sim_study", cfg, {"argv": sys.argv})
    table = []
    for tilt in tilts:
        for mode in ("blind", "vision"):
            ok = n = 0
            for seed in range(args.study_seeds):
                c = copy.deepcopy(cfg)
                world, tray, cam, robot, gripper, T_BC, T_FC = build_sim(c, tilt, seed, args.sim_rows, args.sim_cols)
                a = copy.copy(args)
                a.mode = mode
                sess = Session(root.dir, f"tilt{tilt:g}_{mode}_s{seed}", c)
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = run_session(c, a, tray, cam, robot, gripper, T_BC, sess, [(s.base_B.copy(), s.dir_B.copy()) for s in world.stems],
                                       quiet=True, T_FC=T_FC)
                n += len(rows)
                ok += sum(int(r.get("end_to_end_success") or 0) for r in rows)
            table.append({"tilt_deg": tilt, "mode": mode, "success": ok, "n": n})
            print(f"tilt {tilt:4.0f} deg  {mode:<6}  end-to-end {ok}/{n}")
    (root.dir / "study.json").write_text(json.dumps(table, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.4))
        for mode, mk in (("blind", "o-"), ("vision", "s-")):
            pts = [(r["tilt_deg"], r["success"] / max(r["n"], 1)) for r in table if r["mode"] == mode]
            ax.plot(*zip(*pts), mk, label=mode)
        ax.set_xlabel("seedling tilt (deg)")
        ax.set_ylabel("end-to-end success (sim)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title("Synthetic tray: blind vs per-cell vision\n(geometric gripper model, not real tolerance)", fontsize=9)
        fig.tight_layout()
        fig.savefig(root.dir / "study.png", dpi=150)
    except Exception as exc:
        print("plot skipped:", exc)
    print("outputs:", root.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
