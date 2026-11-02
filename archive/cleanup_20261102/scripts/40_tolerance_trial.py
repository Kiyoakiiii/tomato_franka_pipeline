#!/usr/bin/env python
"""Gripper capture-tolerance experiment at a taught stem position (proposal section 1).

1. Teach the reference: bring the open gripper so the stem is centred between the pads
   (jog with c/a/s commands, or load --ref from an earlier session).
2. For each offset in the (shuffled) schedule along one gripper axis:
   pre-grasp -> approach to reference+offset -> grasp -> stage test -> operator labels -> release -> retreat.

Stages (kept separate, see docs/experiment_protocol_CN.md):
  hold_transport     plant already removed from the tray: lift, hold, carry, put back
  extract_from_tray  plant in its cell: lift with the extraction force limit

    .venvs/pipeline/bin/python scripts/40_tolerance_trial.py --stage hold_transport --axis c --offsets=-6,-4,-2,0,2,4,6 --repeats 2
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import random
import time

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def yes(q: str) -> int | None:
    a = input(f"  {q} [y/n/空=未知]: ").strip().lower()
    return 1 if a.startswith("y") else 0 if a.startswith("n") else None


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--stage", choices=["hold_transport", "extract_from_tray"], required=True)
    ap.add_argument("--axis", choices=["c", "a", "s"], required=True)
    ap.add_argument("--offsets", default="-6,-4,-2,0,2,4,6", help="mm, comma separated")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--ref", default=None, help="reference.json from an earlier session")
    ap.add_argument("--setting-id", default="")
    ap.add_argument("--step", action="store_true", help="confirm every motion (recommended for the first runs)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.gripper import make_gripper
    from seedling.jog import jog_session
    from seedling.robot import FrankyRobot, _console_confirm
    from seedling.trial_log import Session

    robot = FrankyRobot(cfg, step_mode=args.step)
    grip = make_gripper(cfg, sim=False, step_mode=args.step, confirm_fn=_console_confirm)
    session = Session(PROJECT_DIR / "experiments" / "tolerance", f"{args.stage}_{args.axis}", cfg, vars(args))
    fine = float(cfg["robot"]["fine_dynamics"])
    normal = float(cfg["robot"]["dynamics"])
    guard = float(cfg["task"]["guard_force_limit_n"])
    back = float(cfg["grasp"]["pregrasp_distance_m"])

    grip.open()
    if args.ref:
        T_ref = np.array(json.loads(open(args.ref).read())["T_BG_ref"])
        print("loaded reference", np.round(T_ref[:3, 3], 4).tolist())
    else:
        print("示教参考位姿：张开的夹爪应使茎位于两垫中心、目标高度，接近方向即实验方向。")
        T_ref = jog_session(robot, fine, prompt="teach reference")
        if T_ref is None:
            return 1
    (session.dir / "reference.json").write_text(json.dumps({"T_BG_ref": T_ref.tolist(), "axes": "G x=s (stem), y=c (closing), z=a (approach)"}, indent=1))
    a_dir = T_ref[:3, 2]
    T_out = T_ref.copy()
    T_out[:3, 3] -= a_dir * back
    robot.move_contact_to(T_out, dynamics=fine, description="retreat from reference")

    offsets = [float(v) for v in args.offsets.split(",")]
    schedule = [o for o in offsets for _ in range(args.repeats)]
    if not args.no_shuffle:
        random.shuffle(schedule)
    print("schedule (mm):", schedule)
    ax = {"s": 0, "c": 1, "a": 2}[args.axis]
    last_plant = ""
    for k, off in enumerate(schedule, 1):
        print(f"\n=== trial {k}/{len(schedule)}  offset {args.axis}={off:+.1f} mm ===")
        plant = input("  plant_id（回车沿用上一个，q 结束）: ").strip()
        if plant == "q":
            break
        plant = last_plant = plant or last_plant
        T_tgt = T_ref.copy()
        T_tgt[:3, 3] += T_ref[:3, ax] * off / 1000.0
        T_pre = T_tgt.copy()
        T_pre[:3, 3] -= a_dir * back
        row = {"trial_id": f"t{k:03d}", "session_id": session.dir.name, "plant_id": plant, "stage": args.stage, "setting_id": args.setting_id,
               "frame_id": "reference.json (G axes)", "attempt_index": 1, "offset_c_mm": 0.0, "offset_a_mm": 0.0, "offset_s_mm": 0.0,
               "offset_source": "commanded"}
        row[f"offset_{args.axis}_mm"] = off
        r = robot.move_contact_to(T_pre, dynamics=normal, force_limit_n=guard, description="pre-grasp")
        r = robot.move_contact_to(T_tgt, dynamics=fine, force_limit_n=guard, description="approach") if r.ok else r
        if not r.ok:
            row.update(failure_reason=f"approach: {r.reason}")
        else:
            g = grip.grasp(float(cfg["gripper"]["grasp_width_m"]) or float(cfg["gripper"]["empty_close_width_m"]) + 0.003)
            print(f"  grasp: width={None if g.width_m is None else round(g.width_m*1000, 2)} mm holding={g.holding}")
            peak = 0.0
            if args.stage == "extract_from_tray":
                T_up = robot.T_BG()
                T_up[2, 3] += float(cfg["task"]["extraction_height_m"])
                r2 = robot.move_contact_to(T_up, dynamics=fine, force_limit_n=float(cfg["task"]["extraction_force_limit_n"]), description="extract")
                peak = r2.peak_force_n or 0.0
                if r2.reaction_triggered:
                    row["failure_reason"] = "extraction force limit"
            else:
                T_up = robot.T_BG()
                T_up[2, 3] += 0.05
                robot.move_contact_to(T_up, dynamics=fine, description="lift 5 cm")
                time.sleep(2.0)
                T_side = T_up.copy()
                T_side[1, 3] += 0.08
                robot.move_contact_to(T_side, dynamics=normal, description="carry 8 cm")
                robot.move_contact_to(T_up, dynamics=normal, description="carry back")
                robot.move_contact_to(T_tgt, dynamics=fine, description="lower back")
            row["peak_external_wrench_n"] = round(peak, 2)
            row["acquisition_success"] = yes("获取成功（茎被夹住）?")
            if args.stage == "hold_transport":
                row["transport_success"] = yes("搬运过程未掉落?")
            else:
                row["extraction_success"] = yes("拔出成功（根团离开穴孔）?")
            row["slippage"] = yes("有滑移?")
            grip.open()
            row["visible_stem_damage"] = yes("茎部可见新增损伤?")
            row["visible_root_damage"] = yes("根部可见新增损伤?") if args.stage == "extract_from_tray" else None
            T_now = robot.T_BG()
            T_now[:3, 3] -= T_now[:3, 2] * back
            robot.move_contact_to(T_now, dynamics=fine, description="retreat")
        row["notes"] = input("  备注（可空）: ").strip()
        session.append_csv("tolerance_trials.csv", row)
    grip.open()
    print(f"\nresults: {session.dir / 'tolerance_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
