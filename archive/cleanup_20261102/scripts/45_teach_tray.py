#!/usr/bin/env python
"""Teach a tray: contact point of three corner cells -> every cell position.

Bring the gripper (in the orientation you will grasp with) to an upright seedling in each corner cell,
centred on the stem at the intended grasp height above the substrate, using c/a/s/x/y/z jog commands:
  A = first row, first column   B = first row, last column   C = last row, first column
Rows run A -> C, columns A -> B; the row nearest the robot should be A (it is picked first).

    .venvs/pipeline/bin/python scripts/45_teach_tray.py --rows 4 --cols 6 --grasp-height-mm 30 --expected-pitch-mm 50
    .venvs/pipeline/bin/python scripts/45_teach_tray.py --rows 4 --cols 6 --grasp-height-mm 30 \
        --manual "0.42,-0.12,0.08; 0.42,0.13,0.08; 0.57,-0.12,0.08"      # no robot, typed coordinates
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--grasp-height-mm", type=float, required=True, help="grasp height above the substrate surface")
    ap.add_argument("--expected-pitch-mm", type=float, default=None, help="nominal cell pitch, used to pre-position and to sanity-check")
    ap.add_argument("--name", default="tray")
    ap.add_argument("--out", default=None, help="default: tray.file from the config")
    ap.add_argument("--manual", default=None, help='"xA,yA,zA; xB,yB,zB; xC,yC,zC" in metres (skips the robot)')
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.tray import Tray

    if args.manual:
        pts = [np.array([float(v) for v in p.split(",")]) for p in args.manual.split(";")]
        if len(pts) != 3 or any(len(p) != 3 for p in pts):
            raise SystemExit("--manual needs three x,y,z triples")
        A, B, C = pts
        source = "manual"
    else:
        from seedling.jog import jog_session
        from seedling.robot import FrankyRobot

        robot = FrankyRobot(cfg, step_mode=True)
        fine = float(cfg["robot"]["fine_dynamics"])
        taught = {}
        names = {"A": "第一行第一列 (A1)", "B": f"第一行最后一列 (A{args.cols})", "C": f"最后一行第一列 ({chr(64 + args.rows)}1)"}
        for key in ("A", "B", "C"):
            if key != "A" and args.expected_pitch_mm:
                # pre-position 30 mm above the expected cell; the tray's orientation is unknown, so the operator corrects it
                pitch = args.expected_pitch_mm / 1000.0
                guess = taught["A"].copy()
                guess[:3, 3] += (np.array([0, (args.cols - 1) * pitch, 0.03]) if key == "B" else np.array([(args.rows - 1) * pitch, 0, 0.03]))
                if input(f"预移动到 {names[key]} 的估计位置上方 3 cm？[y/N] ").strip().lower().startswith("y"):
                    T_up = robot.T_BG()
                    T_up[2, 3] = guess[2, 3]
                    robot.move_contact_to(T_up, dynamics=fine, description="raise")
                    robot.move_contact_to(guess, dynamics=fine, description=f"pre-position {key}")
            print(f"\n示教 {key} = {names[key]}：让茎位于两指中心，高度为基质上方 {args.grasp_height_mm:.0f} mm，然后输入 done")
            T = jog_session(robot, fine, max_step_mm=50.0, max_total_mm=500.0, prompt=f"teach {key}")
            if T is None:
                print("aborted")
                return 1
            taught[key] = T
            T_up = robot.T_BG()
            T_up[2, 3] += 0.04
            robot.move_contact_to(T_up, dynamics=fine, description="lift 4 cm before the next corner")
        A, B, C = (taught[k][:3, 3] for k in "ABC")
        source = "taught_with_robot"

    tray = Tray(args.name, args.rows, args.cols, A, B, C, args.grasp_height_mm / 1000.0, {"source": source})
    pr, pc = tray.pitch_mm()
    print(f"\nrow pitch {pr:.1f} mm, column pitch {pc:.1f} mm, angle between row/col {tray.squareness_deg():.1f} deg, "
          f"height spread {1000 * (max(A[2], B[2], C[2]) - min(A[2], B[2], C[2])):.1f} mm")
    warn = []
    if abs(tray.squareness_deg() - 90) > 3:
        warn.append("rows/columns not perpendicular (>3 deg): check the taught corners")
    if args.expected_pitch_mm and (abs(pr - args.expected_pitch_mm) > 3 or abs(pc - args.expected_pitch_mm) > 3):
        warn.append("pitch differs from --expected-pitch-mm by >3 mm")
    for w in warn:
        print("WARNING:", w)
    path = tray.save(args.out or cfg["tray"]["file"])
    print("saved", path)
    for cid in tray.cells():
        print(f"  {cid}: {np.round(tray.grasp_point(cid), 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
