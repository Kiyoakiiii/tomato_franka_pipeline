#!/usr/bin/env python
"""One plant, the whole cycle, from the overview pose and back:
observe (soil height of this plug) -> plan -> pre-grasp (turn high, go down) -> approach -> close -> lift
-> [put back: lower -> open] -> retreat -> overview, or with --place: -> carry to the disc -> insert -> open
-> retreat -> overview (71_place_in_disc.py; the disc located by 60_tray_locate.py).

Every step is a separate script (65 / 66 / 68) and stops the cycle if it fails or refuses. Between steps the
file experiments/grasp/STOP is checked: create it (touch) to stop after the current step, so that no motion is
cut off in the middle (which puts the robot into an error state). The tray must have been located first
(60_tray_locate.py, from the overview pose).

    .venvs/pipeline/bin/python scripts/69_pick_cycle.py --cell C5 --lift-mm 60 --put-back
    .venvs/pipeline/bin/python scripts/69_pick_cycle.py --cell C5 --lift-mm 60 --place 1   # transplant into disc hole 1
    touch experiments/grasp/STOP          # stop after the current step
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import subprocess
import sys
import time

from _common import base_parser

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--lift-mm", type=float, default=60.0)
    ap.add_argument("--put-back", action="store_true", help="after lifting, put the plant back and let go")
    ap.add_argument("--hold-s", type=float, default=2.0, help="pause at the top of the lift")
    ap.add_argument("--place", type=int, nargs="?", const=0, default=None, metavar="HOLE",
                    help="transplant: put the plant into this disc hole (1..6; no number: the planner's choice)")
    args = ap.parse_args()

    stop = PROJECT_DIR / "experiments" / "grasp" / "STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    if stop.exists():
        stop.unlink()
    py = sys.executable
    c = ["--cell", args.cell]
    steps = [("observe the plug", ["scripts/65_side_grasp.py", "observe", *c]),
             ("plan", ["scripts/65_side_grasp.py", "plan", *c, "--lift-mm", str(args.lift_mm)]),
             ("pre-grasp", ["scripts/65_side_grasp.py", "pregrasp", *c]),
             ("approach", ["scripts/65_side_grasp.py", "approach", *c]),
             ("close", ["scripts/66_gripper.py", "close"]),
             ("lift", ["scripts/65_side_grasp.py", "lift", *c])]
    if args.place is not None:
        h = [] if args.place == 0 else ["--hole", str(args.place)]
        # the place plan right after the grasp plan: refuse before touching the plant if the hole is out of reach
        steps.insert(2, ("plan the placement", ["scripts/71_place_in_disc.py", "plan", *c, *h]))
        steps += [("carry to the disc", ["scripts/71_place_in_disc.py", "carry", *c]),
                  ("insert into the hole", ["scripts/71_place_in_disc.py", "insert", *c]),
                  ("open", ["scripts/66_gripper.py", "open"]),
                  ("retreat", ["scripts/71_place_in_disc.py", "retreat", *c]),
                  ("back to the overview", ["scripts/68_goto_overview.py"])]
    elif args.put_back:
        steps += [("put back", ["scripts/65_side_grasp.py", "lower", *c]),
                  ("open", ["scripts/66_gripper.py", "open"]),
                  ("retreat", ["scripts/65_side_grasp.py", "retreat", *c]),
                  ("back to the overview", ["scripts/68_goto_overview.py"])]
    for i, (name, cmd) in enumerate(steps, 1):
        if stop.exists():
            print(f"STOPPED before step {i} ({name}) on request")
            return 2
        print(f">>> step {i}/{len(steps)}: {name}", flush=True)
        r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR)
        if r.returncode != 0:
            print(f"step {i} ({name}) failed or refused (exit {r.returncode}): cycle stopped", flush=True)
            return 1
        if name == "lift" and args.hold_s > 0:
            time.sleep(args.hold_s)
    print("CYCLE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
