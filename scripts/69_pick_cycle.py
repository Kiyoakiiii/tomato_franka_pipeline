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

import json
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
    ap.add_argument("--pitch-deg", type=float, nargs="+", default=None, help="grasp pitches to try (65 plan; default from the config)")
    ap.add_argument("--no-return", action="store_true", help="after placing, stay above the disc (72 goes straight on to the next "
                    "plant's observation pose; the tray and disc are located once per run)")
    ap.add_argument("--program", action="store_true", help="the plan steps from the taught program (76 grasp / place: the "
                    "template moved to this cell and hole, no search); needs --place HOLE")
    ap.add_argument("--no-observe", action="store_true", help="no close-up (the program has no observation pose for this "
                    "plant): grasped at the estimated height")
    ap.add_argument("--max-replans", type=int, default=2, help="grasp planned again (another pose) this often when no placement is found from it")
    ap.add_argument("--hold-s", type=float, default=0.5, help="pause at the top of the lift (2 s until 2026-11-02)")
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
             ("plan", ["scripts/65_side_grasp.py", "plan", *c, "--lift-mm", str(args.lift_mm),
                       *(["--pitch-deg", *map(str, args.pitch_deg)] if args.pitch_deg else [])]),
             ("pre-grasp", ["scripts/65_side_grasp.py", "pregrasp", *c]),
             ("approach", ["scripts/65_side_grasp.py", "approach", *c]),
             ("close", ["scripts/66_gripper.py", "close"]),
             ("lift", ["scripts/65_side_grasp.py", "lift", *c])]
    if args.no_observe:
        # no close-up: from the overview pose, as the program checked the turn from there (76 build)
        steps = [("to the overview", ["scripts/68_goto_overview.py"])] + steps[1:]
    if args.program:
        if not args.place:
            print("--program needs --place HOLE")
            return 1
        steps[[n for n, _ in steps].index("plan")] = ("plan", ["scripts/76_program.py", "grasp", *c, "--hole", str(args.place)])
        args.max_replans = 0
    if args.place is not None:
        h = [] if args.place == 0 else ["--hole", str(args.place)]
        # the place plan right after the grasp plan: refuse before touching the plant if the hole is out of reach
        steps.insert([n for n, _ in steps].index("plan") + 1, ("plan the placement", ["scripts/76_program.py", "place", *c, *h, "--other-holes"] if args.program
                         else ["scripts/71_place_in_disc.py", "plan", *c, *h]))
        steps += [("carry to the disc", ["scripts/71_place_in_disc.py", "carry", *c]),
                  ("insert into the hole", ["scripts/71_place_in_disc.py", "insert", *c]),
                  ("open", ["scripts/66_gripper.py", "open"]),
                  ("retreat", ["scripts/71_place_in_disc.py", "retreat", *c])]
        if not args.no_return:
            steps += [("back to the overview", ["scripts/68_goto_overview.py", "--after-place", args.cell])]
    elif args.put_back:
        steps += [("put back", ["scripts/65_side_grasp.py", "lower", *c]),
                  ("open", ["scripts/66_gripper.py", "open"]),
                  ("retreat", ["scripts/65_side_grasp.py", "retreat", *c]),
                  ("back to the overview", ["scripts/68_goto_overview.py"])]
    gdir = PROJECT_DIR / "experiments" / "grasp"
    npf = gdir / f"{args.cell}_no_place.json"  # kept from the selection (72 clears it when it plans this plant)
    replans = 0
    import os

    sim = bool(os.environ.get("SEEDLING_SIM_STATE"))  # the whole-run simulation: no gripper (its scripts not run)
    t_steps = {}
    for i, (name, cmd) in enumerate(steps, 1):
        if stop.exists():
            print(f"STOPPED before step {i} ({name}) on request")
            return 2
        print(f">>> step {i}/{len(steps)}: {name}", flush=True)
        if sim and cmd[0] == "scripts/66_gripper.py":
            print("(simulated: no gripper)")
            continue
        t_s = time.time()
        r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR)
        t_steps[name] = round(time.time() - t_s, 2)
        while r.returncode != 0 and name == "plan the placement" and replans < args.max_replans:
            # no placement from this grasp pose (the hand stays flat while carrying: the grasp's joint configuration
            # decides): another grasp pose (65 leaves out the ones in {cell}_no_place.json), then the placement again
            replans += 1
            avoid = json.loads(npf.read_text()) if npf.exists() else []
            avoid.append(json.loads((gdir / f"{args.cell}_side_plan.json").read_text())["q"]["grasp"])
            npf.write_text(json.dumps(avoid))
            print(f">>> no placement from this grasp: the grasp planned again without it ({replans}/{args.max_replans})", flush=True)
            if subprocess.run([py, "-u", *dict(steps)["plan"]], cwd=PROJECT_DIR).returncode != 0:
                break
            r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR)
        if r.returncode != 0:
            print(f"step {i} ({name}) failed or refused (exit {r.returncode}): cycle stopped", flush=True)
            # exit 3: no plan, the plant not touched (the arm at most at the observation pose): 72 goes on with another
            # (the close-up too: the arm only went to the observation pose, or did not move; "no safe way there" is this
            # plant's alone)
            return 3 if name in ("observe the plug", "plan", "plan the placement") else 1
        if name == "lift" and args.hold_s > 0:
            time.sleep(args.hold_s)
    print("CYCLE DONE", flush=True)
    if sim:
        print("STEP TIMES " + json.dumps(t_steps), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
