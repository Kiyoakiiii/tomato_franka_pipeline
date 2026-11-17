#!/usr/bin/env python
"""Put the held plant into a hole of the transplant disc: the motion phases of a placement the program planned
(scripts/plan.py place writes experiments/grasp/<cell>_place_plan.json), after pick.py has grasped and lifted it.
(2026-11-16: this script's own per-plant placement planner, the 'plan' phase, was removed at the user's request: one
planner only, the program's.)

    carry    (at the lift pose): straight up to the transit height, across to above the hole, down to just above it
    insert   (above the hole): down until the plug is in the hole
    retreat  (after opening the gripper): back along the approach, clear of the stem, then up

    .venvs/pipeline/bin/python scripts/place.py carry --cell C5
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("phase", choices=["carry", "insert", "retreat"])
    ap.add_argument("--cell", required=True, help="the cell the plant came from (its grasp plan)")
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics / fine_dynamics in the config)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    fine = "fine" if args.dynamics is None else min(args.dynamics, 0.03)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    
    load_T_FC(cfg)  # the joint offsets estimated with the camera calibration (kinematics.JOINT_OFFSETS), as the plans used
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    out = PROJECT_DIR / "experiments" / "grasp"
    place_file = out / f"{args.cell}_place_plan.json"
    robot = FrankyRobot(cfg, step_mode=False)
    q_now = robot.q()

    def G_of(q):
        return kinematics.fk(q, F) @ G

    def move(q_target, dyn, name):
        r = robot.move_joints(q_target, dynamics=dyn, description=name)
        g = G_of(robot.q())[:3, 3]
        print(f"{name}: {'ok' if r.ok else 'FAILED ' + r.reason}; grasp centre (open gap centre) now {np.round(g, 4).tolist()}")
        return r.ok

    if not place_file.exists():
        print("no place plan: plan it first (scripts/plan.py place --cell ... --hole ...; the program's plan, 2026-11-16: the old "
              "per-plant placement planner removed)")
        return 1
    plan = json.loads(place_file.read_text())
    if not plan["safe"]:
        print("the saved place plan is not safe")
        return 1
    q = {k: np.array(v) for k, v in plan["q"].items()}
    q["start"] = np.array(plan["start"])

    def at(name, tol_deg=1.0):
        d = float(np.degrees(np.abs(q_now - q[name])).max())
        if d > tol_deg:
            print(f"refused: not at the {name} pose ({d:.1f} deg away)")
        return d <= tol_deg

    def through(key, dyn, name):
        for i_, qv in enumerate(q.get(key, [])):
            if not move(qv, dyn, f"{name} ({i_ + 1}/{len(q[key]) + 1})"):
                return False
        return True

    if args.phase == "carry":
        if float(np.degrees(np.abs(q_now - q["up"])).max()) <= 1.0:
            print("already at the up pose")  # resumed after the carry stopped there
        else:
            if not at("start"):
                return 1
            if not (through("up_vias", args.dynamics, "up") and move(q["up"], args.dynamics, "up")):
                return 1
        if not (through("vias", args.dynamics, "carry") and move(q["across"], args.dynamics, "carry")):
            return 1
        return 0
    if args.phase == "insert":
        if not at("across"):
            return 1
        return 0 if through("in_vias", fine, "down to the hole") and move(q["insert"], fine, "insert into the hole") else 1
    # retreat
    if not at("insert", 2.0):
        return 1
    return 0 if through("out_vias", fine, "back out and up") and move(q["up after"], args.dynamics, "up") else 1


if __name__ == "__main__":
    raise SystemExit(main())
