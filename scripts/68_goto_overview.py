#!/usr/bin/env python
"""Go to the overview pose (tray.overview_q): first straight up where the arm is (only if needed, and
as little as needed), then to the overview along a path (seedling.planner: straight, else OMPL), every pose of it
clear of what is around (seedling.world: Franka's own link meshes, the camera, the gripper; the table, the tray and
the disc as solid blocks, the plants in the tray and in the disc, the arm itself).

    .venvs/pipeline/bin/python scripts/68_goto_overview.py --dry-run
    .venvs/pipeline/bin/python scripts/68_goto_overview.py
    .venvs/pipeline/bin/python scripts/68_goto_overview.py --after-place C7   # the plant just put into the disc counts too
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics in the config)")
    ap.add_argument("--path-time-s", type=float, default=10.0, help="random-tree path search time when no straight or one-via path is safe")
    ap.add_argument("--after-place", default=None, metavar="CELL", help="the plant of this cell was just put into the disc "
                    "(its place plan's hole counts as taken; the last overview did not see it there yet)")
    ap.add_argument("--free", default=None, help="comma list of the free disc holes (default: as seen on the last overview)")
    ap.add_argument("--ignore-cells", default="", help="comma list: cells known to be empty now")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start-q", type=float, nargs=7, default=None, help="with --dry-run: plan from these joints, not from where the arm is")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import load_T_FC
    from seedling.planner import lift_then_plan
    from seedling.robot import FrankyRobot
    from seedling.world import World

    T_FC, _ = load_T_FC(cfg)
    q_ov = np.array(cfg["tray"]["overview_q"], float)
    robot = FrankyRobot(cfg, step_mode=False)
    if args.start_q is not None and not args.dry_run:
        print("--start-q only with --dry-run")
        return 1
    q0 = np.array(args.start_q) if args.start_q is not None else robot.q()

    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    df = PROJECT_DIR / "calibration" / "disc.json"
    disc = json.loads(df.read_text()) if df.exists() else None
    gone = set(filter(None, args.ignore_cells.split(",")))
    plants = [k for k in (grid.get("planted") or grid.get("occupied", [])) if k not in gone]
    taken = None
    if disc is not None and "holes" in disc:
        free = [int(h) for h in args.free.split(",")] if args.free else list(disc.get("free_holes", range(1, len(disc["holes"]) + 1)))
        if args.after_place:
            pf = PROJECT_DIR / "experiments" / "grasp" / f"{args.after_place}_place_plan.json"
            if pf.exists() and json.loads(pf.read_text()).get("hole") in free:
                free.remove(json.loads(pf.read_text())["hole"])
            plants = [k for k in plants if k != args.after_place]
        taken = [h for h in range(1, len(disc["holes"]) + 1) if h not in free]
    world = World(cfg, T_FC, grid, disc, tray_plants=plants, taken_holes=taken)

    steps, msg = lift_then_plan(q0, q_ov, world, cfg, time_s=args.path_time_s)
    if steps is None:
        print(f"to the overview: NOT SAFE: no safe way found ({msg})")
        return 1
    print(msg)
    steps = [(n.replace("the goal", "the overview"), qa, qb) for n, qa, qb in steps]
    for name, qa, qb in steps:
        m, part = world.worst(qa, qb)
        print(f"{name}: max joint change {np.degrees(np.abs(qb - qa)).max():.0f} deg, smallest margin {m * 1000:+.0f} mm ({part})")
    if args.dry_run:
        return 0
    for name, qa, qb in steps:
        r = robot.move_joints(qb, dynamics=args.dynamics, description=name)
        print(name, "ok" if r.ok else "FAILED " + r.reason)
        if not r.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
