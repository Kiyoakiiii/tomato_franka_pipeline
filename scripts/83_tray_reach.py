#!/usr/bin/env python
"""Where must a FULL tray stand so that every one of its cells can be grasped? (2026-11-10: the goal is the whole
tray; at the tray position of 2026-11-06 the rows D-F were out of reach.) Planning only (no robot, no camera).

For the tray as last located, moved by each candidate offset (towards / away from the robot, to its left / right,
up / down), the job in its order (nearest row first, right to left; the plants before it gone): per cell, whether
the program grasps it (76 grasp_plan: the approach turned (as the program's variants, up to +-60 deg), the plant's own arm configurations, the
hand flat or pitched down, the closing point up to 50 mm above the soil) and lifts it straight up to the carry height
(the plug clear of the plants left). The placement is not checked (the container is not decided). The disc is
taken out of the way.

    .venvs/pipeline/bin/python scripts/83_tray_reach.py                     # the default offsets
    .venvs/pipeline/bin/python scripts/83_tray_reach.py --radial -0.20 -0.10 0 --lateral 0 --up 0
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import importlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def evaluate(job):
    """one tray offset: {cell: (ok, grasp height above the soil mm, pitch, why)} in the job's order"""
    cfg, grid, disc, (dr, dl, dz), ov, disc_xy = job
    from types import SimpleNamespace

    from seedling.calibration import load_T_FC
    from seedling.order import plant_order
    from seedling.program import GRASP_PITCHES, HEIGHT_SURE, PLUG, TRANSIT, Program
    from seedling.transforms import make_T
    from seedling.world import World
    from seedling.wrist import path_self_clearance

    m76 = importlib.import_module("76_program")
    T_FC, _ = load_T_FC(cfg)
    c = np.mean([v[:2] for v in grid["cells"].values()], axis=0)
    u = c / np.linalg.norm(c)
    lft = np.array([-u[1], u[0]])
    d = u * dr + lft * dl
    g = json.loads(json.dumps(grid))
    g["cells"] = {k: [v[0] + d[0], v[1] + d[1], v[2] + dz] for k, v in g["cells"].items()}
    g["z_top"] = float(g["z_top"]) + dz
    allc = sorted(g["cells"])
    g["occupied"] = g["planted"] = allc
    g["soil_z_overview"] = {k: g["z_top"] + 0.013 for k in allc}
    dd = json.loads(json.dumps(disc))
    # the disc out of the way, or (disc_xy) its centre there: then each plant grasped is also let go into an empty disc
    far = np.array([0.0, 3.0]) if disc_xy is None else np.asarray(disc_xy, float) - np.asarray(dd["centre"][:2], float)
    dd["centre"] = (np.array(dd["centre"][:2]) + far).tolist()
    dd["holes"] = [(np.array(h[:2]) + far).tolist() for h in dd["holes"]]
    prog = Program(cfg, T_FC, g, dd)
    min_self = float(cfg["tray"].get("observe", {}).get("min_self_clearance_m", 0.09))
    worlds = {}

    def world_disc(extra_taken=(), plants=None, held=None, stems=False):
        key = (tuple(plants if plants is not None else allc), held, stems)
        if key not in worlds:
            worlds[key] = World(cfg, T_FC, g, dd, tray_plants=list(key[0]), taken_holes=[], held_below_m=held,
                                hold_forward_m=prog.hold_fwd if held else 0.0,
                                overrides={**ov, **({"placed_plant_radius_mm": 10.0, "plant_clearance_mm": 5.0} if stems else {})} or None)
        return worlds[key]

    def check_pieces(world, q0, wps, **kw):
        qa = np.asarray(q0, float)
        for qb in wps:
            m, where = world.worst(qa, qb, **kw)
            if m < 0:
                return f"{where} {m * 1000:+.0f} mm"
            if path_self_clearance(qa, np.asarray(qb, float), T_FC, cfg) < min_self:
                return "self-collision reflex"
            qa = qb
        return ""

    args = SimpleNamespace()
    order = plant_order(g, allc)
    out, gone = {}, []
    for k in order:
        left = [x for x in allc if x not in gone]
        si = dict(m76.soil_info(g, k, None, prog.tray_top), height=HEIGHT_SURE, sure=True)  # as after the close-up
        best, why = None, "no grasp"
        for off in m76.variant_offsets(prog, k, [x for x in left if x != k]):
            yaw = prog.yaw + np.radians(off)
            P0 = prog.cell_poses(k, si["soil"] + si["height"], yaw)
            seeds = m76.own_solutions(prog, P0, world_disc(plants=left), cell=k, yaw=yaw)
            if not seeds:
                why = f"approach {off:+d} deg: no clear joint solution of the grasp (out of reach, or the arm into something)"
            for q_seed in seeds:
                for pitch in (0.0,) + tuple(GRASP_PITCHES):
                    r = m76.grasp_plan(prog, k, cfg, g, dd, None, left, world_disc, check_pieces, dict(si, pitch_deg=pitch), args,
                                       variant={"yaw": yaw, "q": q_seed, "cap": 0.050}, stems=[x for x in left if x != k])
                    if isinstance(r, str):
                        why = r
                        continue
                    # straight up to the carry height, the plug clear of the plants left (the hand within 10 deg of its attitude)
                    below = float(r["close_point"][2] - si["soil"] + PLUG)
                    T_l = prog.G_of(np.array(r["q"]["lift"]))
                    z_up = max(prog.tray_top + TRANSIT + below, T_l[2, 3])
                    up, w_ = prog.chain(np.array(r["q"]["lift"]), [("up", make_T(T_l[:3, :3], T_l[:3, 3] + [0, 0, z_up - T_l[2, 3]]))], max_tilt=10.0)
                    if up is None:
                        why = f"no straight lift ({w_})"
                        continue
                    bad = check_pieces(world_disc(plants=[x for x in left if x != k], held=below), np.array(r["q"]["lift"]), up[0][2],
                                       plants=False, plug=False)
                    if bad:
                        why = f"lift: {bad}"
                        continue
                    h = (r["z_close"] - si["soil"]) * 1000
                    if best is None or h < best[0] - 0.5:
                        best = (h, pitch, off, r)
                    break  # the flattest that works for this seed
                if best is not None and best[0] <= 17.5:
                    break
            if best is not None and best[0] <= 17.5:
                break
        placed = None
        if best is not None and disc_xy is not None:
            placed = False
            for hole in range(1, len(dd["holes"]) + 1):
                pl = m76.place_plan(prog, k, hole, best[3], [x for x in left if x != k], world_disc, check_pieces)
                if not isinstance(pl, str):
                    placed = True
                    break
                why = f"grasped, not let go into the disc: {pl}"
        out[k] = (best is not None, None if best is None else round(best[0]), None if best is None else best[1],
                  "" if best and placed is not False else why, placed)
        if best is not None:
            gone.append(k)  # taken; a cell that cannot be taken stays (and may block the ones behind it)
    return (dr, dl, dz), out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--radial", type=float, nargs="+", default=[-0.20, -0.15, -0.10, -0.05, 0.0], help="m, + away from the robot")
    ap.add_argument("--lateral", type=float, nargs="+", default=[0.0], help="m, + to the robot's left")
    ap.add_argument("--up", type=float, nargs="+", default=[0.0], help="m")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--camera-self-clearance-mm", type=float, default=None, help="testing: the camera housing's clearance from the arm's "
                    "links 0-5 in the world model (default seedling.world, 25)")
    ap.add_argument("--disc-xy", type=float, nargs=2, action="append", default=None, metavar=("X", "Y"),
                    help="the disc's centre (m, base frame; repeatable): each plant also let go into an empty disc there")
    ap.add_argument("--grid", default=None, help="the tray record to start from (default: the last locate)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    grid = json.loads((PROJECT_DIR / (args.grid or cfg["tray"]["grid_file"])).read_text())
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    ov = {} if args.camera_self_clearance_mm is None else {"camera_self_clearance_mm": args.camera_self_clearance_mm}
    jobs = [(cfg, grid, disc, (r, l, z), ov, dxy) for r in args.radial for l in args.lateral for z in args.up
            for dxy in (args.disc_xy or [None])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(evaluate, jobs))
    cols = sorted({int(k[1:]) for k in grid["cells"]})
    rows = sorted({k[0] for k in grid["cells"]})
    a1 = np.array(grid["cells"]["A1"][:2])
    print(f"the tray as located (A1 at {np.round(a1, 3).tolist()}, top {grid['z_top']:.3f} m), moved; a full tray, every cell "
          f"grasped in the job's order and lifted to the carry height (placement not checked)")
    for job, ((dr, dl, dz), out) in zip(jobs, res):
        n = sum(v[0] for v in out.values())
        n_pl = sum(bool(v[4]) for v in out.values())
        hs = [v[1] for v in out.values() if v[0]]
        print(f"\nmoved {dr * 100:+.0f} cm away from the robot, {dl * 100:+.0f} cm to its left, {dz * 100:+.0f} cm up: "
              f"{n}/{len(out)} cells grasped" + (f", {n_pl} also let go into the disc at {job[5]}" if job[5] is not None else "")
              + (f"; grasp height above the soil mean {np.mean(hs):.0f} mm, max {max(hs)} mm" if hs else ""))
        print("      " + " ".join(f"{c:>3d}" for c in cols))
        for r_ in rows:
            line = []
            for c in cols:
                v = out.get(f"{r_}{c}")
                line.append("  ." if v is None else ("  X" if not v[0] else "  g" if v[4] is False else f"{v[1]:3d}"))
            print(f"   {r_}  " + " ".join(line))
        fails = [(k, v[3]) for k, v in out.items() if not v[0] or v[4] is False]
        for k, w in fails[:4]:
            print(f"   {k}: {w[:150]}")
    print(f"\n(numbers: the closing point's height above the soil, mm; X: not grasped; g: grasped but not let go into the disc; "
          f"{time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
