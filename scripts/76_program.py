#!/usr/bin/env python
"""The transplant job as an industrial pick-and-place program (seedling.program): taught once per layout, each
plant is the same template moved to its cell and hole. Writes the plan files the motion phases of 65 / 71 execute
({cell}_side_plan.json, {cell}_place_plan.json), so 69 runs it unchanged.

    76_program.py build                    # after 60 located tray and disc: teach, then check the whole job
    76_program.py grasp --cell C8 [--from-observe]   # the grasp for this cell (soil from the close-up if there is one)
    76_program.py place --cell C8 --hole 3            # carrying it into this hole
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import time

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR

OUT = PROJECT_DIR / "experiments" / "grasp"
PROG = OUT / "program.json"


def save_program(d: dict):
    """written whole or not at all: the run (77) reads it while the build goes on"""
    import os

    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, PROG)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("action", choices=["build", "grasp", "place"])
    ap.add_argument("--cell")
    ap.add_argument("--hole", type=int, help="grasp / place: the hole the plant goes into (it decides the arm configuration)")
    ap.add_argument("--from-observe", action="store_true", help="grasp: from the observation pose the cycle goes to first "
                    "(worked out as from the overview), not from where the arm is")
    ap.add_argument("--ignore-cells", default="", help="comma list: cells known to be empty now")
    ap.add_argument("--approach-deg", type=float, default=15.0, help="build: the approach direction for every cell, turned "
                    "this far towards the robot's right from straight out at the tray's centre (plants taken right to left)")
    ap.add_argument("--finger-clearance-mm", type=float, default=3.0)
    ap.add_argument("--other-holes", action="store_true", help="place: if the hole given does not work, the other free holes")
    ap.add_argument("--taken", default="", help="place: comma list of holes to take as filled too (planning the next plant ahead)")
    ap.add_argument("--grid", default=None, help="build: the tray file to use (testing; default the current one)")
    ap.add_argument("--free", default=None, help="build: comma list of the free holes to assume (testing)")
    ap.add_argument("--only", default="", help="build: work out the program only for these cells (comma list; testing)")
    ap.add_argument("--reteach", action="store_true", help="build: teach again even if the layout did not change")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import load_T_FC
    from seedling.program import Program
    from seedling.world import World

    t0 = time.time()
    T_FC, _ = load_T_FC(cfg)
    grid = json.loads(((PROJECT_DIR / args.grid) if args.grid else (PROJECT_DIR / cfg["tray"]["grid_file"])).read_text())
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    if args.free is not None:
        disc["free_holes"] = [int(h) for h in args.free.split(",") if h]
    gone = set(filter(None, args.ignore_cells.split(",")))
    planted = [k for k in (grid.get("planted") or grid.get("occupied", [])) if k not in gone]
    occupied = [k for k in grid.get("occupied", []) if k not in gone]
    free = list(disc.get("free_holes", []))
    from seedling.disc import taken_holes

    taken = taken_holes(disc)
    prog = Program(cfg, T_FC, grid, disc, approach_deg=args.approach_deg)
    worlds = {}

    def world_disc(extra_taken=(), plants=None, held=None, stems=False):
        """the world with these holes taken; stems: letting go, the plants in the disc taken as their stems (10 mm, 5 mm
        clear: the open fingers pass the neighbours as they pass the neighbours in the tray; holes 57.5 mm apart) instead
        of 20 mm + 10 mm (2026-11-05: H2 and H4 were refused for fingers 2-24 mm into that)"""
        key = (tuple(sorted(set(taken) | set(extra_taken))), tuple(plants if plants is not None else planted), held, stems)
        if key not in worlds:
            worlds[key] = World(cfg, T_FC, grid, disc, tray_plants=list(key[1]), taken_holes=list(key[0]),
                                held_below_m=held, hold_forward_m=prog.fwd if held else 0.0,
                                overrides={"placed_plant_radius_mm": 10.0, "plant_clearance_mm": 5.0} if stems else None)
        return worlds[key]

    def check_pieces(world, q0, wps, **kw):
        qa = np.asarray(q0, float)
        for qb in wps:
            m, where = world.worst(qa, qb, **kw)
            if m < 0:
                return f"{where} {m * 1000:+.0f} mm"
            qa = qb
        return ""

    # ---- build -----------------------------------------------------------------------------------------------------
    if args.action == "build":
        from seedling.order import plant_order

        plants = plant_order(grid, occupied)
        if args.grid or args.free is not None:
            PROG_SAVE = False
        else:
            PROG_SAVE = True
        if args.only:
            plants = [k for k in plants if k in args.only.split(",")]
        if not plants:
            print("no plants in the tray")
            return 1
        # taught once per layout: the program of the last build is used again when the tray and the disc are where they
        # were (within 5 mm / 1 deg) and it was made with the same approach direction
        old = json.loads(PROG.read_text()) if PROG.exists() and not args.reteach else None
        if old is not None and old.get("configs") and prog.same_layout(old.get("layout", {})) and \
                abs(float(old.get("approach_deg", -999)) - args.approach_deg) < 1e-6:
            prog.load(old)
            print(f"the tray and the disc are where they were: the program taught then is used ({len(prog.configs)} configuration(s))")
            ok = True
        else:
            ok = prog.teach(plants, world_disc(), lambda filled: world_disc(filled, stems=True), holes=free)
            prog.cell_prog = {}
        if not ok:
            if PROG_SAVE:
                save_program({**prog.to_dict(), "approach_deg": args.approach_deg, "building": False})
            return 1
        no_way = {}

        def save(building=True):
            # after every plant: the run starts with the first plant while the others are worked out
            if PROG_SAVE:
                save_program({**prog.to_dict(), "approach_deg": args.approach_deg, "no_way": no_way, "building": building})

        # the whole job once, in its order (each plant's cell gone after it, each hole taken after it): which plants and
        # holes the program reaches
        # the holes in the program's order, those the taught configurations let go into first (a plant's own variant
        # may reach the others)
        holes = [h for h in prog.hole_order if h in free and any(h in c_["hole_heading"] for c_ in prog.configs)]
        holes += [h for h in prog.hole_order if h in free and h not in holes]
        print(f"the job: {len(plants)} plants, {len(free)} free holes")
        # the program for every plant, in the job's order (the plants before it gone, the holes before it filled):
        # the default template (the program's approach, the taught configuration) if it grasps the plant and lets it go
        # into a free hole; otherwise the variant that does: the approach turned (straight and from the emptied side
        # first) and / or the plant's own arm configuration. Worked out once; the run only moves it to the soil height
        # kept from the last build (the same layout): each cell's program, and the cells found impossible with the same
        # holes left (worked out again when the holes differ, e.g. an emptied disc)
        no_way = (old or {}).get("no_way", {}) if prog.cell_prog else {}
        save()
        if True:
            done, left_holes = [], list(holes)
            for k in plants:
                if not left_holes:
                    break
                if k in prog.cell_prog:
                    # its program is known: which of the holes left it goes into (the plan, ~0.5 s)
                    r = grasp_plan(prog, k, cfg, grid, disc, None, [p for p in planted if p not in done], world_disc, check_pieces,
                                   soil_info(grid, k, None, prog.tray_top), args, write=False, variant=prog.cell_prog[k],
                                   stems=[x for x in occupied if x not in done])
                    h_ok = None
                    if not isinstance(r, str):
                        for h in left_holes:
                            if not isinstance(place_plan(prog, k, h, r, [p for p in planted if p not in done + [k]], world_disc,
                                                         check_pieces, extra_taken=[x for x in holes if x not in left_holes]), str):
                                h_ok = h
                                break
                    if h_ok is not None:
                        print(f"  {k} -> H{h_ok}: {prog.cell_prog[k].get('note', '')} (as before)", flush=True)
                        done.append(k)
                        left_holes.remove(h_ok)
                        continue
                    del prog.cell_prog[k]
                if no_way.get(k) == left_holes:
                    print(f"  {k}: no way (found before, with the same holes left)")
                    continue
                stems = [x for x in occupied if x not in done]
                si_k = soil_info(grid, k, None, prog.tray_top)
                found, why_k = None, []
                grasped = [False]  # some variant grasps it (then failing means no hole left that a flat hand reaches)
                placed_any = [False]  # ... and some variant places it (then it failed on its observation pose / turn)

                def try_variant(variant, ci=0, note=""):
                    r = grasp_plan(prog, k, cfg, grid, disc, None, [p for p in planted if p not in done], world_disc, check_pieces,
                                   si_k, args, write=False, ci=ci, variant=variant, stems=stems)
                    if isinstance(r, str):
                        why_k.append(f"{note}: no grasp ({r})")
                        return None
                    # which holes a flat hand reaches from this grasp (quick: continuity, release clear), then the full
                    # placement plan for those only
                    from seedling.program import PLUG

                    grasped[0] = True
                    taken_b = [x for x in holes if x not in left_holes]
                    below_k = float(r["close_point"][2] - r["soil_z"] + PLUG)
                    hh = prog.quick_holes(np.array(r["q"]["lift"]), below_k, world_disc(taken_b, stems=True), left_holes)
                    pl = "no hole a flat hand reaches from this grasp"
                    for h in left_holes:
                        if h not in hh:
                            continue
                        pl = place_plan(prog, k, h, r, [p for p in planted if p not in done + [k]], world_disc, check_pieces,
                                        extra_taken=taken_b, write=False, prefer=hh[h])
                        if not isinstance(pl, str):
                            placed_any[0] = True
                            # a complete program: also an observation pose near its stage, and the standard turn from
                            # there (straight up, one joint move high up, straight down; no path search)
                            pl_o = observe_pose(cfg, grid, k, T_FC, prog.tray_top, q_from=np.array(r["q"]["stage"]))
                            if not pl_o.ok:
                                pl = "no observation pose near its stage (the wrist would turn over)"
                                continue
                            r2 = grasp_plan(prog, k, cfg, grid, disc, pl_o.q, [p for p in planted if p not in done], world_disc,
                                            check_pieces, si_k, args, write=False, ci=ci, variant=variant, stems=stems,
                                            turn_search=False)
                            if isinstance(r2, str):
                                pl = f"from its observation pose: {r2}"
                                continue
                            return (h, r, note)
                    why_k.append(f"{note}: {pl}")
                    return None

                for ci in range(len(prog.configs)):
                    found = try_variant(None, ci, f"default template, configuration {ci + 1}")
                    if found:
                        break
                t_k = time.time()
                for cap in (0.035, 0.050):  # grasped at most 35 mm above the soil, then (deep in the tray) 50
                    if found or time.time() - t_k > 15.0:
                        break
                    for off in variant_offsets(prog, k, stems):
                        if time.time() - t_k > 15.0:
                            why_k.append("15 s of variants tried")
                            break
                        yaw = prog.yaw + np.radians(off)
                        P = prog.cell_poses(k, si_k["soil"] + si_k["height"], yaw)
                        seeds = [c_["q_ref"] for c_ in prog.configs] + own_solutions(prog, P, world_disc())
                        for si_, q0 in enumerate(seeds):
                            found = try_variant({"yaw": yaw, "q": q0, "cap": cap}, 0, f"approach {off:+.0f} deg from the program's, "
                                                f"{'configuration ' + str(si_ + 1) if si_ < len(prog.configs) else 'its own arm configuration'}"
                                                + (f", grasped up to {cap * 1000:.0f} mm above the soil" if cap > 0.035 else ""))
                            if found:
                                break
                        if found:
                            break
                if not found:
                    no_way[k] = list(left_holes)
                    save()
                    print(f"  {k}: no way found ({'; '.join(why_k[:2] + ['...'] + why_k[-2:] if len(why_k) > 4 else why_k)})")
                    if grasped[0] and not placed_any[0]:
                        print(f"  the free holes left ({', '.join(f'H{h}' for h in left_holes)}) are not reached by a flat hand from "
                              f"{k}'s grasps: the disc counts as full for this job")
                        break
                    continue
                h, r, note = found
                prog.cell_prog[k] = {"yaw": float(r["yaw"]), "q": np.array(r["q"]["grasp"]), "note": note,
                                     "cap": 0.050 if "50 mm" in note else 0.035}
                print(f"  {k} -> H{h}: {note}", flush=True)
                done.append(k)
                left_holes.remove(h)
                save()
            save(building=False)
        print(f"program built in {time.time() - t0:.1f} s")
        return 0

    prog.load(json.loads(PROG.read_text()))
    if args.action == "grasp":
        variant = prog.cell_prog.get(args.cell)
        if variant is None:
            print(f"{args.cell}: not in the program (no way to grasp it and let it go into a free hole was found when it was built)")
            return 1
        robot_q = None
        obs_from = None
        if args.from_observe:
            # the observation pose nearest (in joints) to where the hand goes next, the stage of this cell's program: the
            # turn after looking is then small (chosen as from the overview, it needed the wrist turned over and swept
            # the fingers through the plant, 2026-11-05)
            r0 = grasp_plan(prog, args.cell, cfg, grid, disc, None, planted, world_disc, check_pieces,
                            soil_info(grid, args.cell, None, prog.tray_top), args, variant=variant,
                            stems=[k for k in occupied if k != args.cell])
            q_near = np.array(r0["q"]["stage"]) if isinstance(r0, dict) else np.array(cfg["tray"]["overview_q"], float)
            pl = observe_pose(cfg, grid, args.cell, T_FC, prog.tray_top, q_from=q_near)
            if not pl.ok:
                pl = observe_pose(cfg, grid, args.cell, T_FC, prog.tray_top)
            if not pl.ok:
                print("no safe observation pose:", pl.tried[-3:])
                return 1
            q_ov = np.array(cfg["tray"]["overview_q"], float)
            obs_from = {"observe_from": q_ov.tolist(), "observe_q": pl.q.tolist(), "observe_desc": pl.desc,
                        "observe_cell_xyz": [float(v) for v in grid["cells"][args.cell]]}
            robot_q = pl.q
        else:
            from seedling.robot import FrankyRobot

            robot_q = FrankyRobot(cfg, step_mode=False).q()
        obs_file = OUT / f"{args.cell}_observe.json"
        obs = json.loads(obs_file.read_text()) if obs_file.exists() and not args.from_observe else None
        si = soil_info(grid, args.cell, obs, prog.tray_top)
        if obs is not None:
            obs["soil_used"] = si["soil"]
            obs_file.write_text(json.dumps(obs, indent=1))
        r = grasp_plan(prog, args.cell, cfg, grid, disc, robot_q, planted, world_disc, check_pieces, si, args,
                       write=True, obs_from=obs_from, finger_floor=finger_floor_fn(cfg, args, si, obs, T_FC), variant=variant,
                       stems=[k for k in occupied if k != args.cell])
        if isinstance(r, str):
            print(f"{args.cell}: no grasp in the program ({r})")
            return 1
        print(f"grasp planned in {time.time() - t0:.1f} s")
        return 0

    # place
    gp = json.loads((OUT / f"{args.cell}_side_plan.json").read_text())
    if args.hole not in free:
        print(f"refused: H{args.hole} is taken; free: {free}")
        return 1
    left = [k for k in planted if k != args.cell]
    extra = [int(h) for h in args.taken.split(",") if h]
    pl = place_plan(prog, args.cell, args.hole, gp, left, world_disc, check_pieces, write=True, extra_taken=extra)
    if isinstance(pl, str) and args.other_holes:
        # the hole meant for it does not take it from this grasp (the close-up moved the grasp): the other free holes,
        # in the program's order (2026-11-05: C8 was left for H6 alone)
        print(f"{args.cell}: not into H{args.hole} ({pl}); the other free holes")
        for h in prog.hole_order:
            if h == args.hole or h not in free or h in extra:
                continue
            pl = place_plan(prog, args.cell, h, gp, left, world_disc, check_pieces, write=True, extra_taken=extra)
            if not isinstance(pl, str):
                print(f"{args.cell}: into H{h} instead")
                break
    if isinstance(pl, str):
        print(f"{args.cell}: not into H{args.hole} ({pl})")
        return 1
    print(f"placement planned in {time.time() - t0:.1f} s")
    return 0


def variant_offsets(prog, cell, stems, offsets=(0, -10, 10, -20, 20, -30, 30, 45, -45, 60, -60)):
    """approach offsets (deg from the program's direction) to try for a cell, best first: off straight costs its
    angle, 5 per mm the open hand comes closer than 15 mm to another plant's stem; closer than 5 mm left out"""
    out = []
    for off in offsets:
        clr = prog.hand_clearance(cell, prog.yaw + np.radians(off), stems) * 1000
        if clr >= 5.0:
            out.append((abs(off) + 5.0 * max(0.0, 15.0 - clr), off))
    return [o for _, o in sorted(out)][:6]


def own_solutions(prog, P, world, n=4):
    """the plant's own clear joint solutions of its grasp pose, the most natural first"""
    from seedling import cartesian, kinematics
    from seedling.transforms import inv_T

    sols = kinematics.ik_all(P["grasp"] @ inv_T(prog.G), prog.F, limit_margin_rad=prog.margin)
    c = sorted(((cartesian.posture_cost(q, P["close"][:2]), q) for q in sols
                if np.linalg.norm(prog.G_of(q)[:3, 3] - P["grasp"][:3, 3]) <= 0.0015 and world.ok(q, fingers=False, plants=False)),
               key=lambda t: t[0])
    return [q for _, q in c[:n]]


def soil_info(grid, cell, obs, tray_top, disagree=0.008, plug_dz=0.013, leafy_cover=0.35):
    """the soil surface of this plug and the closing height above it (as 65: close measurement, overview, fallback)"""
    from seedling.program import HEIGHT_EST, HEIGHT_LEAFY, HEIGHT_SURE

    z_ov = (grid.get("soil_z_overview") or {}).get(cell)
    z_near = obs["measure"]["soil_z"] if obs and obs["measure"].get("ok") else None
    off_c = obs and np.hypot(*obs["measure"].get("shift_mm", [0, 0])) > 12.0
    fb = tray_top + plug_dz
    if z_near is not None and z_ov is not None and off_c and z_near > z_ov + 0.003:
        soil, src, sure = z_ov, "overview (the close patch was off the cell centre)", True
    elif z_near is not None and z_ov is not None and abs(z_near - z_ov) > disagree:
        soil, src = max(z_near, z_ov), "overview and close disagree: the higher"
        sure = soil == z_near
    elif z_near is not None:
        soil, src, sure = z_near, "close measurement", True
    elif z_ov is not None:
        soil, src, sure = (z_ov, "overview", False) if z_ov >= fb else (fb, "overview below the usual plug height: tray top + 13 mm", False)
    else:
        soil, src, sure = fb, "NOT MEASURED: tray top + 13 mm", False
    h = HEIGHT_SURE if sure else HEIGHT_EST
    cov = (obs or {}).get("leaf_cover")
    if sure and cov is not None and cov > leafy_cover:
        h = min(h, HEIGHT_LEAFY)
    return {"soil": float(soil), "src": src, "sure": sure, "height": h}


def finger_floor_fn(cfg, args, si, obs, T_FC):
    """lowest closing height whose open finger bottoms clear the neighbouring soil under their tracks (65), or None"""
    if not si["sure"] or obs is None:
        return None
    npz = OUT / f"{args.cell}_observe.npz"
    if not npz.exists():
        return None
    from seedling import kinematics
    from seedling.camera import Frame
    from seedling.plugsense import substrate_mask
    from seedling.stemline import _mask
    from seedling.transforms import apply_T

    fr = Frame.load(npz)
    okp = np.isfinite(fr.xyz).all(-1) & substrate_mask(fr.rgb, cfg["stemline"]) & ~_mask(fr.rgb, cfg["stemline"])
    P = apply_T(kinematics.fk_flange(np.array(obs["q"])) @ T_FC, fr.xyz[okp])

    def floor(c, yaw):
        ah = np.array([np.cos(yaw), np.sin(yaw)])
        d = P[:, :2] - c[:2]
        s_ = d @ ah
        lat = np.abs(d @ np.array([-ah[1], ah[0]]))
        m = (lat > 0.012) & (lat < 0.034) & (s_ > -0.045) & (s_ < 0.015)
        if m.sum() < 30:
            return None
        return float(np.percentile(P[m, 2] + args.finger_clearance_mm / 1000 + 0.0145, 98))

    return floor


def observe_pose(cfg, grid, cell, T_FC, tray_top, q_from=None):
    """the observation pose of 65 (camera ~0.34 m above the cell, looking down), the one nearest q_from in joints
    (default the overview pose)"""
    import copy

    from seedling.wrist import plan_observation

    c = np.array(grid["cells"][cell], float)
    guess = (grid.get("soil_z_overview") or {}).get(cell) or tray_top + 0.013
    c2 = copy.deepcopy(cfg)
    c2["tray"]["observe"].update(distance_m=0.34, elevation_deg=[65, 55, 70, 75, 80], azimuth_offsets_deg=[0, -20, 20, -40, 40],
                                 min_clearance_m=0.15, all_elevations=True, elevation_penalty_deg=2.0)
    q0 = np.array(cfg["tray"]["overview_q"], float) if q_from is None else np.asarray(q_from, float)
    return plan_observation(np.array([c[0], c[1], guess]), q0, T_FC, guess + 0.15, c2)


def grasp_plan(prog, cell, cfg, grid, disc, q_start, planted, world_disc, check_pieces, si, args, write=False,
               obs_from=None, finger_floor=None, ci=0, variant=None, stems=None, turn_search=True):
    """the template at this cell: the plan dict (65's format), or why not (str)"""
    from seedling.program import RIM
    from seedling import cartesian
    from seedling.planner import plan_path

    yaw = float(variant["yaw"]) if variant is not None else prog.yaw
    seed = variant["q"] if variant is not None else None
    # the open hand's way in must not pass through another plant (its stem within 5 mm of what the hand sweeps)
    if stems is not None:
        clr = prog.hand_clearance(cell, yaw, stems)
        if clr < 0.005:
            return f"the hand would pass through other plants (within {clr * 1000:.0f} mm of a stem)"
    soil, height = si["soil"], si["height"]
    # the highest closing point: 35 mm above the soil (the estimate), a variant may allow more (deep in the tray the
    # wrist behind the flat hand is over the tray: 2026-11-05, row D, link7 4 mm over the tray top at 35 mm)
    z_cap = soil + max(height, float((variant or {}).get("cap", 0.035)))
    z_close = max(soil + height, prog.tray_top + RIM)
    c = np.array(grid["cells"][cell], float)
    if finger_floor is not None:
        zf = finger_floor(c, yaw)
        if zf is not None and zf > z_close:
            print(f"closing point raised {(zf - z_close) * 1000:.0f} mm: the open fingers clear the neighbouring soil")
            z_close = zf
    w_at = world_disc()  # at the plant: fingers and plants not checked (the heading keeps the hand off the others)
    for _ in range(4):
        P = prog.cell_poses(cell, z_close, yaw)
        q_g, why = prog.grasp_joints(P["grasp"], ci, q_seed=seed)
        if q_g is None:
            return why
        cl = w_at.clearances(q_g, fingers=False, plants=False)
        worst = min(cl.values(), key=lambda v: v[0])
        if worst[0] >= 0:
            break
        if not worst[1].endswith(("- tray", "- table", "- disc")) or z_close - worst[0] + 0.003 > z_cap + 1e-6:
            return f"the grasp itself: {worst[1]} {worst[0] * 1000:+.0f} mm"
        print(f"closing point raised {(-worst[0] + 0.003) * 1000:.0f} mm ({worst[1]} {worst[0] * 1000:+.0f} mm)")
        z_close -= worst[0] - 0.003
    else:
        return "the grasp itself: too close to the tray"
    tpl, why = prog.template(q_g, P)
    if tpl is None:
        return why
    for name, q0, wps in tpl["pieces"]:
        bad = check_pieces(w_at, q0, wps, fingers=False, plants=False)
        if bad:
            return f"{name}: {bad}"
    vias = []
    if q_start is not None:
        # the turn from where the arm is (the observation pose) to the stage, as a robot program does it: straight up
        # from where it is (0-15 cm), one joint move high up to straight above the stage (0-30 cm), straight down to
        # the stage; the least lifting, then the least joint motion. A path search only when none of these is clear,
        # kept near the natural posture (2026-11-05: the search's via pose swung the hand to y = +0.35 m, z = 0.19 m)
        scene = world_disc(plants=sorted(set(planted) | {cell}))
        q_st = tpl["stage"]
        T_obs = prog.G_of(q_start)
        best = None
        # (from the observation pose the hand is ~5 cm over the plant: turning the wrist there sweeps the fingers through
        # it, so lifting first is the usual way)
        for up_d in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            if up_d == 0.0:
                lift_w, q_a = [], np.asarray(q_start, float)
            else:
                T_u = T_obs.copy()
                T_u[2, 3] += up_d
                lp, _w = prog.chain(q_start, [("up", T_u)])
                if lp is None or check_pieces(scene, q_start, lp[0][2]):
                    continue
                lift_w, q_a = list(lp[0][2]), lp[0][1][-1]
            for hi in (0.0, 0.10, 0.20, 0.30):
                if hi == 0.0:
                    q_hi, down_w = q_st, []
                else:
                    T_hi = P["stage"].copy()
                    T_hi[2, 3] += hi
                    upc, _w = prog.chain(q_st, [("above the stage", T_hi)])
                    if upc is None:
                        continue
                    q_hi = upc[0][1][-1]
                    dn, _w = prog.chain(q_hi, [("down to the stage", P["stage"])])
                    if dn is None or check_pieces(scene, q_hi, dn[0][2]):
                        continue
                    down_w = list(dn[0][2])
                if scene.worst(q_a, q_hi)[0] < 0:
                    continue
                cost = (up_d + hi, float(np.abs(q_hi - q_a).max()))
                if best is None or cost < best[0]:
                    best = (cost, lift_w + [q_hi] + down_w[:-1] if hi else lift_w)
            if best is not None:
                break  # the least lifting that works
        if best is not None:
            vias = best[1]
        elif not turn_search:
            return "no turn of the standard kind from the observation pose to the stage"
        else:
            band = cartesian.posture_cost(q_st, P["close"][:2]) + 40.0

            def sensible(q_):
                return cartesian.posture_cost(q_, P["close"][:2]) <= band and scene.ok(q_)

            path = plan_path(q_start, q_st, sensible, time_s=3.0)
            if path is None:
                return "no safe turn from the observation pose to the stage"
            vias = list(path[:-1])
    yaw_rel = float(np.degrees((yaw - np.arctan2(c[1], c[0]) + np.pi) % (2 * np.pi) - np.pi))
    plan = {"cell": cell, "close_point": P["close"].tolist(), "offset_mm": [0.0, 0.0], "R": P["R"].tolist(),
            "q": {"start": (q_start if q_start is not None else tpl["stage"]).tolist(), "stage": tpl["stage"].tolist(),
                  "pregrasp": tpl["pregrasp"].tolist(), "grasp": tpl["grasp"].tolist(), "lift": tpl["lift"].tolist()},
            "vias": [np.asarray(v).tolist() for v in vias],
            **{f"{n}_vias": [np.asarray(w).tolist() for w in wps[:-1]] for n, _q0, wps in tpl["pieces"]},
            "z_close": round(float(z_close), 4), "tray_top": round(prog.tray_top, 4), "yaw": yaw, "soil_z": si["soil"],
            **(obs_from or {}), "cell_xy": [float(v) for v in c[:2]], "program": True, "config": ci, "safe": True}
    if write:
        print(f"soil surface of {cell}: z {si['soil']:.3f} ({(si['soil'] - prog.tray_top) * 1000:+.0f} mm vs the tray top) from {si['src']}")
        print(f"cell {cell}: closing point {np.round(P['close'], 4).tolist()}, {(z_close - si['soil']) * 1000:.0f} mm above the soil; "
              f"the turn {'one joint move' if not vias else f'through {len(vias)} via pose(s)'}")
        print(f"approach pitched 0 deg down, heading {yaw_rel:+.0f} deg from straight out (the program's direction); camera above the gripper")
        (OUT / f"{cell}_side_plan.json").write_text(json.dumps(plan, indent=1))
        print("plan saved", (OUT / f"{cell}_side_plan.json").relative_to(PROJECT_DIR))
    return plan


def place_plan(prog, cell, hole, gp, left, world_disc, check_pieces, extra_taken=(), write=False, prefer=None):
    """the template from this grasp into this hole: the plan dict (71's format), or why not (str)"""
    from seedling.program import HEADINGS, MAX_TILT, PLUG

    q_lift = np.array(gp["q"]["lift"])
    soil = gp.get("soil_z", gp["close_point"][2] - 0.035)
    below = float(gp["close_point"][2] - soil + PLUG)
    T_l = prog.G_of(q_lift)
    T_up = T_l.copy()
    T_up[2, 3] = prog.z_carry(T_l[2, 3], below)
    taken_now = list(extra_taken)
    w_carry = world_disc(taken_now, plants=left, held=below)
    w_rel = world_disc(taken_now, plants=left, held=below, stems=True)  # going in: the neighbours as stems
    w_after = world_disc(taken_now + [hole], plants=left, stems=True)  # letting go and backing out
    up, why = prog.chain(q_lift, [("up", T_up)], max_tilt=MAX_TILT)
    if up is None:
        return f"straight up: {why}"
    bad = check_pieces(w_carry, q_lift, up[0][2], plants=False, plug=False)
    if bad:
        return f"straight up: {bad}"
    q_up = up[0][1][-1]
    taught = prog.configs[int(gp.get("config", 0))]["hole_heading"] if prog.configs else {}
    first = [prefer] if prefer is not None else ([taught[hole]] if hole in taught else [])
    heads = first + [h for h in HEADINGS if h not in first]
    last = "no heading"
    for dy in heads:
        for raise_m in (0.0, 0.01, 0.02, 0.03):
            H = prog.hole_poses(hole, dy, below, raise_m)
            q_ac, carry = prog.carry(q_up, T_up, H["at"](T_up[2, 3]))
            if q_ac is None:
                last = f"heading {dy:+.0f}: no straight flat carry"
                break
            bad = check_pieces(w_carry, q_up, carry)
            if bad:
                last = f"heading {dy:+.0f}: carry {bad}"
                break
            pin, why = prog.chain(q_ac, [("above", H["above"]), ("insert", H["insert"])], max_tilt=MAX_TILT)
            if pin is None:
                last = f"heading {dy:+.0f}: {why}"
                break
            # going down over the hole and in: the neighbours as stems (the carry above them kept the full plants)
            bad = check_pieces(w_rel, q_ac, pin[0][2]) or check_pieces(w_rel, pin[0][1][-1], pin[1][2], plug=False)
            q_ins = pin[1][1][-1]
            pout, why = prog.chain(q_ins, [("back", H["back"]), ("up after", H["up after"])])
            if pout is None:
                last = f"heading {dy:+.0f}: {why}"
                break
            bad = bad or check_pieces(w_after, q_ins, pout[0][2], ignore=(f"plant in H{hole}",)) or \
                check_pieces(w_after, pout[0][1][-1], pout[1][2])
            if bad:
                last = f"heading {dy:+.0f}, let go {raise_m * 1000:.0f} mm higher: {bad}"
                continue  # letting go higher may clear it
            in_w = list(pin[0][2]) + list(pin[1][2])
            out_w = list(pout[0][2]) + list(pout[1][2])
            plan = {"cell": cell, "hole": hole, "hole_xy": prog.holes[hole - 1].tolist(),
                    "q": {"up_vias": [w.tolist() for w in up[0][2][:-1]], "up": q_up.tolist(),
                          "vias": [w.tolist() for w in carry[:-1]], "across": q_ac.tolist(),
                          "in_vias": [w.tolist() for w in in_w[:-1]], "insert": q_ins.tolist(),
                          "out_vias": [w.tolist() for w in out_w[:-1]], "up after": out_w[-1].tolist()},
                    "start": q_lift.tolist(), "release_raised_mm": round(raise_m * 1000, 1), "below_mm": round(below * 1000, 1),
                    "dyaw": float(dy), "program": True, "safe": True}
            if write:
                from seedling import cartesian

                tl = max(cartesian.max_tilt_between(a_, b_, prog.F, prog.G) for a_, b_ in zip([q_up] + carry[:-1], carry))
                if dy != taught.get(hole):
                    print(f"H{hole}: the taught heading did not work now, {dy:+.0f} deg instead")
                print(f"hole H{hole}: hand heading {dy:+.0f} deg from straight out; plug bottom {below * 1000:.0f} mm below the held stem; "
                      f"carried {len(carry)} straight piece waypoints, the hand at most {tl:.1f} deg from flat; "
                      + (f"let go {raise_m * 1000:.0f} mm higher" if raise_m else "let go with the plug bottom at the disc top"))
                (OUT / f"{cell}_place_plan.json").write_text(json.dumps(plan, indent=1))
                print("place plan saved")
            return plan
    return last


if __name__ == "__main__":
    raise SystemExit(main())
