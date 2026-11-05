#!/usr/bin/env python
"""Put the held plant into a hole of the transplant disc (calibration/disc.json from 60_tray_locate.py), after
65_side_grasp.py has grasped and lifted it. The gripper keeps its side-grasp attitude (pitched down from the
robot side); only its heading turns to point at the hole.

    plan     (any time after the grasp plan): the poses below, checked; refuses if anything is unsafe
    carry    (at the lift pose): straight up to the transit height, across to above the hole, down to just above it
    insert   (above the hole): slowly down until the plug bottom is --insert-mm into the hole
    retreat  (after opening the gripper): back along the approach, clear of the stem, then up

Held geometry: the stem is held at the grasp plan's closing point; the plug hangs below it: its bottom is
(closing point - soil surface) + --plug-mm lower. The fingers' bottom (14.5 mm below the closing point) stays
above the disc top.

    .venvs/pipeline/bin/python scripts/71_place_in_disc.py plan --cell C5 [--hole 3]
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("phase", choices=["plan", "carry", "insert", "retreat"])
    ap.add_argument("--cell", required=True, help="the cell the plant came from (its grasp plan)")
    ap.add_argument("--hole", type=int, default=None, help="disc hole 1..6 as drawn on the overview overlay (default: nearest the robot)")
    ap.add_argument("--free", default=None, help="comma list of the free holes (default: as seen on the last overview)")
    ap.add_argument("--ignore-cells", default="", help="comma list: cells known to be empty now (not obstacles)")
    ap.add_argument("--plug-mm", type=float, default=40.0, help="plug height")
    ap.add_argument("--insert-mm", type=float, default=0.0, help="plug bottom this far into the hole before letting go (0: level "
                    "with the disc top, the hole guides it down; 18 until 2026-10-28: the wrist came down to the tray's edge)")
    ap.add_argument("--above-mm", type=float, default=30.0, help="plug bottom this far above the disc top before going in")
    ap.add_argument("--transit-mm", type=float, default=150.0, help="plug bottom above the tray / disc top while carrying (well over "
                    "the other plants' leaves; 100 until 2026-10-28: leaves were brushed)")
    ap.add_argument("--back-mm", type=float, default=60.0, help="retreat along the approach after letting go")
    ap.add_argument("--up-mm", type=float, default=100.0, help="then up")
    ap.add_argument("--finger-bottom-mm", type=float, default=14.5, help="finger bottom below the closing point")
    ap.add_argument("--wrist-clearance-mm", type=float, default=None, help="default grasp.side_wrist_clearance_mm (as for the grasp)")
    ap.add_argument("--headings-deg", type=float, nargs="+", default=[0.0] + [s_ * d_ for d_ in range(10, 91, 10) for s_ in (-1, 1)],
                    help="hand headings over the hole, relative to straight out from the robot, tried in turn")
    ap.add_argument("--wrist-clearance-off-tray-mm", type=float, default=None,
                    help="the same away from the tray (over the disc and the table): default grasp.side_wrist_clearance_off_tray_mm")
    ap.add_argument("--wrist-reach-mm", type=float, default=80.0, help="the wrist housings reach this far sideways: the surface "
                    "within it counts as under them (the joint 5/6 housing hit the tray's edge from 80 mm beside it)")
    ap.add_argument("--max-branches", type=int, default=8, help="of all joint solutions of the across pose (same family, joint 7 swept), "
                    "how many are tried per hole and heading, least joint change first")
    ap.add_argument("--max-raise-mm", type=float, default=30.0, help="let go up to this much higher when the wrist would come too close to the disc")
    ap.add_argument("--tray-plants-mm", type=float, default=50.0, help="while carrying, the plug bottom this far above the tray top where over the tray")
    ap.add_argument("--placed-plant-mm", type=float, default=150.0, help="plants already in the disc: this tall above the disc top")
    ap.add_argument("--placed-plant-radius-mm", type=float, default=40.0, help="... and the hand keeps this far from their stems")
    ap.add_argument("--camera-clearance-mm", type=float, default=20.0)
    ap.add_argument("--self-clearance-mm", type=float, default=90.0, help="hand from the arm's own upper arm and forearm (see 65)")
    ap.add_argument("--path-time-s", type=float, default=12.0, help="random-tree path search time for the carry across (all holes "
                    "and headings together; each search at most a quarter of it), when no straight or one-via path is safe")
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics / fine_dynamics in the config)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    fine = "fine" if args.dynamics is None else min(args.dynamics, 0.03)
    if args.wrist_clearance_mm is None:
        args.wrist_clearance_mm = float(cfg["grasp"].get("side_wrist_clearance_mm", 70.0))
    if args.wrist_clearance_off_tray_mm is None:
        args.wrist_clearance_off_tray_mm = float(cfg["grasp"].get("side_wrist_clearance_off_tray_mm", 70.0))
    # the wrist housings are ~6 cm in radius around the checked joint centres: over the tray the user's own setting
    # (tested by hand), over the disc and the table the full clearance
    extra_off = max(0.0, args.wrist_clearance_off_tray_mm - args.wrist_clearance_mm) / 1000

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T
    from seedling.motion import find_path, lift_then_path
    from seedling.scene import Scene
    from seedling.wrist import Surfaces, body_points_B, path_self_clearance, self_clearance

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    out = PROJECT_DIR / "experiments" / "grasp"
    grasp_file = out / f"{args.cell}_side_plan.json"
    place_file = out / f"{args.cell}_place_plan.json"
    robot = FrankyRobot(cfg, step_mode=False)
    q_now = robot.q()

    def G_of(q):
        return kinematics.fk(q, F) @ G

    def ik(T_BG, seed):
        q = kinematics.solve_ik_near(T_BG @ inv_T(G), seed, F, margin)
        if q is None or np.linalg.norm(G_of(q)[:3, 3] - T_BG[:3, 3]) > 0.0015:
            return None
        return q

    def move(q_target, dyn, name):
        r = robot.move_joints(q_target, dynamics=dyn, description=name)
        g = G_of(robot.q())[:3, 3]
        print(f"{name}: {'ok' if r.ok else 'FAILED ' + r.reason}; grasp centre (open gap centre) now {np.round(g, 4).tolist()}")
        return r.ok

    if args.phase == "plan":
        if not grasp_file.exists():
            print("no grasp plan for", args.cell)
            return 1
        gp = json.loads(grasp_file.read_text())
        disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
        grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
        tray_top, disc_top = float(grid["z_top"]), float(disc["z_top"])
        obs = out / f"{args.cell}_observe.json"
        soil = json.loads(obs.read_text()).get("soil_used") if obs.exists() else None
        cp = np.array(gp["close_point"])
        if soil is None:
            soil = cp[2] - 0.035
            print("soil height of the grasp unknown: assumed 35 mm below the closing point")
        below = (cp[2] - soil) + args.plug_mm / 1000.0  # plug bottom below the closing point (the held stem)
        holes = np.array(disc["holes"])
        fwd = float(cfg["gripper"]["pin_brake"].get("close_forward_m", 0.0))
        pitch = np.arcsin(-np.array(gp["R"])[2, 2])
        top = max(tray_top, disc_top)
        surf = Surfaces(grid, disc, cfg)
        floor_at = surf.under

        z_t = max(top + args.transit_mm / 1000 + below, cp[2] + 0.0)
        z_above = disc_top + args.above_mm / 1000 + below
        z_ins = disc_top - args.insert_mm / 1000 + below
        fingers = z_ins - args.finger_bottom_mm / 1000
        if fingers < disc_top + 0.005:
            print(f"refused: the finger bottom would come within {(fingers - disc_top) * 1000:.0f} mm of the disc top")
            return 1
        q_lift = np.array(gp["q"]["lift"])
        R0 = np.array(gp["R"])
        lift_stem = G_of(q_lift)[:3, 3] + R0[:, 2] * fwd

        def attitude(xy, dyaw=0.0):
            yaw = np.arctan2(xy[1], xy[0]) + np.radians(dyaw)
            a = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
            across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
            return np.column_stack([np.cross(across, a), across, a]), a

        def pose(xy, z_close, dyaw=0.0):
            Rr, a = attitude(xy, dyaw)
            return make_T(Rr, np.array([xy[0], xy[1], z_close]) - a * fwd)  # G = open gap centre; the closed fingers hold the stem fwd ahead

        free = [int(h) for h in args.free.split(",")] if args.free else disc.get("free_holes", list(range(1, len(holes) + 1)))
        if args.hole and args.hole not in free:
            print(f"refused: H{args.hole} is taken (a plug in it on the last overview); free: {free}")
            return 1
        taken = np.array([holes[i] for i in range(len(holes)) if i + 1 not in free]).reshape(-1, 2)
        plant_top = disc_top + args.placed_plant_mm / 1000
        # above the disc the plug only has to clear the disc and the plants already in it (lower than over the tray:
        # a flat hand high above a hole close to the robot folds onto the forearm)
        z_t_disc = disc_top + max(args.transit_mm, args.placed_plant_mm) / 1000 + below
        sc_min = args.self_clearance_mm / 1000
        z_up = max(z_t, lift_stem[2])
        T_up = make_T(R0, np.array([lift_stem[0], lift_stem[1], z_up]) - R0[:, 2] * fwd)
        q_up = ik(T_up, q_lift)

        def natural(q, xy):
            """the arm's own family of joint solutions for a flat hand: base turned towards the target (within 100 deg)
            and the shoulder leaning forward. The mirrored family (base turned away, shoulder leaning back, joint 3
            mirrored) reaches the same poses, but switching between the two turned joint 2 by 157 deg in one move
            (2026-11-02, C8: Franka stopped it with a velocity discontinuity)"""
            d = (q[0] - np.arctan2(xy[1], xy[0]) + np.pi) % (2 * np.pi) - np.pi
            return abs(d) < np.radians(100) and q[1] > -0.3

        def wrist_low(q, extra=True):
            pts = body_points_B(q, T_FC, cfg)
            wr = np.vstack([pts["wrist joints"], pts["flange"]])
            return float((wr[:, 2] - floor_at(wr, args.wrist_reach_mm / 1000, extra_off)).min())

        def move_check(name, qa, qb):
            # along the straight joint path: wrist/flange and camera clear of what lies under them (the tray top over the
            # tray, the disc top over the disc, else the table); the held plug's bottom above the tops while carrying; the
            # hand clear of the arm itself; near the disc, the hand (gripper body, flange, wrist, camera) clear of the
            # plants already in other holes. Few samples for the short moves (planning time)
            long_ = name in ("up", "via", "across")
            n = 25 if long_ else 7
            cw = cc = lg = pm = np.inf
            hit = None
            for t in np.linspace(0, 1, n):
                qq = qa + t * (qb - qa)
                pts = body_points_B(qq, T_FC, cfg)
                wr = np.vstack([pts["wrist joints"], pts["flange"]])
                cw = min(cw, float((wr[:, 2] - floor_at(wr, args.wrist_reach_mm / 1000, extra_off)).min()))
                cm = pts["camera housing"]
                cc = min(cc, float((cm[:, 2] - floor_at(cm)).min()))
                Gq = G_of(qq)[:3, 3]
                lg = min(lg, float(Gq[2]))
                # the plug bottom: over the tray (+3 cm) at least --tray-plants-mm above its top (leaves of the plants
                # still in it), elsewhere 20 mm above what lies under it
                fz = float(surf.under(Gq, 0.03)[0])
                need = fz + (args.tray_plants_mm / 1000 if fz >= tray_top - 1e-6 else 0.02)
                pm = min(pm, float(Gq[2]) - below - need) if (name != "up" or t == 1.0) else pm
                if len(taken) and hit is None and name != "up":
                    fl = pts["flange"][0]
                    hand = np.vstack([wr, cm, [fl + s * (Gq - fl) for s in (0.0, 0.33, 0.67, 1.0)]])
                    low = hand[hand[:, 2] < plant_top]
                    if len(low):
                        dd = np.hypot(low[:, None, 0] - taken[None, :, 0], low[:, None, 1] - taken[None, :, 1])
                        if (dd < args.placed_plant_radius_mm / 1000).any():
                            hit = "a plant already in the disc"
            bad = []
            if cw < args.wrist_clearance_mm / 1000:
                bad.append("wrist/flange")
            if cc < args.camera_clearance_mm / 1000:
                bad.append("camera")
            if long_ and pm < 0:  # the plug starts low at the lift pose: on the way up only where it ends counts
                bad.append("plug bottom")
            if hit:
                bad.append(hit)
            sc = path_self_clearance(qa, qb, T_FC, cfg, n=21 if long_ else 7)
            if sc < sc_min:
                bad.append(f"hand {sc * 1000:.0f} mm from the arm itself")
            line = (f"  {name:9s}: max joint change {np.degrees(np.abs(qb - qa)).max():5.1f} deg; clearance wrist/flange {cw * 1000:.0f} mm, "
                    f"camera {cc * 1000:.0f} mm over what lies below; grasp centre down to {lg:.3f}" + (f"  <-- TOO LOW {bad}" if bad else ""))
            return not bad, line

        gone = set(filter(None, args.ignore_cells.split(",")))
        scene = Scene(grid, disc, cfg, T_FC, tray_plants=[k for k in (grid.get("planted") or grid.get("occupied", []))
                                                          if k != args.cell and k not in gone],
                      taken_holes=[h for h in range(1, len(holes) + 1) if h not in free], held_below_m=below, hold_forward_m=fwd)

        q_ov = np.array(cfg["tray"]["overview_q"], float)

        def scene_back(hole):
            return Scene(grid, disc, cfg, T_FC, tray_plants=[k for k in (grid.get("planted") or grid.get("occupied", []))
                                                             if k != args.cell and k not in gone],
                         taken_holes=[h for h in range(1, len(holes) + 1) if h not in free or h == hole])

        rrt_left = [args.path_time_s]

        def plan_hole(hi, dyaw, raise_m=0.0):
            """joints for up, across, [above], insert, back, up after (+ via); or None and why. The plant is let go
            raise_m higher than --insert-mm when the wrist would otherwise come too close to the disc. Carried across
            at the lift height first (level over the tray), else lower: only clear of the disc and its plants (a flat
            hand high above a hole close to the robot can fold onto the forearm)"""
            res = (None, None)
            for zc in sorted({round(max(z_up, z_above + raise_m), 4), round(max(z_t_disc, z_above + raise_m), 4)}, reverse=True):
                r = plan_hole_at(hi, dyaw, raise_m, zc)
                if r[0] is not None:
                    return r
                res = r if res[1] is None else res
            return res

        def plan_hole_at(hi, dyaw, raise_m, z_carry):
            hxy = holes[hi]
            Rh, ah = attitude(hxy, dyaw)
            ah_h = np.array([ah[0], ah[1], 0.0]) / np.linalg.norm(ah[:2])
            z_i, z_a = z_ins + raise_m, z_above + raise_m
            T_ins = pose(hxy, z_i, dyaw)
            T_back = make_T(Rh, T_ins[:3, 3] - ah_h * args.back_mm / 1000)
            hole_side = ([("above", pose(hxy, z_a, dyaw))] if z_carry > z_a + 1e-3 else []) + \
                [("insert", T_ins), ("back", T_back), ("up after", make_T(Rh, T_back[:3, 3] + [0, 0, args.up_mm / 1000]))]
            T_ac = pose(hxy, z_carry, dyaw)
            # "across" is the long move from the grasp heading to the hole's. A flat hand high above a disc close to the
            # robot folds the wrist back towards the forearm in most joint solutions (7 joints: many per pose): the one
            # near the lift pose first, then other solutions of the same family (least joint change first); the moves
            # at the hole are solved near it. Cheap checks first: the end poses, then the short moves, the long one last;
            # when only the across move fails, it goes through a via pose (the wrist joints 5-7 turned first, or last)
            fail, seen = None, []

            def candidates():
                # ALL joint solutions of the across pose (kinematics.ik_all: joint 7 swept on every branch), the one
                # reached straight from the lift pose first, then the least joint change from the up pose
                q0 = ik(T_ac, q_up)
                if q0 is not None:
                    yield q0
                more = [q for q in kinematics.ik_all(T_ac @ inv_T(G), F, seeds=[q_up], limit_margin_rad=margin)
                        if np.linalg.norm(G_of(q)[:3, 3] - T_ac[:3, 3]) <= 0.0015]
                yield from sorted(more, key=lambda q_: float(np.abs(q_ - q_up).max()))

            for q_ac in candidates():
                if len(seen) >= args.max_branches:
                    break
                if any(np.abs(q_ac - s_).max() < 0.05 for s_ in seen) or not natural(q_ac, hxy):
                    continue
                seen.append(q_ac)
                if self_clearance(q_ac, T_FC, cfg) < sc_min:
                    fail = fail or [f"H{hi + 1}: across: hand too close to the arm itself"]
                    continue
                qs, seed, q = {"up": q_up, "across": q_ac}, q_ac, q_ac
                for name, T in hole_side:
                    q = ik(T, seed)
                    if q is None:
                        break
                    qs[name], seed = q, q
                if q is None:
                    fail = fail or [f"H{hi + 1}: not reachable ({name})"]
                    continue
                # the wrist over the disc when letting go: raise the release (at most --max-raise-mm) rather than give up
                short = args.wrist_clearance_mm / 1000 - wrist_low(qs["insert"])
                if short > 0:
                    if raise_m + short + 0.003 <= args.max_raise_mm / 1000 + 1e-9:
                        return "raise", raise_m + short + 0.003
                    fail = [f"H{hi + 1}: insert: wrist/flange {short * 1000:.0f} mm too low even with the release raised"]
                    continue
                names = [n_ for n_, _ in hole_side]
                lines, ok = {}, True
                for a_, b_ in zip(["across"] + names, names):  # the short moves at the hole first
                    ok, lines[b_] = move_check(b_, qs[a_], qs[b_])
                    if not ok:
                        break
                if not ok:
                    fail = [lines[k_] for k_ in lines]
                    continue
                ok_up, l_up = move_check("up", q_lift, q_up)
                if not ok_up:
                    return None, [l_up]  # the same for every hole and heading
                # the carry across: a path found by seedling.motion (straight, one via, or a random-tree search), every
                # pose clear of what is around (seedling.scene, with the held plug: tray, disc, the plants in both)
                import time as _t

                t0 = _t.time()
                path = find_path(q_up, q_ac, scene.free_ok, time_s=min(args.path_time_s / 4, rrt_left[0]))
                rrt_left[0] = max(0.0, rrt_left[0] - (_t.time() - t0))
                if path is None:
                    m_, part_ = scene.worst(q_up, q_ac)
                    fail = [l_up, f"  across   : max joint change {np.degrees(np.abs(q_ac - q_up)).max():5.1f} deg; no safe path (straight: "
                                  f"smallest margin {m_ * 1000:+.0f} mm, {part_})  <-- TOO LOW ['{part_}']"]
                    continue
                ls, qa_ = [], q_up
                for i_, qb_ in enumerate(path):
                    m_, part_ = scene.worst(qa_, qb_)
                    ls.append(f"  {'via' if i_ < len(path) - 1 else 'across':9s}: max joint change {np.degrees(np.abs(qb_ - qa_)).max():5.1f} deg; "
                              f"smallest margin {m_ * 1000:+.0f} mm ({part_}) over the tray, disc, plants and the arm itself")
                    qa_ = qb_
                # and the way back to the overview after letting go (as 68 will plan it: this hole taken, the plant gone
                # from the tray): a placement that leaves the arm where it cannot get back is no placement (2026-11-04
                # simulation: the hand 0 mm too close to the arm after the lift)
                back_steps, back_msg = lift_then_path(qs["up after"], q_ov, scene_back(hi + 1), cfg, T_FC, time_s=6.0)
                if back_steps is None:
                    fail = [l_up] + ls + [f"  way back : no safe way back to the overview ({back_msg})  <-- TOO LOW ['no way back']"]
                    continue
                ls.append(f"  way back : {back_msg}, {len(back_steps)} moves to the overview")
                return ({**qs, "vias": np.array(path[:-1])} if len(path) > 1 else qs), [l_up] + ls + [lines[n_] for n_ in names]
            return None, fail or [f"H{hi + 1}: not reachable (across)"]

        if q_up is None:
            print("not reachable: straight up from the lift pose")
            place_file.write_text(json.dumps({"cell": args.cell, "safe": False}))
            return 1
        # farthest from the robot first: the flat hand trails back towards the robot over the holes nearer to it, so
        # these are filled last (their plants would be in the way of the hand). --hole is tried first, and when it
        # cannot be reached now (the grasp measured after the first plan changes the carry pose by a few mm) the other
        # free holes follow
        order = [i for i in np.argsort(-np.hypot(holes[:, 0], holes[:, 1])) if i + 1 in free]
        if args.hole:
            order = [args.hole - 1] + [i for i in order if i != args.hole - 1]
        if not order:
            print("no free hole left in the disc")
            return 1
        # the hand's heading over the hole: straight out from the robot first, then turned (the plant turns about its
        # own stem, it stays upright)
        qs, raised = None, 0.0
        for hi in order:
            why = []  # per heading: the first move that failed and the check (or where the IK ran out)
            for dyaw in args.headings_deg:
                raised = 0.0
                qs, lines = plan_hole(int(hi), dyaw)
                if isinstance(qs, str):
                    raised = lines
                    qs, lines = plan_hole(int(hi), dyaw, raised)
                    if isinstance(qs, str):
                        qs, lines = None, [f"H{hi + 1}: insert: wrist/flange too low even with the release raised"]
                if qs is not None:
                    break
                bad = [ln.split(":")[0].strip() + ": " + ln.split("<-- TOO LOW")[1].strip() for ln in lines if "TOO LOW" in ln]
                why.append(f"heading {dyaw:+.0f}: " + (bad[0] if bad else lines[-1].split(": ", 1)[-1]))
                if lines and lines[0].lstrip().startswith("up ") and "TOO LOW" in lines[0]:
                    break
            if qs is not None:
                break
            print(f"H{hi + 1}: no safe path from any heading, next hole")
            for w in why:
                print("    " + w)
        for ln in lines:
            print(ln)
        if qs is None:
            print("no free hole of the disc can be reached safely")
            place_file.write_text(json.dumps({"cell": args.cell, "safe": False}))
            return 1
        hi = int(hi)
        if args.hole and hi + 1 != args.hole:
            print(f"H{args.hole} cannot be reached now: H{hi + 1} instead")
        print(f"hole H{hi + 1} at {np.round(holes[hi], 4).tolist()}, hand heading {dyaw:+.0f} deg from straight out, disc top {disc_top:.3f}; plug bottom {below * 1000:.0f} mm below the held stem; "
              f"lifted to plug bottom {(z_up - below - tray_top) * 1000:.0f} mm above the tray top, over the disc {(G_of(qs["across"])[2, 3] - below - disc_top) * 1000:.0f} mm above its top, " + (f"let go with the plug bottom {raised * 1000:.0f} mm above the hole (raised: the wrist over the disc), " if raised > 0 else
                    f"let go with it {args.insert_mm:.0f} mm in the hole, ") +
              f"finger bottom {(fingers + raised - disc_top) * 1000:.0f} mm above the disc")
        place_file.write_text(json.dumps({"cell": args.cell, "hole": hi + 1, "hole_xy": holes[hi].tolist(), "q": {k: v.tolist() for k, v in qs.items()},
                                          "start": q_lift.tolist(), "release_raised_mm": round(raised * 1000, 1), "safe": True}, indent=1))
        print("place plan saved")
        return 0

    if not place_file.exists():
        print("no place plan: run the 'plan' phase first")
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

    if args.phase == "carry":
        if not at("start"):
            return 1
        if not move(q["up"], args.dynamics, "up"):
            return 1
        for i_, qv in enumerate(q.get("vias", [])):  # via poses of the path search
            if not move(qv, args.dynamics, f"via {i_ + 1}/{len(q['vias'])}"):
                return 1
        for name in [k for k in ("across", "above") if k in q]:
            if not move(q[name], args.dynamics, name):
                return 1
        return 0
    if args.phase == "insert":
        last = "above" if "above" in q else "across"
        if not at(last):
            return 1
        return 0 if move(q["insert"], fine, "insert into the hole") else 1
    # retreat
    if not at("insert", 2.0):
        return 1
    if not move(q["back"], fine, "back out of the stem"):
        return 1
    return 0 if move(q["up after"], args.dynamics, "up") else 1


if __name__ == "__main__":
    raise SystemExit(main())
