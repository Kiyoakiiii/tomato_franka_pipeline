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
    ap.add_argument("--headings-deg", type=float, nargs="+", default=[0.0] + [s_ * d_ for d_ in (20, 40, 60, 80, 10, 30, 50, 70, 90) for s_ in (-1, 1)],
                    help="hand headings over the hole, relative to straight out from the robot, tried in turn")
    ap.add_argument("--wrist-clearance-off-tray-mm", type=float, default=None,
                    help="the same away from the tray (over the disc and the table): default grasp.side_wrist_clearance_off_tray_mm")
    ap.add_argument("--wrist-reach-mm", type=float, default=0.0, help="the wrist housings reach this far sideways: the surface "
                    "within it counts as under them (the joint 5/6 housing hit the tray's edge from 80 mm beside it)")
    ap.add_argument("--max-branches", type=int, default=3, help="of all joint solutions of the pose above the hole, how many are "
                    "tried per hole and heading (nearest to the current joints first: one posture family through the transfer)")
    ap.add_argument("--time-budget-s", type=float, default=25.0, help="planning stops after this long (the best found, or why not)")
    ap.add_argument("--max-raise-mm", type=float, default=30.0, help="let go up to this much higher when the wrist would come too close to the disc")
    ap.add_argument("--tray-plants-mm", type=float, default=50.0, help="while carrying, the plug bottom this far above the tray top where over the tray")
    ap.add_argument("--placed-plant-mm", type=float, default=150.0, help="plants already in the disc: this tall above the disc top")
    ap.add_argument("--placed-plant-radius-mm", type=float, default=40.0, help="... and the hand keeps this far from their stems")
    ap.add_argument("--camera-clearance-mm", type=float, default=20.0)
    ap.add_argument("--self-clearance-mm", type=float, default=90.0, help="hand from the arm's own upper arm and forearm (see 65)")
    ap.add_argument("--path-time-s", type=float, default=12.0, help="random-tree path search time for the carry across (all holes "
                    "and headings together; each search at most a quarter of it), when no straight or one-via path is safe")
    ap.add_argument("--max-tilt-deg", type=float, default=5.0, help="while the plant is held, the hand at most this far from flat "
                    "(fingers level) everywhere on the way, between waypoints too (seedlings break: 2026-11-05 the carry tipped it up to 90 deg)")
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics / fine_dynamics in the config)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    fine = "fine" if args.dynamics is None else min(args.dynamics, 0.03)
    if args.wrist_clearance_mm is None:
        args.wrist_clearance_mm = float(cfg["grasp"].get("side_wrist_clearance_mm", 70.0))
    if args.wrist_clearance_off_tray_mm is None:
        args.wrist_clearance_off_tray_mm = float(cfg["grasp"].get("side_wrist_clearance_off_tray_mm", 70.0))

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T
    from seedling import cartesian
    from seedling.planner import lift_then_plan, segment_ok
    from seedling.world import World

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
        z_above = disc_top + args.above_mm / 1000 + below
        z_ins = disc_top - args.insert_mm / 1000 + below
        if z_ins - args.finger_bottom_mm / 1000 < disc_top + 0.005:
            print("refused: the finger bottom would come within 5 mm of the disc top")
            return 1
        q_lift = np.array(gp["q"]["lift"])
        max_posture = float(cfg["grasp"].get("max_posture_deg", 60.0))

        def attitude(xy, dyaw=0.0):
            yaw = np.arctan2(xy[1], xy[0]) + np.radians(dyaw)
            a = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
            across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
            return np.column_stack([np.cross(across, a), across, a]), a

        def pose(xy, z_close, dyaw=0.0):
            Rr, a = attitude(xy, dyaw)
            return make_T(Rr, np.array([xy[0], xy[1], z_close]) - a * fwd)  # G = open gap centre; the stem is fwd ahead

        free = [int(h) for h in args.free.split(",")] if args.free else disc.get("free_holes", list(range(1, len(holes) + 1)))
        # the plan made when this plant was chosen is used as it is when the grasp plan is still the same one (its lift
        # pose) and the hole is still free
        prev_choice = None
        if place_file.exists() and not args.hole:
            try:
                old_ = json.loads(place_file.read_text())
                if old_.get("safe") and old_.get("hole") in free and "dyaw" in old_:
                    prev_choice = (int(old_["hole"]) - 1, float(old_["dyaw"]), float(old_.get("release_raised_mm", 0.0)) / 1000)
                if (old_.get("safe") and float(np.degrees(np.abs(np.array(old_["start"]) - q_lift)).max()) < 0.1
                        and old_.get("hole") in free and old_.get("below_mm") == round(below * 1000, 1)):
                    print(f"the place plan made when {args.cell} was chosen still holds (H{old_['hole']}): used as it is")
                    return 0
            except Exception:  # noqa: BLE001
                pass
        if args.hole and args.hole not in free:
            print(f"refused: H{args.hole} is taken (a plug in it on the last overview); free: {free}")
            return 1
        gone = set(filter(None, args.ignore_cells.split(",")))
        left = [k for k in (grid.get("planted") or grid.get("occupied", [])) if k != args.cell and k not in gone]
        taken = [h for h in range(1, len(holes) + 1) if h not in free]
        # one model of everything around (seedling.world): carrying with the plug; after letting go without it and
        # with this hole taken
        w_carry = World(cfg, T_FC, grid, disc, tray_plants=left, taken_holes=taken, held_below_m=below, hold_forward_m=fwd)
        q_ov = np.array(cfg["tray"]["overview_q"], float)

        def piece_lines(name, q0_, wps, world_, **kw):
            """check the joint waypoints of one straight piece; (ok, lines)"""
            ls, qa_, ok_ = [], q0_, True
            for i_, qb_ in enumerate(wps):
                m_, where_ = world_.worst(qa_, qb_, **kw)
                nm = name if len(wps) == 1 else f"{name} {i_ + 1}/{len(wps)}"
                ls.append(f"  {nm:10s}: max joint change {np.degrees(np.abs(qb_ - qa_)).max():5.1f} deg; smallest margin "
                          f"{m_ * 1000:+.0f} mm ({where_})" + ("  <-- TOO LOW" if m_ < 0 else ""))
                ok_ &= m_ >= 0
                qa_ = qb_
            return ok_, ls

        # 1. straight up from the lift pose, the plug's bottom --transit-mm over the tray top (its neighbours' leaves are
        # not checked on the way up: it rises out of them)
        z_up = max(tray_top + args.transit_mm / 1000 + below, G_of(q_lift)[2, 3])
        T_up = G_of(q_lift).copy()
        T_up[2, 3] = z_up
        up_pcs, why = cartesian.chain(q_lift, [("up", T_up)], F, G, margin, max_tilt=args.max_tilt_deg)
        if up_pcs is None:
            print(f"not reachable: straight up from the lift pose ({why})")
            place_file.write_text(json.dumps({"cell": args.cell, "safe": False}))
            return 1
        ok_up, l_up = piece_lines("up", q_lift, up_pcs[0][2], w_carry, plants=False, plug=False)  # the plug rises out of its own cell
        if not ok_up:
            print("\n".join(l_up))
            print("not safe: straight up from the lift pose")
            place_file.write_text(json.dumps({"cell": args.cell, "safe": False}))
            return 1
        q_up = up_pcs[0][1][-1]

        T_upG = G_of(q_up)

        def carry_flat(T_ac, z_hi):
            """the carry as straight moves of a flat hand (it only turns about the vertical, so the plant stays upright):
            (joints above the hole, waypoints) or None. Tried: moving and turning together; turning first (in place),
            then moving; moving first, then turning above the hole"""
            p_hi = np.array([T_ac[0, 3], T_ac[1, 3], z_hi])
            down = [("carry down", T_ac)] if z_hi > T_ac[2, 3] + 1e-3 else []
            routes = [[("carry", make_T(T_ac[:3, :3], p_hi))] + down,
                      [("turn", make_T(T_ac[:3, :3], T_upG[:3, 3])), ("carry", make_T(T_ac[:3, :3], p_hi))] + down,
                      [("carry", make_T(T_upG[:3, :3], p_hi)), ("turn", make_T(T_ac[:3, :3], p_hi))] + down]
            for r_ in routes:
                # high over everything: the hand may leave the straight line by up to 3 cm between waypoints (fewer stops;
                # every executed piece is still checked against everything, and kept flat)
                pcs_, _w = cartesian.chain(q_up, r_, F, G, margin, max_dev_m=0.03, max_tilt=args.max_tilt_deg)
                if pcs_ is not None:
                    return pcs_[-1][1][-1], [w for p_ in pcs_ for w in p_[2]]
            return None

        def plan_hole(hi, dyaw, raise_m):
            """(plan dict, lines) or (None, why) for hole hi at hand heading dyaw, letting go raise_m higher"""
            hxy = holes[hi]
            Rh, ah = attitude(hxy, dyaw)
            ah_h = np.array([ah[0], ah[1], 0.0]) / np.linalg.norm(ah[:2])
            # above the disc the hand only has to clear the disc and the plants in it (the safe height there); over the
            # tray it is higher (z_up): the joint move between the two is checked against everything
            z_carry = max(z_above + raise_m, disc_top + (args.placed_plant_mm + 20.0) / 1000 + below)
            T_ac, T_ins = pose(hxy, z_carry, dyaw), pose(hxy, z_ins + raise_m, dyaw)
            T_back = make_T(Rh, T_ins[:3, 3] - ah_h * args.back_mm / 1000)
            T_ua = make_T(Rh, T_back[:3, 3] + [0, 0, args.up_mm / 1000])
            w_after = World(cfg, T_FC, grid, disc, tray_plants=left, taken_holes=taken + [hi + 1])
            # the release pose first (as the grasp is chosen at the plant): every joint solution of it that is clear of
            # everything (the plug is in the hole), nearest to the current joints first; the straight way down to it is
            # worked out backwards from it (up to the carry height), so it exists whenever the release pose does
            # (2026-11-04: chosen from above instead, the straight way down ran into a joint limit for every candidate)
            cands = []
            for q_ in kinematics.ik_all(T_ins @ inv_T(G), F, seeds=[q_up], limit_margin_rad=margin):
                if np.linalg.norm(G_of(q_)[:3, 3] - T_ins[:3, 3]) > 0.0015:
                    continue
                pc_ = cartesian.posture_cost(q_, hxy)
                if pc_ <= max_posture and w_carry.ok(q_, plug=False):
                    cands.append((float(np.abs(q_ - q_up).max()), pc_, q_))
            cands.sort(key=lambda c_: c_[:2])
            # first the carry as straight moves of a flat hand from the up pose, the way in from where it ends; then the
            # release poses (the way up from each worked out backwards) with a carry path that keeps the hand flat
            starts = []
            cf = carry_flat(T_ac, max(T_upG[2, 3], z_carry))
            if cf is not None:
                starts.append((cf[0], cf[1], None))
            starts += [(None, None, c_) for c_ in cands[:args.max_branches]]
            if not starts:
                return None, [f"H{hi + 1}: no joint solution clear at the release pose"]
            why_ = []
            for q_ac, carry_wps, cand_ in starts:
                if _t.time() > deadline:
                    why_.append(f"H{hi + 1}: planning time used up (--time-budget-s {args.time_budget_s:.0f})")
                    break
                if cand_ is not None:
                    _, pc_, q_rel = cand_
                    steps_up = ([("above", pose(hxy, z_above + raise_m, dyaw))] if z_carry > z_above + raise_m + 1e-3 else []) + [("carry height", T_ac)]
                    bk, w0 = cartesian.chain(q_rel, steps_up, F, G, margin)
                    if bk is None:
                        why_.append(f"H{hi + 1}: {w0}")
                        continue
                    q_ac = bk[-1][1][-1]
                else:
                    pc_ = cartesian.posture_cost(q_ac, hxy)
                if not w_carry.ok(q_ac):
                    why_.append(f"H{hi + 1}: the pose above the hole is not clear")
                    continue
                # down into the hole, as executed (from above; the same joints as worked out backwards, within mm)
                steps_in = ([("above", pose(hxy, z_above + raise_m, dyaw))] if z_carry > z_above + raise_m + 1e-3 else []) + [("insert", T_ins)]
                pin_, w1 = cartesian.chain(q_ac, steps_in, F, G, margin, max_tilt=args.max_tilt_deg)
                if pin_ is None:
                    why_.append(f"H{hi + 1}: {w1}")
                    continue
                q_ins = pin_[-1][1][-1]
                pout, w2 = cartesian.chain(q_ins, [("back", T_back), ("up after", T_ua)], F, G, margin)
                if pout is None:
                    why_.append(f"H{hi + 1}: {w2}")
                    continue
                allq = [q_ for pcs_ in (pin_, pout) for p_ in pcs_ for q_ in p_[1]]
                pc = max(cartesian.posture_cost(q_, hxy) for q_ in allq)
                if pc > max_posture:
                    why_.append(f"H{hi + 1}: posture {pc:.0f} deg on the way in or out (limit {max_posture:.0f})")
                    continue
                lines_, ok_, qa_ = [], True, q_ac
                for name_, dense_, wps_ in pin_:  # the plug goes into the hole on the last piece: not checked against the disc there
                    o_, l_ = piece_lines(name_, qa_, wps_, w_carry, plug=name_ != "insert")
                    ok_ &= o_
                    lines_ += l_
                    qa_ = wps_[-1]
                for name_, dense_, wps_ in pout:  # let go: no plug, the hole taken (backing out, the fingers leave its stem)
                    o_, l_ = piece_lines(name_, qa_, wps_, w_after, ignore=(f"plant in H{hi + 1}",) if name_ == "back" else ())
                    ok_ &= o_
                    lines_ += l_
                    qa_ = wps_[-1]
                if not ok_:
                    bad = [ln for ln in lines_ if "TOO LOW" in ln]
                    why_.append(f"H{hi + 1}: {bad[0].strip()}")
                    continue
                # the carry from the up pose to above the hole: a path in the world with the plug (straight, else OMPL),
                # staying near natural postures (no more than 15 deg more twisted than its ends)
                if carry_wps is not None:
                    path = carry_wps
                    o_, l_c = piece_lines("carry", q_up, path, w_carry)
                    if not o_:
                        bad = [ln for ln in l_c if "TOO LOW" in ln]
                        why_.append(f"H{hi + 1}: flat carry {bad[0].strip()}")
                        continue
                else:
                    band = max(cartesian.posture_cost(q_up, G_of(q_up)[:2, 3]), pc_) + 15.0

                    def valid(q_, band=band):
                        return (cartesian.tilt_deg(q_, F, G) <= args.max_tilt_deg
                                and cartesian.posture_cost(q_, G_of(q_)[:2, 3]) <= band and w_carry.ok(q_))

                    # only the straight joint move (checked every 1.5 deg, the hand kept flat): a path search under the
                    # flat-hand condition practically never succeeds and cost 1 s per candidate (2026-11-05: 25 s per plant)
                    path = [q_ac] if valid(q_ac) and segment_ok(q_up, q_ac, valid) else None
                    if path is None:
                        why_.append(f"H{hi + 1}: no safe carry path to above the hole with the hand kept flat")
                        continue
                    o_, l_c = piece_lines("carry", q_up, path, w_carry)
                tl = max(cartesian.max_tilt_between(a_, b_, F, G) for a_, b_ in zip([q_up] + list(path[:-1]), path))
                l_c.append(f"  carry     : the hand at most {tl:.1f} deg from flat (limit {args.max_tilt_deg:g})")
                # the way back to the overview after letting go (as 68 will plan it)
                back_steps, back_msg = lift_then_plan(qa_, q_ov, w_after, cfg, lifts_m=(0.0, 0.10), time_s=2.0)
                if back_steps is None:
                    why_.append(f"H{hi + 1}: no safe way back to the overview ({back_msg})")
                    continue
                plan_ = {"up_vias": up_pcs[0][2][:-1], "up": q_up, "vias": path[:-1], "across": q_ac,
                         "in_vias": [w for p_ in pin_ for w in p_[2]][:-1], "insert": q_ins,
                         "out_vias": [w for p_ in pout for w in p_[2]][:-1], "up after": qa_}
                return plan_, l_up + l_c + lines_ + [f"  way back  : {back_msg}, {len(back_steps)} moves to the overview",
                                                       f"  posture   : at most {pc:.0f} deg"]
            return None, why_ or [f"H{hi + 1}: nothing safe"]

        # farthest hole from the robot first (the flat hand trails back towards the robot over the nearer holes);
        # --hole first if given
        order = [i for i in np.argsort(-np.hypot(holes[:, 0], holes[:, 1])) if i + 1 in free]
        if args.hole:
            order = [args.hole - 1] + [i for i in order if i != args.hole - 1]
        if not order:
            print("no free hole left in the disc")
            return 1
        import time as _t

        deadline = _t.time() + args.time_budget_s
        res = None
        if prev_choice is not None:
            # the hole and heading chosen when the plant was chosen, from the grasp as it is now (planned again after
            # the close-up): one try, ~1 s, before the full search (2026-11-05: 8-25 s per plant in the cycle)
            plan_, lines = plan_hole(*prev_choice)
            if plan_ is not None:
                res = (prev_choice[0], prev_choice[1], prev_choice[2], plan_, lines)
                print(f"the hole and heading chosen before (H{prev_choice[0] + 1}, {prev_choice[1]:+.0f} deg) work from this grasp too")
        for hi in ([] if res else order):
            if _t.time() > deadline:
                print(f"planning time used up (--time-budget-s {args.time_budget_s:.0f}) before H{hi + 1}")
                break
            why = []
            # quick check first (a few IK solutions, no paths): is there a natural posture for letting go into this hole
            # at all? (a flat hand low and close to the robot has none: the disc too low or too close)
            quick = False
            for dyaw in args.headings_deg[::2]:
                T_q = pose(holes[hi], z_ins, dyaw)
                if any(cartesian.posture_cost(q_, holes[hi]) <= max_posture
                       for q_ in kinematics.ik_all(T_q @ inv_T(G), F, seeds=[q_up], limit_margin_rad=margin, n_random=16)):
                    quick = True
                    break
            if not quick:
                print(f"H{hi + 1}: no natural posture to let go into it from any heading (the disc is too low or too close "
                      f"to the robot for a flat hand: raise it to about the tray's height, see README 'where flat grasps work')")
                continue
            for dyaw in args.headings_deg:
                for raise_m in np.arange(0.0, args.max_raise_mm / 1000 + 1e-9, 0.01):
                    plan_, lines = plan_hole(int(hi), dyaw, float(raise_m))
                    if plan_ is not None:
                        res = (int(hi), dyaw, float(raise_m), plan_, lines)
                        break
                    why.append(f"heading {dyaw:+.0f}, release +{raise_m * 1000:.0f} mm: {lines[-1]}")
                    # letting go higher only helps when the arm came too close to something on the way in or out
                    if not any("TOO LOW" in ln and ("insert" in ln or "back" in ln or "up after" in ln or "above" in ln) for ln in lines):
                        break
                if res:
                    break
            if res:
                break
            print(f"H{hi + 1}: no safe placement from any heading, next hole")
            for w_ in why[:6]:
                print("    " + w_)
        if res is None:
            print("no free hole of the disc can be reached safely")
            place_file.write_text(json.dumps({"cell": args.cell, "safe": False}))
            return 1
        hi, dyaw, raised, plan_, lines = res
        for ln in lines:
            print(ln)
        if args.hole and hi + 1 != args.hole:
            print(f"H{args.hole} cannot be reached now: H{hi + 1} instead")
        print(f"hole H{hi + 1} at {np.round(holes[hi], 4).tolist()}, hand heading {dyaw:+.0f} deg from straight out, disc top {disc_top:.3f}; "
              f"plug bottom {below * 1000:.0f} mm below the held stem; carried with its bottom {(z_up - below - tray_top) * 1000:.0f} mm above "
              f"the tray top; " + (f"let go with the plug bottom {raised * 1000:.0f} mm above the hole" if raised > 0 else
                                   f"let go with it {args.insert_mm:.0f} mm in the hole"))
        enc = {k: ([w.tolist() for w in v] if isinstance(v, list) else v.tolist()) for k, v in plan_.items()}
        place_file.write_text(json.dumps({"cell": args.cell, "hole": hi + 1, "hole_xy": holes[hi].tolist(), "q": enc,
                                          "start": q_lift.tolist(), "release_raised_mm": round(raised * 1000, 1), "dyaw": float(dyaw),
                                          "below_mm": round(below * 1000, 1), "safe": True}, indent=1))
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
