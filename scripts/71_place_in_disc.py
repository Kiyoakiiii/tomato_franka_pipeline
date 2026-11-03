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
    ap.add_argument("--headings-deg", type=float, nargs="+", default=[0, -15, 15, -30, 30, -45, 45, -60, 60],
                    help="hand headings over the hole, relative to straight out from the robot, tried in turn")
    ap.add_argument("--wrist-clearance-off-tray-mm", type=float, default=None,
                    help="the same away from the tray (over the disc and the table): default grasp.side_wrist_clearance_off_tray_mm")
    ap.add_argument("--wrist-reach-mm", type=float, default=80.0, help="the wrist housings reach this far sideways: the surface "
                    "within it counts as under them (the joint 5/6 housing hit the tray's edge from 80 mm beside it)")
    ap.add_argument("--ik-tries", type=int, default=40, help="IK solutions of the across pose collected (random starts)")
    ap.add_argument("--max-branches", type=int, default=8, help="of those, how many are tried (clear of the arm first)")
    ap.add_argument("--camera-clearance-mm", type=float, default=20.0)
    ap.add_argument("--self-clearance-mm", type=float, default=90.0, help="hand from the arm's own upper arm and forearm (see 65)")
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

        def plan_hole(hi, dyaw):
            hxy = holes[hi]
            steps = [("up", make_T(R0, np.array([lift_stem[0], lift_stem[1], max(z_t, lift_stem[2])]) - R0[:, 2] * fwd)),
                     ("across", pose(hxy, max(z_t, z_above), dyaw))]
            if z_t > z_above + 1e-3:
                steps.append(("above", pose(hxy, z_above, dyaw)))
            steps.append(("insert", pose(hxy, z_ins, dyaw)))
            Rh, ah = attitude(hxy, dyaw)
            ah_h = np.array([ah[0], ah[1], 0.0]) / np.linalg.norm(ah[:2])
            T_back = make_T(Rh, steps[-1][1][:3, 3] - ah_h * args.back_mm / 1000)
            steps += [("back", T_back), ("up after", make_T(Rh, T_back[:3, 3] + [0, 0, args.up_mm / 1000]))]
            # "across" is a long move from the grasp heading to the hole's. A flat hand high above a disc close to the
            # robot folds the wrist back towards the forearm in most joint solutions (7 joints: many solutions per
            # pose), as for the grasp (65): --ik-tries solutions of the across pose are collected, those with the hand
            # clear of the arm first (the one near the lift pose first among them, then the least joint change: a big
            # wrist roll strains the camera cable);
            # the short moves after it are solved near it. When only the across move itself fails (its straight joint
            # path sweeps the camera past the forearm or dips the plug), it goes through a via pose: the wrist joints
            # 5-7 turned first, or last
            q_up = ik(steps[0][1], q_lift)
            if q_up is None:
                return None, [f"H{hi + 1}: not reachable (up)"]
            T_ac = steps[1][1]
            sc_min = args.self_clearance_mm / 1000
            cands = []
            for k in range(args.ik_tries + 2):
                if k == 0:
                    q = ik(T_ac, q_up)
                else:
                    start = q_up if k == 1 else np.random.default_rng(k).uniform(kinematics.Q_MIN, kinematics.Q_MAX)
                    q = kinematics.solve_ik(T_ac @ inv_T(G), start, F, margin, n_random_seeds=0 if k > 1 else 4, rng_seed=k)
                    if q is not None and np.linalg.norm(G_of(q)[:3, 3] - T_ac[:3, 3]) > 0.0015:
                        q = None
                if q is not None and not any(np.abs(q - c_[3]).max() < 0.05 for c_ in cands):
                    cands.append((self_clearance(q, T_FC, cfg) < sc_min, k != 0, float(np.abs(q - q_up).max()), q))
            cands.sort(key=lambda c_: c_[:3])
            fail = None if cands else [f"H{hi + 1}: not reachable (across)"]
            for *_, q_ac in cands[:args.max_branches]:
                qs, seed, q = {"up": q_up, "across": q_ac}, q_ac, q_ac
                for name, T in steps[2:]:
                    q = ik(T, seed)
                    if q is None:
                        break
                    qs[name], seed = q, q
                if q is None:
                    fail = fail or [f"H{hi + 1}: not reachable ({name})"]
                    continue
                ok, lines = check(qs, steps)
                if ok:
                    return qs, lines
                fail = lines
                if any("TOO LOW" in ln for ln in lines if not ln.lstrip().startswith("across")):
                    continue  # a via pose only changes the across move
                for qv in (np.r_[q_up[:4], q_ac[4:]], np.r_[q_ac[:4], q_up[4:]]):
                    steps_v = steps[:1] + [("via", None)] + steps[1:]
                    ok, lines_v = check({**qs, "via": qv}, steps_v)
                    if ok:
                        return {**qs, "via": qv}, lines_v
            return None, fail

        def check(qs, steps):
            # along every joint path: wrist/flange and camera clear of what lies under them (the tray top over the
            # tray, the disc top over the disc, else the table); the held plug's bottom above the tops while carrying
            ok, prev, lines = True, q_lift, []
            for name, _ in steps:
                q = qs[name]
                cw = cc = lg = np.inf
                for t in np.linspace(0, 1, 61):
                    qq = prev + t * (q - prev)
                    pts = body_points_B(qq, T_FC, cfg)
                    wr = np.vstack([pts["wrist joints"], pts["flange"]])
                    cw = min(cw, float((wr[:, 2] - floor_at(wr, args.wrist_reach_mm / 1000, extra_off)).min()))
                    cm = pts["camera housing"]
                    cc = min(cc, float((cm[:, 2] - floor_at(cm)).min()))
                    lg = min(lg, float(G_of(qq)[2, 3]))
                bad = []
                if cw < args.wrist_clearance_mm / 1000:
                    bad.append("wrist/flange")
                if cc < args.camera_clearance_mm / 1000:
                    bad.append("camera")
                # the plug starts low at the lift pose: on the way up only where it ends counts
                lg_check = float(G_of(q)[2, 3]) if name == "up" else lg
                if name in ("up", "via", "across") and lg_check - below < top + 0.02:
                    bad.append("plug bottom")
                sc = path_self_clearance(prev, q, T_FC, cfg)
                if sc < args.self_clearance_mm / 1000:
                    bad.append(f"hand {sc * 1000:.0f} mm from the arm itself")
                ok &= not bad
                lines.append(f"  {name:9s}: max joint change {np.degrees(np.abs(q - prev)).max():5.1f} deg; clearance wrist/flange {cw * 1000:.0f} mm, "
                             f"camera {cc * 1000:.0f} mm over what lies below; grasp centre down to {lg:.3f}" + (f"  <-- TOO LOW {bad}" if bad else ""))
                prev = q
            return ok, lines

        free = [int(h) for h in args.free.split(",")] if args.free else disc.get("free_holes", list(range(1, len(holes) + 1)))
        if args.hole and args.hole not in free:
            print(f"refused: H{args.hole} is taken (a plug in it on the last overview); free: {free}")
            return 1
        # nearest the robot first; --hole is tried first, and when it cannot be reached now (the grasp measured after the
        # first plan can change the carry pose by a few mm, enough at a joint limit) the other free holes follow
        order = [i for i in np.argsort(np.hypot(holes[:, 0], holes[:, 1])) if i + 1 in free]
        if args.hole:
            order = [args.hole - 1] + [i for i in order if i != args.hole - 1]
        if not order:
            print("no free hole left in the disc")
            return 1
        # the hand's heading over the hole: straight out from the robot first, then turned (the plant turns about its
        # own stem, it stays upright). A flat hand straight out is at joint 6's limit, as when grasping
        qs = None
        for hi in order:
            why = []  # per heading: the first move that failed and the check (or where the IK ran out)
            for dyaw in args.headings_deg:
                qs, lines = plan_hole(int(hi), dyaw)
                if qs is not None:
                    break
                bad = [ln.split(":")[0].strip() + ": " + ln.split("<-- TOO LOW")[1].strip() for ln in lines if "TOO LOW" in ln]
                why.append(f"heading {dyaw:+.0f}: " + (bad[0] if bad else lines[-1].split(": ", 1)[-1]))
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
              f"carried at plug bottom {(z_t - below - top) * 1000:.0f} mm above the tops, let go with it {args.insert_mm:.0f} mm in the hole, "
              f"finger bottom {(fingers - disc_top) * 1000:.0f} mm above the disc")
        place_file.write_text(json.dumps({"cell": args.cell, "hole": hi + 1, "hole_xy": holes[hi].tolist(), "q": {k: v.tolist() for k, v in qs.items()},
                                          "start": q_lift.tolist(), "safe": True}, indent=1))
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
        for name in [k for k in ("up", "via", "across", "above") if k in q]:
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
