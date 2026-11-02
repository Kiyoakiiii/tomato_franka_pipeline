#!/usr/bin/env python
"""Side grasp of one plant from the tray grid (vision not needed): the gripper comes from the robot's side towards
the cell, pitched down --pitch-deg (26 deg: the pose set by hand for the first grasp), closing axis horizontal,
camera above the gripper, and is moved so that the fingers CLOSE on the stem, taken at the cell centre (+ an
offset) a little above the plug top.
The fingers swing forward when closing (gripper.pin_brake.close_forward_m), so the open gripper stops that far
behind the stem. One phase per call, so that every motion is confirmed:

    $P scripts/65_side_grasp.py --cell C7 observe     # look at the plug from ~0.34 m: its own soil surface height
    $P scripts/65_side_grasp.py --cell C7 plan        # plan from the current pose, saved; no motion
    $P scripts/65_side_grasp.py --cell C7 pregrasp    # open the gripper, go to 6 cm behind the grasp pose
    $P scripts/65_side_grasp.py --cell C7 approach    # slide forward to the grasp pose (slow)
    $P scripts/65_side_grasp.py --cell C7 nudge --dx-mm 4 --dy-mm -2    # fine shift (x forward, y towards higher columns)
    $P scripts/66_gripper.py close                     # close (separate script)
    $P scripts/65_side_grasp.py --cell C7 lift        # lift straight up (slow)
    $P scripts/65_side_grasp.py --cell C7 check       # look at the cell again: is the soil gone (plug pulled out)?
    $P scripts/65_side_grasp.py --cell C7 lower       # put it back down (to the grasp pose, slow)
    $P scripts/66_gripper.py open
    $P scripts/65_side_grasp.py --cell C7 retreat     # back to the pre-grasp
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("phase", choices=["observe", "plan", "pregrasp", "approach", "nudge", "lift", "check", "lower", "retreat"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--height-mm", type=float, default=35.0, help="closing point above an ESTIMATED soil surface (overview / fallback; 25 until 2026-10-26: C4 grasped too low)")
    ap.add_argument("--height-close-mm", type=float, default=17.0, help="closing point above a soil surface measured close up "
                    "(finger bottom ~2.5 mm above it: the stem is grasped where it leaves the soil, at the cell coordinate; 25 until "
                    "2026-10-28: D4 missed, its slightly leaning stem passed below/beside the fingers)")
    ap.add_argument("--max-stem-shift-mm", type=float, default=0.0, help="closing point moved at most this far towards the stem base seen close up "
                    "(0 = off: the depth on a 2-3 mm stem is the soil behind it, so that base estimate is not reliable)")
    ap.add_argument("--rim-clearance-mm", type=float, default=20.0,
                    help="closing point at least this far above the tray top (finger bottom 14.5 mm below it: the fingers must "
                         "pass over the cell walls when the plug sits low in its cell)")
    ap.add_argument("--observe-distance", type=float, default=0.34)
    ap.add_argument("--disagree-mm", type=float, default=8.0, help="overview and close soil heights further apart: take the higher")
    ap.add_argument("--plug-dz-mm", type=float, default=13.0, help="fallback only: plug top above the tray top when no soil was measured")
    ap.add_argument("--offset-mm", type=float, nargs=2, default=None, metavar=("ALONG", "ACROSS"),
                    help="closing point shift from the cell centre: along the approach (horizontal) and along the closing "
                         "axis (default: grasp.side_offset_mm in the config)")
    ap.add_argument("--pitch-deg", type=float, default=26.0, help="approach pitched down from horizontal (wrist >= 7 cm above the tray needs ~26 deg or more)")
    ap.add_argument("--standoff-mm", type=float, default=60.0, help="pre-grasp distance behind the grasp pose")
    ap.add_argument("--ignore-cells", default="", help="comma list: cells known to be empty now (not obstacles)")
    ap.add_argument("--headings-deg", type=float, nargs="+", default=[0, -30, 30, -60, 60, -90, 90],
                    help="approach headings tried in turn, relative to straight out from the robot (first free and safe one wins)")
    ap.add_argument("--stage-up-mm", type=float, default=120.0, help="turn the gripper at this height above the pre-grasp, then go down")
    ap.add_argument("--transit-clearance-mm", type=float, default=120.0, help="grasp centre above the tray top while turning (over the plants)")
    ap.add_argument("--lift-mm", type=float, default=20.0)
    ap.add_argument("--dx-mm", type=float, default=0.0, help="nudge: horizontal, along the approach")
    ap.add_argument("--dy-mm", type=float, default=0.0, help="nudge: along the closing axis")
    ap.add_argument("--dz-mm", type=float, default=0.0, help="nudge: up")
    ap.add_argument("--wrist-clearance-mm", type=float, default=70.0, help="wrist joints / flange centres above the tray top")
    ap.add_argument("--camera-clearance-mm", type=float, default=20.0, help="camera housing above the tray top")
    ap.add_argument("--dynamics", type=float, default=0.03)
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T
    from seedling.wrist import body_points_B

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    tray_top = float(grid["z_top"])
    out = PROJECT_DIR / "experiments" / "grasp"
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"{args.cell}_side_plan.json"
    robot = FrankyRobot(cfg, step_mode=False)
    q_now = robot.q()

    def G_of(q):
        return kinematics.fk(q, F) @ G

    def ik(T_BG, seed):
        q = kinematics.solve_ik_near(T_BG @ inv_T(G), seed, F, margin)
        if q is None or np.linalg.norm(G_of(q)[:3, 3] - T_BG[:3, 3]) > 0.0015:
            return None
        return q

    def check(qa, qb, n=61, gripper_floor=None):
        lows = {"wrist": np.inf, "camera": np.inf, "gripper": np.inf}
        dev = 0.0
        pa, pb = G_of(qa)[:3, 3], G_of(qb)[:3, 3]
        for s in np.linspace(0, 1, n):
            q = qa + s * (qb - qa)
            pts = body_points_B(q, T_FC, cfg)
            lows["wrist"] = min(lows["wrist"], float(pts["wrist joints"][:, 2].min()), float(pts["flange"][:, 2].min()))
            lows["camera"] = min(lows["camera"], float(pts["camera housing"][:, 2].min()))
            g = G_of(q)[:3, 3]
            lows["gripper"] = min(lows["gripper"], float(g[2]))
            dev = max(dev, float(np.linalg.norm(np.cross(g - pa, pb - pa)) / max(np.linalg.norm(pb - pa), 1e-9)))
        bad = []
        if lows["wrist"] < tray_top + args.wrist_clearance_mm / 1000:
            bad.append(f"wrist/flange {lows['wrist']:.3f}")
        if lows["camera"] < tray_top + args.camera_clearance_mm / 1000:
            bad.append(f"camera {lows['camera']:.3f}")
        if gripper_floor is not None and lows["gripper"] < gripper_floor:
            bad.append(f"grasp centre {lows['gripper']:.3f}")
        step = float(np.degrees(np.abs(qb - qa)).max())
        return step, lows, dev, bad

    def report(name, qa, qb, gripper_floor=None):
        step, lows, dev, bad = check(qa, qb, gripper_floor=gripper_floor)
        print(f"  {name:10s}: max joint change {step:5.1f} deg; lowest wrist/flange {lows['wrist']:.3f}, camera {lows['camera']:.3f}, "
              f"grasp centre {lows['gripper']:.3f} "
              f"(tray top {tray_top:.3f}); grasp centre off the straight line {dev * 1000:4.1f} mm" + (f"  <-- TOO LOW {bad}" if bad else ""))
        return not bad

    def move(q_target, dyn, name):
        r = robot.move_joints(q_target, dynamics=dyn, description=name)
        g = G_of(robot.q())[:3, 3]
        print(f"{name}: {'ok' if r.ok else 'FAILED ' + r.reason}; grasp centre (open gap centre) now {np.round(g, 4).tolist()}")
        return r.ok

    obs_file = out / f"{args.cell}_observe.json"

    def look(tag):
        """Observation pose for this cell (camera ~0.34 m away, looking down), picture, soil surface of the cell."""
        import copy
        import time

        from seedling.calibration import flange_pose
        from seedling.camera import ZedCamera
        from seedling.plugsense import measure_plug
        from seedling.wrist import plan_observation

        c = np.array(grid["cells"][args.cell], float)
        guess = (grid.get("soil_z_overview") or {}).get(args.cell) or tray_top + args.plug_dz_mm / 1000
        c2 = copy.deepcopy(cfg)
        o = c2["tray"]["observe"]
        # all elevations compared: the least joint change wins (a 65 deg view can need a 70 deg wrist turn where an
        # 80 deg view needs 45); 2 deg of joint change per place down the list keeps 65 on a near tie
        o.update(distance_m=args.observe_distance, elevation_deg=[65, 55, 70, 75, 80], azimuth_offsets_deg=[0, -20, 20, -40, 40],
                 min_clearance_m=0.15, all_elevations=True, elevation_penalty_deg=2.0)
        target = np.array([c[0], c[1], guess])
        pl = plan_observation(target, q_now, T_FC, guess + 0.15, c2)
        if not pl.ok:
            print("no safe observation pose:", pl.tried[-3:])
            return None
        print(f"{tag}: observation pose {pl.desc}, max joint change {pl.max_joint_step_deg:.0f} deg, lowest {pl.lowest_part} z={pl.lowest_z:.3f}")
        r = robot.move_joints(pl.q, dynamics=args.dynamics, description=f"observe {args.cell}")
        if not r.ok:
            print("motion failed:", r.reason)
            return None
        time.sleep(0.5)
        cam = ZedCamera(cfg)
        try:
            f = cam.grab(n_average=5)
        finally:
            cam.close()
        T_BC = flange_pose(robot) @ T_FC
        f.meta["q"] = robot.q().tolist()
        f.save(out / f"{args.cell}_{tag}", save_png=False)
        m = measure_plug(f, T_BC, c[:2], cfg)
        print(f"{tag}: soil " + (f"z {m['soil_z']:.3f} ({(m['soil_z'] - tray_top) * 1000:+.0f} mm vs the tray top), {m['n']} points" if m["ok"] else m["status"]))
        # where the stem leaves the soil: green points just above the soil near the cell centre (the plug sits at
        # the cell centre, but the stem can come out off-centre and lean; D4 2026-10-28 was missed that way)
        stem = None
        if m["ok"]:
            from seedling.stemline import _mask
            from seedling.transforms import apply_T as _ap

            okg = np.isfinite(f.xyz).all(-1) & _mask(f.rgb, cfg["stemline"])
            G3 = _ap(T_BC, f.xyz[okg])
            d = G3[:, :2] - c[:2]
            sel = (np.hypot(*d.T) < 0.025) & (G3[:, 2] > m["soil_z"]) & (G3[:, 2] < m["soil_z"] + 0.015)
            if sel.sum() >= 80:
                off_xy = np.median(d[sel], axis=0)
                stem = {"offset_mm": [round(float(v) * 1000, 1) for v in off_xy], "n": int(sel.sum())}
                print(f"{tag}: stem base {stem['offset_mm']} mm from the cell centre ({stem['n']} green points just above the soil)")
        return {"q": robot.q().tolist(), "measure": m, "stem": stem, "overview": (grid.get("soil_z_overview") or {}).get(args.cell)}

    if args.phase == "observe":
        rec = look("observe")
        if rec is None:
            return 1
        obs_file.write_text(json.dumps(rec, indent=1))
        return 0

    if args.phase == "check":
        plan_q = {k: np.array(v) for k, v in json.loads(plan_file.read_text())["q"].items()}
        if float(np.degrees(np.abs(q_now - plan_q["lift"])).max()) > 1.0:
            print("refused: check starts at the lift pose")
            return 1
        before = json.loads(obs_file.read_text()) if obs_file.exists() else None
        rec = look("check")
        if rec is None:
            return 1
        # back to the lift pose (the plant still held), so that 'lower' can put it back
        r_back = robot.move_joints(plan_q["lift"], dynamics=args.dynamics, description="back to the lift pose")
        print("back to the lift pose:", "ok" if r_back.ok else "FAILED " + r_back.reason)
        m, z0 = rec["measure"], (before or {}).get("soil_used")
        if z0 is None:
            print("no soil height from before the grasp: cannot judge")
        elif m["ok"] and abs(m["soil_z"] - z0) < 0.010:
            print(f"CHECK: soil still at {m['soil_z']:.3f} in the cell (before {z0:.3f}) -> the plug was NOT pulled out")
        else:
            print(f"CHECK: no soil left at {z0:.3f} in the cell -> plug pulled out" + (f" (soil seen at {m['soil_z']:.3f})" if m["ok"] else ""))
        (out / f"{args.cell}_check.json").write_text(json.dumps(rec, indent=1))
        return 0

    if args.phase == "plan":
        off = args.offset_mm if args.offset_mm is not None else cfg["grasp"].get("side_offset_mm", [0.0, 0.0])
        c = np.array(grid["cells"][args.cell], float)
        # soil surface of THIS plug: close measurement, overview measurement, fallback tray top + plug_dz
        z_ov = (grid.get("soil_z_overview") or {}).get(args.cell)
        obs = json.loads(obs_file.read_text()) if obs_file.exists() else None
        z_near = obs["measure"]["soil_z"] if obs and obs["measure"].get("ok") else None
        if z_near is not None and z_ov is not None and abs(z_near - z_ov) > args.disagree_mm / 1000:
            soil_z, src = max(z_near, z_ov), f"overview {z_ov:.3f} and close {z_near:.3f} disagree: the higher"
        elif z_near is not None:
            soil_z, src = z_near, "close measurement" + (f" (overview {z_ov:.3f})" if z_ov is not None else "")
        elif z_ov is not None:
            # the overview sees the plug from far and at an angle, often half under leaves: it tends to read low
            # (C4, 2026-10-26: grasped too low). Never below the usual plug height then
            fb = tray_top + args.plug_dz_mm / 1000
            soil_z, src = (z_ov, "overview measurement (no close one)") if z_ov >= fb else \
                (fb, f"overview {z_ov:.3f} (no close one) below the usual plug height: tray top + plug_dz")
        else:
            soil_z, src = tray_top + args.plug_dz_mm / 1000.0, "NOT MEASURED: tray top + plug_dz"
        print(f"soil surface of {args.cell}: z {soil_z:.3f} ({(soil_z - tray_top) * 1000:+.0f} mm vs the tray top) from {src}")
        if obs is not None:
            obs["soil_used"] = soil_z
            obs_file.write_text(json.dumps(obs, indent=1))
        # closing height: 17 mm above a soil surface measured close up (sure), 35 mm when it is only estimated
        sure = z_near is not None and src.startswith("close")
        height = args.height_close_mm if sure else args.height_mm
        args.height_mm = height  # used in the messages below
        z_close = max(soil_z + height / 1000.0, tray_top + args.rim_clearance_mm / 1000.0)
        stem = (obs or {}).get("stem")
        if stem and args.max_stem_shift_mm > 0:
            sh = np.array(stem["offset_mm"]) / 1000.0
            n_ = np.linalg.norm(sh)
            if n_ > args.max_stem_shift_mm / 1000:
                sh *= args.max_stem_shift_mm / 1000 / n_
            c = c.copy()
            c[:2] += sh
            print(f"closing point moved {np.round(sh * 1000, 1).tolist()} mm to the stem base seen close up")
        if z_close > soil_z + args.height_mm / 1000.0 + 1e-6:
            print(f"the soil lies {(tray_top - soil_z) * 1000:.0f} mm below the tray top: closing point raised to "
                  f"{args.rim_clearance_mm:.0f} mm above the tray top ({(z_close - soil_z) * 1000:.0f} mm above the soil)")
        fwd = float(cfg["gripper"]["pin_brake"].get("close_forward_m", 0.0))
        pt = np.radians(args.pitch_deg)
        # approach heading: straight out from the robot if nothing grows in the way, else turned up to 90 deg (a
        # tray with its long side along the reach has the plants in lines along the approach). In the way = another
        # occupied cell within 30 mm of the gripper's path over the 14 cm behind the stem (fingers, then the body
        # rising at the pitch angle; the plants' stems and leaves stand there)
        gone = set(filter(None, args.ignore_cells.split(",")))
        obstacles = [k for k in (grid.get("planted") or grid.get("occupied", [])) if k != args.cell and k not in gone]
        occupied = [np.array(grid["cells"][k][:2]) for k in obstacles]
        radial = np.arctan2(c[1], c[0])

        def blocked(yaw):
            back = -np.array([np.cos(yaw), np.sin(yaw)])
            hits = []
            for k, o in zip(obstacles, occupied):
                d = o - c[:2]
                s_ = float(d @ back)
                lat = abs(float(d @ np.array([-back[1], back[0]])))
                # behind the stem: the gripper's path; beside it: the open fingers (44 mm gap + finger, and stems
                # lean): a neighbour 39 mm off the line got caught instead of the target (D4 -> C4, 2026-10-28)
                if (0.015 < s_ < 0.14 and lat < 0.030) or (-0.03 < s_ <= 0.015 and lat < 0.042):
                    hits.append(k)
            return hits

        chosen = None
        for dyaw in args.headings_deg:
            yaw = radial + np.radians(dyaw)
            hits = blocked(yaw)
            if hits:
                print(f"  heading {dyaw:+.0f} deg: plants in the way ({', '.join(hits)})")
                continue
            a = np.array([np.cos(pt) * np.cos(yaw), np.cos(pt) * np.sin(yaw), -np.sin(pt)])  # approach
            across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # closing axis, horizontal (as on the first grasp: towards -y)
            R = np.column_stack([np.cross(across, a), across, a])
            a_h = np.array([a[0], a[1], 0.0]) / np.linalg.norm(a[:2])
            close_pt = np.array([c[0], c[1], z_close]) + a_h * off[0] / 1000.0 + across * off[1] / 1000.0
            grasp = close_pt - a * fwd  # the open gap centre: the fingers swing forward onto close_pt
            pre = grasp - a * args.standoff_mm / 1000.0
            lift = grasp + [0, 0, args.lift_mm / 1000.0]
            stage = pre + [0, 0, args.stage_up_mm / 1000.0]
            q_st = ik(make_T(R, stage), q_now)
            if q_st is None:  # far from the current joints: any solution, then the rest near it
                q_st = kinematics.solve_ik(make_T(R, stage) @ inv_T(G), q_now, F, margin)
            q_pre = ik(make_T(R, pre), q_st) if q_st is not None else None
            q_gr = ik(make_T(R, grasp), q_pre) if q_pre is not None else None
            q_li = ik(make_T(R, lift), q_gr) if q_gr is not None else None
            if q_li is None:
                print(f"  heading {dyaw:+.0f} deg: not reachable")
                continue
            floor_g = tray_top + args.transit_clearance_mm / 1000.0
            near = float(np.degrees(np.abs(q_st - q_now)).max()) < 5.0
            print(f"  heading {dyaw:+.0f} deg from straight out:")
            ok = (True if near else report("to stage", q_now, q_st, gripper_floor=floor_g)) & report("down", q_st, q_pre) \
                & report("approach", q_pre, q_gr) & report("lift", q_gr, q_li) & report("retreat", q_li, q_pre)
            chosen = (dyaw, a, across, R, close_pt, q_st, q_pre, q_gr, q_li, ok)
            if ok:
                break
        if chosen is None:
            print("not reachable from any heading without going through other plants")
            return 1
        dyaw, a, across, R, close_pt, q_st, q_pre, q_gr, q_li, ok = chosen
        grasp = close_pt - a * fwd
        cam_above = (kinematics.fk_flange(q_gr) @ T_FC)[2, 3] > grasp[2]
        print(f"cell {args.cell}: closing point {np.round(close_pt, 4).tolist()} = cell centre + {off[0]:+.0f} mm along, {off[1]:+.0f} mm across "
              f"(closing axis {np.round(across, 2).tolist()}), {args.height_mm:.0f} mm above the soil; finger bottom ~"
              f"{(close_pt[2] - 0.0145 - soil_z) * 1000:.0f} mm above it")
        print(f"approach pitched {np.degrees(np.arcsin(-a[2])):.0f} deg down, heading {dyaw:+.0f} deg from straight out; open gripper stops "
              f"{fwd * 1000:.0f} mm behind the closing point; camera {'above' if cam_above else 'BELOW'} the gripper")
        plan_file.write_text(json.dumps({"cell": args.cell, "close_point": close_pt.tolist(), "offset_mm": list(off), "R": R.tolist(),
                                         "q": {"start": q_now.tolist(), "stage": q_st.tolist(), "pregrasp": q_pre.tolist(), "grasp": q_gr.tolist(), "lift": q_li.tolist()},
                                         "safe": bool(ok)}, indent=1))
        print("plan saved" if ok else "plan saved but NOT SAFE: the motion phases will refuse it", plan_file.relative_to(PROJECT_DIR))
        return 0 if ok else 1

    if not plan_file.exists():
        print("no plan: run the 'plan' phase first")
        return 1
    plan = json.loads(plan_file.read_text())
    if not plan["safe"]:
        print("the saved plan is not safe")
        return 1
    q = {k: np.array(v) for k, v in plan["q"].items()}

    def at(name, tol_deg=1.0):
        d = float(np.degrees(np.abs(q_now - q[name])).max())
        if d > tol_deg:
            print(f"refused: not at the {name} pose ({d:.1f} deg away)")
        return d <= tol_deg

    if args.phase == "pregrasp":
        if not at("start", 2.0):
            return 1
        from seedling.gripper import make_gripper
        import termios

        g = make_gripper(cfg, sim=False, step_mode=False)
        g.open()
        ser = g.dev.ser
        g.dev.running = False
        attrs = termios.tcgetattr(ser.fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(ser.fd, termios.TCSANOW, attrs)
        ser.close()
        if not move(q["stage"], args.dynamics, "turn above the pre-grasp"):
            return 1
        return 0 if move(q["pregrasp"], args.dynamics, "down to the pre-grasp") else 1
    if args.phase == "approach":
        if not at("pregrasp"):
            return 1
        return 0 if move(q["grasp"], min(args.dynamics, 0.02), "approach") else 1
    if args.phase == "nudge":
        T = G_of(q_now)
        a = T[:3, 2]
        a_h = np.array([a[0], a[1], 0.0]) / np.linalg.norm(a[:2])
        p = T[:3, 3] + a_h * args.dx_mm / 1000 + T[:3, 1] * args.dy_mm / 1000 + np.array([0, 0, args.dz_mm / 1000])
        if np.linalg.norm(p - T[:3, 3]) > 0.015:
            print("refused: nudges are limited to 15 mm")
            return 1
        qn = ik(make_T(T[:3, :3], p), q_now)
        if qn is None or not report("nudge", q_now, qn):
            print("refused")
            return 1
        if not move(qn, 0.02, "nudge"):
            return 1
        # keep the plan consistent: the grasp pose is where we are now, lift from here
        plan["q"]["grasp"] = qn.tolist()
        ql = ik(make_T(T[:3, :3], p + [0, 0, args.lift_mm / 1000.0]), qn)
        if ql is not None:
            plan["q"]["lift"] = ql.tolist()
        plan.setdefault("nudges_mm", []).append([args.dx_mm, args.dy_mm, args.dz_mm])
        plan_file.write_text(json.dumps(plan, indent=1))
        return 0
    if args.phase == "lift":
        if not at("grasp"):
            return 1
        return 0 if move(q["lift"], min(args.dynamics, 0.02), "lift") else 1
    if args.phase == "lower":
        if not at("lift"):
            return 1
        return 0 if move(q["grasp"], min(args.dynamics, 0.02), "put back down") else 1
    # retreat
    if not any(np.degrees(np.abs(q_now - q[k])).max() <= 2.0 for k in ("lift", "grasp")):
        print("refused: neither at the lift nor at the grasp pose")
        return 1
    return 0 if move(q["pregrasp"], args.dynamics, "back to the pre-grasp") else 1


if __name__ == "__main__":
    raise SystemExit(main())
