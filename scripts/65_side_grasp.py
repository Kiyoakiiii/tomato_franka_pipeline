#!/usr/bin/env python
"""Side grasp of one plant from the tray grid (vision not needed): the gripper comes from the robot's side towards
the cell, pitched down as little as is safe (--pitch-deg / grasp.side_pitch_deg, flattest first: the fingers
parallel to the tray as asked on 2026-11-02; 26 deg alone before), closing axis horizontal,
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
    ap.add_argument("--leafy-cover", type=float, default=0.35, help="share of green over the plug top above which the plug counts as leafy")
    ap.add_argument("--height-leafy-mm", type=float, default=14.5, help="closing point above the soil for a leafy plug")
    ap.add_argument("--finger-clearance-mm", type=float, default=3.0, help="open finger bottom above the soil under its track (neighbouring plugs)")
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
    ap.add_argument("--pitch-deg", type=float, nargs="+", default=None,
                    help="approach pitches down from horizontal, tried in turn (default grasp.side_pitch_deg: the flattest first, the "
                         "fingers parallel to the tray as asked on 2026-11-02; a flat hand puts the wrist at the stem's height, so most "
                         "cells need more; 26 deg alone until then)")
    ap.add_argument("--standoff-mm", type=float, default=60.0, help="pre-grasp distance behind the grasp pose")
    ap.add_argument("--ignore-cells", default="", help="comma list: cells known to be empty now (not obstacles)")
    ap.add_argument("--headings-deg", type=float, nargs="+", default=[0.0] + [s_ * d_ for d_ in range(5, 91, 5) for s_ in (-1, 1)],
                    help="approach headings tried in turn, relative to straight out from the robot (first free and safe one wins)")
    ap.add_argument("--stage-up-mm", type=float, default=120.0, help="turn the gripper at this height above the pre-grasp, then go down")
    ap.add_argument("--self-clearance-mm", type=float, default=90.0, help="hand (flange, camera, gripper) from the arm's own upper arm and "
                    "forearm axes on every move: Franka's self-collision reflex stopped moves at 69 mm (2026-11-02)")
    ap.add_argument("--max-posture-deg", type=float, default=None, help="posture limit (default grasp.max_posture_deg, 60): base turn "
                    "from the plant, elbow sideways swing, wrist twist")
    ap.add_argument("--time-budget-s", type=float, default=15.0, help="plan: planning stops after this long (the best found, or why not)")
    ap.add_argument("--max-branches", type=int, default=4, help="of all joint solutions of the grasp pose (the arm's own family, joint 7 "
                    "swept in 10 deg steps), at most this many tried per heading (clear at the grasp first, then least joint change)")
    ap.add_argument("--lift-mm", type=float, default=20.0)
    ap.add_argument("--dx-mm", type=float, default=0.0, help="nudge: horizontal, along the approach")
    ap.add_argument("--dy-mm", type=float, default=0.0, help="nudge: along the closing axis")
    ap.add_argument("--dz-mm", type=float, default=0.0, help="nudge: up")
    ap.add_argument("--wrist-clearance-mm", type=float, default=None, help="wrist joints / flange centres above what lies under them "
                    "(default grasp.side_wrist_clearance_mm: 70 for the pitched grasp until 2026-11-02; a flat hand has its flange at "
                    "the stem's height, the user judged the arm clear of the tray then)")
    ap.add_argument("--wrist-clearance-off-tray-mm", type=float, default=None,
                    help="the same away from the tray (over the disc and the table): default grasp.side_wrist_clearance_off_tray_mm")
    ap.add_argument("--wrist-reach-mm", type=float, default=0.0, help="the wrist housings reach this far sideways (see 71)")
    ap.add_argument("--camera-clearance-mm", type=float, default=20.0, help="camera housing above what lies under it")
    ap.add_argument("--max-line-dev-mm", type=float, default=8.0, help="the moves at the plant stay this close to their straight line")
    ap.add_argument("--disc-keepout-mm", type=float, default=0.0, help="while grasping, the arm (wrist, flange, gripper body, "
                    "camera) stays this far above the disc top wherever it is over the disc (2026-11-04: touched the disc in "
                    "front of the tray, the wrist centres 10 cm above it)")
    ap.add_argument("--turn-search-s", type=float, default=3.0, help="plan: path search time for the turns that are no single joint "
                    "move (all candidates together; the hand stays high and near the straight line)")
    ap.add_argument("--from-observe", action="store_true", help="plan: from the observation pose the 'observe' phase would go to "
                    "(not from where the arm is): what the cycle will plan from after observing (72 chooses plants this way)")
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

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T
    from seedling import cartesian
    from seedling.planner import plan_path

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    tray_top = float(grid["z_top"])
    disc_file = PROJECT_DIR / "calibration" / "disc.json"
    # wrist and camera clearances are over what lies under them (tray top, disc top or table), as in 71
    out = PROJECT_DIR / "experiments" / "grasp"
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"{args.cell}_side_plan.json"
    from seedling.world import World

    disc_now = json.loads(disc_file.read_text()) if disc_file.exists() else None
    world = World(cfg, T_FC, grid, disc_now)  # the moves at the plant (plants not checked there)
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
        """a move at the plant: the arm, camera and gripper body clear of the table, tray, disc and of the arm itself
        (seedling.world: Franka's own link meshes); the fingers and the plants are not checked here (the fingers go
        to the plant; the heading is chosen so that no other plant stands in the gripper's way, see blocked())"""
        m, where = world.worst(qa, qb, fingers=False, plants=False)
        dev = 0.0
        pa, pb = G_of(qa)[:3, 3], G_of(qb)[:3, 3]
        glow = np.inf
        for s_ in np.linspace(0, 1, 21):
            g = G_of(qa + s_ * (qb - qa))[:3, 3]
            glow = min(glow, float(g[2]))
            dev = max(dev, float(np.linalg.norm(np.cross(g - pa, pb - pa)) / max(np.linalg.norm(pb - pa), 1e-9)))
        bad = []
        if m < 0:
            bad.append(f"{where} {m * 1000:+.0f} mm")
        if gripper_floor is not None and glow < gripper_floor:
            bad.append(f"grasp centre {glow:.3f}")
        if dev > args.max_line_dev_mm / 1000:
            bad.append(f"grasp centre {dev * 1000:.0f} mm off the straight line")
        step = float(np.degrees(np.abs(qb - qa)).max())
        return step, {"margin": m, "where": where, "gripper": glow}, dev, bad

    def report(name, qa, qb, gripper_floor=None, out=None):
        step, lows, dev, bad = check(qa, qb, gripper_floor=gripper_floor)
        (print if out is None else out.append)(f"  {name:10s}: max joint change {step:5.1f} deg; smallest margin {lows['margin'] * 1000:+.0f} mm "
              f"({lows['where']}); grasp centre down to {lows['gripper']:.3f} (tray top {tray_top:.3f}), {dev * 1000:4.1f} mm off the "
              f"straight line" + (f"  <-- NOT SAFE {bad}" if bad else ""))
        return not bad

    def move(q_target, dyn, name):
        r = robot.move_joints(q_target, dynamics=dyn, description=name)
        g = G_of(robot.q())[:3, 3]
        print(f"{name}: {'ok' if r.ok else 'FAILED ' + r.reason}; grasp centre (open gap centre) now {np.round(g, 4).tolist()}")
        return r.ok

    obs_file = out / f"{args.cell}_observe.json"

    def observe_pose(q_from):
        """the observation pose for this cell (camera ~0.34 m away, looking down) reached from q_from"""
        import copy

        from seedling.wrist import plan_observation

        c = np.array(grid["cells"][args.cell], float)
        guess = (grid.get("soil_z_overview") or {}).get(args.cell) or tray_top + args.plug_dz_mm / 1000
        c2 = copy.deepcopy(cfg)
        o = c2["tray"]["observe"]
        # all elevations compared: the least joint change wins (a 65 deg view can need a 70 deg wrist turn where an
        # 80 deg view needs 45); 2 deg of joint change per place down the list keeps 65 on a near tie
        o.update(distance_m=args.observe_distance, elevation_deg=[65, 55, 70, 75, 80], azimuth_offsets_deg=[0, -20, 20, -40, 40],
                 min_clearance_m=0.15, all_elevations=True, elevation_penalty_deg=2.0)
        return plan_observation(np.array([c[0], c[1], guess]), q_from, T_FC, guess + 0.15, c2)

    def look(tag):
        """Observation pose for this cell (camera ~0.34 m away, looking down), picture, soil surface of the cell."""
        import time

        from seedling.calibration import flange_pose
        from seedling.camera import ZedCamera
        from seedling.plugsense import measure_plug

        c = np.array(grid["cells"][args.cell], float)
        pl = None
        # the observation pose worked out when this plant was chosen (72: 65 plan --from-observe), from this same pose
        if plan_file.exists():
            try:
                old_ = json.loads(plan_file.read_text())
                if ("observe_q" in old_ and float(np.degrees(np.abs(np.array(old_["observe_from"]) - q_now)).max()) < 0.5
                        and np.linalg.norm(np.array(old_["observe_cell_xyz"]) - np.array(grid["cells"][args.cell], float)) < 0.003):
                    from seedling.wrist import ObservationPlan, path_clearance

                    q_o = np.array(old_["observe_q"])
                    z_, part_ = path_clearance(q_now, q_o, T_FC, cfg)
                    pl = ObservationPlan(True, q_o, None, old_["observe_desc"] + " (worked out when the plant was chosen)",
                                         float(np.degrees(np.abs(q_o - q_now)).max()), z_, part_)
            except Exception:  # noqa: BLE001
                pl = None
        if pl is None:
            pl = observe_pose(q_now)
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
        # every close-up kept (the file above is overwritten per cell): training data for a stem fine-tune model
        (out / "closeups").mkdir(exist_ok=True)
        from seedling.camera import save_in_background

        save_in_background([(f, out / "closeups" / f"{args.cell}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}", False)])
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
        # leaves lying on the plug: the share of green over its top (C4 2026-10-28: 43%, a short stem under a rosette,
        # grasped 18 mm up the fingers held only leaves, which slipped; the others 0-25%)
        cover = None
        if m["ok"]:
            from seedling.stemline import _mask as _gm
            from seedling.transforms import apply_T as _ap2

            okc = np.isfinite(f.xyz).all(-1)
            Pc = _ap2(T_BC, f.xyz[okc])
            gc_ = _gm(f.rgb, cfg["stemline"])[okc]
            sel = (np.hypot(Pc[:, 0] - c[0], Pc[:, 1] - c[1]) < 0.015) & (Pc[:, 2] > m["soil_z"] - 0.005) & (Pc[:, 2] < m["soil_z"] + 0.05)
            if sel.sum() >= 50:
                cover = round(float(gc_[sel].mean()), 2)
                print(f"{tag}: leaves cover {cover * 100:.0f}% of the plug top")
        return {"q": robot.q().tolist(), "measure": m, "stem": stem, "leaf_cover": cover, "overview": (grid.get("soil_z_overview") or {}).get(args.cell)}

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
        obs_from = None
        if args.from_observe:
            pl = observe_pose(q_now)
            if not pl.ok:
                print("no safe observation pose:", pl.tried[-3:])
                return 1
            print(f"planned from the observation pose ({pl.desc}, max joint change {pl.max_joint_step_deg:.0f} deg)")
            obs_from = {"observe_from": q_now.tolist(), "observe_q": pl.q.tolist(), "observe_desc": pl.desc,
                        "observe_cell_xyz": [float(v) for v in grid["cells"][args.cell]]}
            q_now = pl.q
        off = args.offset_mm if args.offset_mm is not None else cfg["grasp"].get("side_offset_mm", [0.0, 0.0])
        c = np.array(grid["cells"][args.cell], float)
        # soil surface of THIS plug: close measurement, overview measurement, fallback tray top + plug_dz
        z_ov = (grid.get("soil_z_overview") or {}).get(args.cell)
        obs = json.loads(obs_file.read_text()) if obs_file.exists() else None
        z_near = obs["measure"]["soil_z"] if obs and obs["measure"].get("ok") else None
        off_c = obs and np.hypot(*obs["measure"].get("shift_mm", [0, 0])) > 12.0
        if z_near is not None and z_ov is not None and off_c and z_near > z_ov + 0.003:
            # the close soil patch lies off the cell centre (the plug's rim or a neighbour seen, D6 2026-10-28: +27 mm
            # vs the tray top, 15 mm off centre, grasped above the stem): the overview value, at the close-up height
            soil_z, src = z_ov, f"overview {z_ov:.3f}: the close {z_near:.3f} came from soil {np.hypot(*obs['measure']['shift_mm']):.0f} mm off the cell centre"
        elif z_near is not None and z_ov is not None and abs(z_near - z_ov) > args.disagree_mm / 1000:
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
        # (the close one also when it disagrees with the overview and is the higher: C5 2026-10-28 got 35 mm that way)
        sure = z_near is not None and (soil_z == z_near or src.startswith(f"overview {z_ov:.3f}: the close"))
        height = args.height_close_mm if sure else args.height_mm
        cov = (obs or {}).get("leaf_cover")
        if sure and cov is not None and cov > args.leafy_cover:
            # leaves over the plug: the fingers' bottom at the soil, so they pinch the stem where it leaves the soil
            height = min(height, args.height_leafy_mm)
            print(f"leaves cover {cov * 100:.0f}% of the plug top: closing {height:.1f} mm above the soil (finger bottom at it)")
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
        pitches = args.pitch_deg or cfg["grasp"].get("side_pitch_deg", [26.0])
        # approach heading: straight out from the robot if nothing grows in the way, else turned up to 90 deg (a
        # tray with its long side along the reach has the plants in lines along the approach). In the way = another
        # occupied cell within 30 mm of the gripper's path over the 14 cm behind the stem (fingers, then the body
        # rising at the pitch angle; the plants' stems and leaves stand there); a flatter hand stays low for longer
        # behind the stem: up to 30 cm at 0 deg, 45 mm wide past the first 14 cm (the flange and wrist)
        gone = set(filter(None, args.ignore_cells.split(",")))
        # soil lying at the bottom of a cell (fallen scraps, 2026-10-28 E2/F2: 40 mm below the tray top) stands in no
        # one's way: an obstacle is a plant (green over it) or soil reaching within 20 mm of the tray top
        sov, green = grid.get("soil_z_overview") or {}, set(grid.get("green") or [])
        obstacles = [k for k in (grid.get("planted") or grid.get("occupied", [])) if k != args.cell and k not in gone
                     and (k in green or sov.get(k) is None or sov[k] > tray_top - 0.020)]
        occupied = [np.array(grid["cells"][k][:2]) for k in obstacles]
        radial = np.arctan2(c[1], c[0])

        def blocked(yaw, pitch_deg):
            reach = 0.14 if pitch_deg >= 26 else min(0.30, 0.14 * np.tan(np.radians(26)) / max(np.tan(np.radians(pitch_deg)), 1e-6))
            back = -np.array([np.cos(yaw), np.sin(yaw)])
            hits = []
            for k, o in zip(obstacles, occupied):
                d = o - c[:2]
                s_ = float(d @ back)
                lat = abs(float(d @ np.array([-back[1], back[0]])))
                # behind the stem: the gripper's path; beside it: the open fingers (44 mm gap + finger, and stems
                # lean): a neighbour 39 mm off the line got caught instead of the target (D4 -> C4, 2026-10-28)
                if (0.015 < s_ < 0.14 and lat < 0.030) or (-0.03 < s_ <= 0.015 and lat < 0.042) or (0.14 <= s_ < reach and lat < 0.045):
                    hits.append(k)
            return hits

        # the open fingers (+-22 mm across, a finger wide) pass over the NEIGHBOURING plugs, which can stand higher than this
        # one (D5 2026-10-28: grasped 17 mm above its soil, the fingers hit a plug 6-9 mm higher and moved the tray). From
        # the close-up, the soil under each finger track (from 45 mm behind the closing point to 15 mm past it; taken as
        # flat: a neighbour 15-45 mm behind and 8-9 mm higher was hit, so the pitch gives no credit there): the finger
        # bottom must clear it by --finger-clearance-mm, so the closing point is raised as far as that needs (per heading)
        soilP = None
        npz = out / f"{args.cell}_observe.npz"
        if sure and npz.exists():
            from seedling.camera import Frame
            from seedling.plugsense import substrate_mask
            from seedling.stemline import _mask
            from seedling.transforms import apply_T

            fr = Frame.load(npz)
            okp = np.isfinite(fr.xyz).all(-1)
            okp &= substrate_mask(fr.rgb, cfg["stemline"]) & ~_mask(fr.rgb, cfg["stemline"])
            soilP = apply_T(kinematics.fk_flange(np.array(obs["q"])) @ T_FC, fr.xyz[okp])

        def finger_floor(yaw):
            """lowest closing-point height whose finger bottom clears the soil under both finger tracks (None: no data)"""
            if soilP is None:
                return None
            ah = np.array([np.cos(yaw), np.sin(yaw)])
            d = soilP[:, :2] - c[:2]
            s_ = d @ ah  # ahead of the closing point (+) / behind it, towards the robot (-)
            lat = np.abs(d @ np.array([-ah[1], ah[0]]))
            m = (lat > 0.012) & (lat < 0.034) & (s_ > -0.045) & (s_ < 0.015)
            if m.sum() < 30:
                return None
            need = soilP[m, 2] + args.finger_clearance_mm / 1000 + 0.0145
            return float(np.percentile(need, 98))

        accept_first = [False]
        # the plan made when this plant was chosen (72: from the predicted observation pose) is used as it is when the
        # arm is at that pose now and its closing point still comes out the same (within 3 mm, with the close-up's
        # soil and the neighbours' soil under the finger tracks) (2026-11-04: planning twice per plant cost ~15 s)
        if plan_file.exists() and not args.from_observe:
            try:
                old_ = json.loads(plan_file.read_text())
                same_start = float(np.degrees(np.abs(np.array(old_["q"]["start"]) - q_now)).max()) < 1.0
                same_cell = "cell_xy" in old_ and float(np.hypot(*(np.array(old_["cell_xy"]) - c[:2]))) < 0.002
                if old_.get("safe") and same_start and same_cell and "yaw" in old_:
                    z_ff_ = finger_floor(float(old_["yaw"]))
                    z_new = max(z_close, z_ff_) if z_ff_ is not None else z_close
                    if abs(z_new - float(old_["close_point"][2])) < 0.003 and abs(float(old_.get("tray_top", -1)) - tray_top) < 0.002:
                        print(f"the plan made when {args.cell} was chosen still holds (closing point within "
                              f"{abs(z_new - float(old_['close_point'][2])) * 1000:.0f} mm): used as it is")
                        return 0
                    print(f"planned again: the closing point moved {(z_new - float(old_['close_point'][2])) * 1000:+.0f} mm since the plant was chosen")
                    # the heading chosen then (and its neighbours) first, and the first safe plan is taken
                    d0 = float(np.degrees((float(old_["yaw"]) - radial + np.pi) % (2 * np.pi) - np.pi))
                    hs = sorted(args.headings_deg, key=lambda h_: abs(h_ - d0))
                    args.headings_deg[:] = hs
                    accept_first[0] = True
            except Exception:  # noqa: BLE001  (an old plan file: plan again)
                pass

        # the flattest pitch first (fingers parallel to the tray), every heading at it, then the next pitch: the first
        # fully safe plan wins. Only the chosen plan's moves are printed in full, the others as one line each
        chosen, raised, best = None, set(), None
        # posture limit (config grasp.max_posture_deg): the base at most this far turned from the plant, the elbow swung
        # sideways or the wrist twisted at most this much; beyond it the plant is reported, not grasped (2026-11-04:
        # close to the robot a flat grasp needed the wrist flipped and a 183 deg joint turn)
        max_posture = float(args.max_posture_deg if args.max_posture_deg is not None else cfg["grasp"].get("max_posture_deg", 60.0))
        over_limit = [np.inf]
        turn_budget = [args.turn_search_s]  # path search time left for the turns (all headings and solutions)
        import time as _tt

        deadline = _tt.time() + args.time_budget_s
        # the free moves (the turn to the stage) are checked against seedling.scene: surfaces, the plants in the tray
        # (this one too) and in the disc, the arm itself
        # the free moves (the turn to the stage) are checked against everything, the plants too (this one included)
        scene = World(cfg, T_FC, grid, disc_now, tray_plants=obstacles + [args.cell])

        for pitch in pitches:
            pt = np.radians(pitch)
            for dyaw in args.headings_deg:
                if (best is not None and (best[2] <= 15.0 or accept_first[0])) or _tt.time() > deadline:
                    break  # a natural plan found (headings come straightest first), or the time is up
                yaw = radial + np.radians(dyaw)
                tag = f"pitch {pitch:g} deg, heading {dyaw:+.0f} deg"
                hits = blocked(yaw, pitch)
                if hits:
                    print(f"  {tag}: plants in the way ({', '.join(hits)})")
                    continue
                a = np.array([np.cos(pt) * np.cos(yaw), np.cos(pt) * np.sin(yaw), -np.sin(pt)])  # approach
                across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # closing axis, horizontal (as on the first grasp: towards -y)
                R = np.column_stack([np.cross(across, a), across, a])
                a_h = np.array([a[0], a[1], 0.0]) / np.linalg.norm(a[:2])
                z_c, z_ff = z_close, finger_floor(yaw)
                if z_ff is not None and z_ff > z_close:
                    z_c = z_ff
                    if dyaw not in raised:
                        raised.add(dyaw)
                        print(f"  heading {dyaw:+.0f} deg: closing point raised {(z_c - z_close) * 1000:.0f} mm so the open fingers clear the "
                              f"neighbouring soil by {args.finger_clearance_mm:.0f} mm")
                close_pt = np.array([c[0], c[1], z_c]) + a_h * off[0] / 1000.0 + across * off[1] / 1000.0
                grasp = close_pt - a * fwd  # the open gap centre: the fingers swing forward onto close_pt
                pre = grasp - a * args.standoff_mm / 1000.0
                lift = grasp + [0, 0, args.lift_mm / 1000.0]
                # every joint solution of the grasp pose (kinematics.ik_all: joint 7 swept on every branch), clear at the
                # grasp itself; the most natural ones first (posture cost: how far the base is turned from the plant,
                # the elbow swung sideways, the wrist twisted). Close to the robot a flat hand has no natural solution
                # at all (joint 6 at its limit): then the least twisted one (2026-11-04: the first-found solution had
                # the base 70 deg away from the plant)
                T_gr, cands, why_g = make_T(R, grasp), [], []
                for q_g in kinematics.ik_all(T_gr @ inv_T(G), F, seeds=[q_now], limit_margin_rad=margin):
                    if np.linalg.norm(G_of(q_g)[:3, 3] - grasp) > 0.0015:
                        continue
                    cl_ = world.clearances(q_g, fingers=False, plants=False)
                    mg = min(v_[0] for v_ in cl_.values())
                    if mg < 0:
                        w_ = min(cl_.values(), key=lambda v_: v_[0])
                        if not why_g or w_[0] > why_g[0][0]:
                            why_g[:] = [w_]
                    if mg >= 0:
                        pc_g = cartesian.posture_cost(q_g, c[:2])
                        if pc_g <= max_posture:
                            cands.append((pc_g, q_g))
                        else:
                            over_limit[0] = min(over_limit[0], pc_g)
                cands.sort(key=lambda c_: c_[0])
                if not cands:
                    print(f"  {tag}: no joint solution clear at the grasp itself" + (f" ({why_g[0][1]}: {why_g[0][0] * 1000:+.0f} mm)" if why_g else ""))
                n_tried = 0
                for pc0, q_gr in cands:
                    if n_tried >= args.max_branches or (best is not None and pc0 >= best[0]) or _tt.time() > deadline:
                        break
                    n_tried += 1
                    lines = [f"  {tag}, posture {pc0:.0f} deg:"]
                    # the way in, as straight lines worked out backwards from the grasp: back to the pre-grasp
                    # (horizontal), up to the stage; executed the other way round (down, approach), then the lift
                    back, why = cartesian.chain(q_gr, [("pre-grasp", make_T(R, pre)), ("stage", make_T(R, pre + [0, 0, args.stage_up_mm / 1000.0]))], F, G, margin)
                    if back is None:
                        print(f"{lines[0][:-1]}: {why}")
                        continue
                    q_pre, q_st = back[0][1][-1], back[1][1][-1]
                    fwd_, why = cartesian.chain(q_st, [("down", make_T(R, pre)), ("approach", T_gr)], F, G, margin)
                    up_, why2 = cartesian.chain(q_gr, [("lift", make_T(R, lift))], F, G, margin)
                    if fwd_ is None or up_ is None:
                        print(f"{lines[0][:-1]}: {why or why2}")
                        continue
                    q_pre, q_gr2, q_li = fwd_[0][1][-1], fwd_[1][1][-1], up_[0][1][-1]
                    pc = max(cartesian.posture_cost(q_, c[:2]) for pcs_ in (fwd_, up_) for piece in pcs_ for q_ in piece[1])
                    if pc > max_posture:
                        over_limit[0] = min(over_limit[0], pc)
                        print(f"{lines[0][:-1]}: posture {pc:.0f} deg on the way in or out (limit {max_posture:.0f})")
                        continue
                    # each executed piece (joint waypoints of the straight lines) checked
                    ok = True
                    segs = []
                    for name_, piece in (("down", fwd_[0]), ("approach", fwd_[1]), ("lift", up_[0])):
                        qa_ = q_st if name_ == "down" else (q_pre if name_ == "approach" else q_gr2)
                        for i_, qb_ in enumerate(piece[2]):
                            nm = name_ if len(piece[2]) == 1 else f"{name_} {i_ + 1}/{len(piece[2])}"
                            ok &= report(nm, qa_, qb_, out=lines)
                            segs.append((name_, qb_))
                            qa_ = qb_
                        if not ok:
                            break
                    if not ok:
                        worst = next(ln for ln in lines[1:] if "NOT SAFE" in ln)
                        print(f"{lines[0][:-1]}: NOT SAFE at {worst.split(':')[0].strip()} {worst[worst.index('NOT SAFE') + 9:]}")
                        continue
                    # the turn from where the arm is to the stage: one joint move between two poses clear in the scene, the
                    # start optionally first straight up from where the arm is, the end optionally straight above the stage
                    # (in the grasp's own posture, reached from the stage by continuation), from where the hand goes
                    # straight down to the stage: high up the arm has the room to turn the hand flat
                    T0, Ts = G_of(q_now), G_of(q_st)
                    zmax = cfg["robot"]["workspace_m"]["z"][1] - 0.02
                    ups = [(0.0, q_now, [])]
                    for up_m in (0.08, 0.16, 0.25):
                        T1 = T0.copy()
                        T1[2, 3] = min(T0[2, 3] + up_m, zmax)
                        pc_, _ = cartesian.chain(q_now, [("lift", T1)], F, G, margin)
                        if pc_ is not None and scene.worst(q_now, pc_[0][1][-1])[0] >= min(0.0, min(scene.margins(q_now).values())) - 0.001:
                            ups.append((up_m, pc_[0][1][-1], pc_[0][2]))
                    highs = [(0.0, q_st, [])]
                    for hi_m in (0.10, 0.20, 0.30):
                        T2 = Ts.copy()
                        T2[2, 3] = min(Ts[2, 3] + hi_m, zmax)
                        pc_, _ = cartesian.chain(q_st, [("up", T2)], F, G, margin)
                        if pc_ is None:
                            continue
                        q_hi = pc_[0][1][-1]
                        dn, _ = cartesian.chain(q_hi, [("down", Ts)], F, G, margin)  # the way down from there, as executed
                        if dn is not None and scene.worst(q_hi, q_st)[0] >= 0:
                            highs.append((hi_m, q_hi, dn[0][2][:-1]))
                    turn = None
                    for (up_m, q_up, up_w), (hi_m, q_hi, dn_w) in sorted(((u, h) for u in ups for h in highs), key=lambda uh: uh[0][0] + uh[1][0]):
                        m_, part_ = scene.worst(q_up, q_hi)
                        if m_ >= 0:
                            turn = (list(up_w) + ([q_hi] if hi_m > 0 else []) + list(dn_w), m_, part_, up_m, hi_m, q_up, q_hi)
                            break
                    if turn is None and turn_budget[0] > 0:
                        # no single joint move: a short path search that keeps the turn sensible: the hand never lower than
                        # the lower of its two ends (minus 2 cm) and never more than 15 cm sideways off the straight line
                        # between them (2026-11-04: an unconstrained search swung the hand low beside the robot)
                        import time as _t

                        (up_m, q_up, up_w), (hi_m, q_hi, dn_w) = ups[-1], highs[-1]
                        ga, gb = G_of(q_up)[:3, 3], G_of(q_hi)[:3, 3]
                        zlo = min(ga[2], gb[2]) - 0.02

                        def sensible(q_, ga=ga, gb=gb, zlo=zlo):
                            g = G_of(q_)[:3, 3]
                            ab = gb - ga
                            t_ = float(np.clip((g - ga) @ ab / max(float(ab @ ab), 1e-9), 0, 1))
                            return g[2] >= zlo and np.linalg.norm((g - (ga + t_ * ab))[:2]) <= 0.15 and scene.free_ok(q_)

                        t0_ = _t.time()
                        path_ = plan_path(q_up, q_hi, sensible, time_s=min(3.0, turn_budget[0]))
                        turn_budget[0] -= _t.time() - t0_
                        if path_ is not None:
                            m_, part_ = min((scene.worst(a_, b_) for a_, b_ in zip([q_up] + path_[:-1], path_)), key=lambda x: x[0])
                            turn = (list(up_w) + list(path_[:-1]) + ([q_hi] if hi_m > 0 else []) + list(dn_w), m_, part_, up_m, hi_m, q_up, q_hi)
                    if turn is None:
                        m_, part_ = scene.worst(q_now, q_st)
                        print(f"{lines[0][:-1]}: NOT SAFE at the turn to the stage (also from up to 25 cm higher, to up to 30 cm above the "
                              f"stage: straight, smallest margin {m_ * 1000:+.0f} mm, {part_})")
                        continue
                    lines.append(f"  turn      : {'straight up ' + format(turn[3] * 100, '.0f') + ' cm, ' if turn[3] else ''}one joint move of at most "
                                 f"{np.degrees(np.abs(turn[6] - turn[5])).max():.0f} deg{' to ' + format(turn[4] * 100, '.0f') + ' cm above the stage, straight down' if turn[4] else ''} "
                                 f"to the stage; smallest margin {turn[1] * 1000:+.0f} mm ({turn[2]})")
                    cost = pc + 0.05 * float(np.degrees(np.abs(q_st - q_now)).max())
                    if best is None or cost < best[0]:
                        best = (cost, (pitch, dyaw, a, across, R, close_pt, q_st, q_pre, q_gr2, q_li, turn[0],
                                       {k_: [q_ for nm_, q_ in segs if nm_ == k_][:-1] for k_ in ("down", "approach", "lift")}, True, lines), pc)
                    print(f"{lines[0][:-1]}: ok (max posture {pc:.0f} deg)")
            if best is not None:
                chosen = best[1]
                break
            if _tt.time() > deadline:
                print(f"planning time used up (--time-budget-s {args.time_budget_s:.0f})")
                break
        if chosen is None:
            if np.isfinite(over_limit[0]):
                print(f"NO NATURAL FLAT GRASP: the least twisted safe-looking posture for {args.cell} is {over_limit[0]:.0f} deg "
                      f"(limit {max_posture:.0f}): too close to the robot or too low for a flat hand there (raise the tray or move "
                      f"it further out; see README 'where flat grasps work')")
            else:
                print("not reachable from any heading without going through other plants")
            return 1
        print("\n".join(chosen[-1]))
        pitch, dyaw, a, across, R, close_pt, q_st, q_pre, q_gr, q_li, q_via, q_down, ok, _ = chosen
        grasp = close_pt - a * fwd
        cam_above = (kinematics.fk_flange(q_gr) @ T_FC)[2, 3] > grasp[2]
        print(f"cell {args.cell}: closing point {np.round(close_pt, 4).tolist()} = cell centre + {off[0]:+.0f} mm along, {off[1]:+.0f} mm across "
              f"(closing axis {np.round(across, 2).tolist()}), {(close_pt[2] - soil_z) * 1000:.0f} mm above the soil; finger bottom ~"
              f"{(close_pt[2] - 0.0145 - soil_z) * 1000:.0f} mm above it")
        print(f"approach pitched {np.degrees(np.arcsin(-a[2])):.0f} deg down, heading {dyaw:+.0f} deg from straight out; open gripper stops "
              f"{fwd * 1000:.0f} mm behind the closing point; camera {'above' if cam_above else 'BELOW'} the gripper")
        plan_file.write_text(json.dumps({"cell": args.cell, "close_point": close_pt.tolist(), "offset_mm": list(off), "R": R.tolist(),
                                         "q": {"start": q_now.tolist(), "stage": q_st.tolist(), "pregrasp": q_pre.tolist(), "grasp": q_gr.tolist(), "lift": q_li.tolist(),
                                               },
                                         "vias": [v.tolist() for v in q_via],
                                         **{f"{k_}_vias": [v.tolist() for v in q_down[k_]] for k_ in ("down", "approach", "lift")},
                                         "z_close": round(float(z_close), 4), "tray_top": round(tray_top, 4),
                                         "yaw": float(radial + np.radians(dyaw)), **(obs_from or {}),
                                         "cell_xy": [float(v) for v in grid["cells"][args.cell][:2]],
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
        for i_, qv in enumerate(plan.get("vias", [])):  # the turn, through the via poses of the path search
            if not move(np.array(qv), args.dynamics, f"turn (via pose {i_ + 1}/{len(plan['vias'])})"):
                return 1
        if not move(q["stage"], args.dynamics, "turn above the pre-grasp"):
            return 1
        for i_, qv in enumerate(plan.get("down_vias", [])):  # the way down through via poses (same corridor)
            if not move(np.array(qv), args.dynamics, f"down (via pose {i_ + 1}/{len(plan['down_vias'])})"):
                return 1
        return 0 if move(q["pregrasp"], args.dynamics, "down to the pre-grasp") else 1
    if args.phase == "approach":
        if not at("pregrasp"):
            return 1
        for i_, qv in enumerate(plan.get("approach_vias", [])):  # the straight approach as joint waypoints
            if not move(np.array(qv), fine, f"approach ({i_ + 1}/{len(plan['approach_vias']) + 1})"):
                return 1
        return 0 if move(q["grasp"], fine, "approach") else 1
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
        for i_, qv in enumerate(plan.get("lift_vias", [])):  # the straight lift as joint waypoints
            if not move(np.array(qv), fine, f"lift ({i_ + 1}/{len(plan['lift_vias']) + 1})"):
                return 1
        return 0 if move(q["lift"], fine, "lift") else 1
    if args.phase == "lower":
        if not at("lift"):
            return 1
        return 0 if move(q["grasp"], fine, "put back down") else 1
    # retreat
    if not any(np.degrees(np.abs(q_now - q[k])).max() <= 2.0 for k in ("lift", "grasp")):
        print("refused: neither at the lift nor at the grasp pose")
        return 1
    return 0 if move(q["pregrasp"], fine, "back to the pre-grasp") else 1


if __name__ == "__main__":
    raise SystemExit(main())
