#!/usr/bin/env python
"""Side grasp of one plant: the motion phases of a grasp the program planned (scripts/plan.py grasp writes
experiments/grasp/<cell>_side_plan.json), and the close-up before it. The fingers swing forward when closing
(gripper.pin_brake.close_forward_m), so the open gripper stops that far behind the stem. One phase per call, so that
every motion is confirmed. (2026-11-16: this script's own per-plant planner, the 'plan' phase, was removed at the
user's request: one planner only, the program's.)

    $P scripts/pick.py --cell C7 observe     # look at the plug from ~0.34 m: its own soil surface height
    $P scripts/plan.py grasp --cell C7        # the program's plan (after plan.py build), saved; no motion
    $P scripts/pick.py --cell C7 pregrasp    # open the gripper, go to 6 cm behind the grasp pose
    $P scripts/pick.py --cell C7 approach    # slide forward to the grasp pose (slow)
    $P scripts/pick.py --cell C7 nudge --dx-mm 4 --dy-mm -2    # fine shift (x forward, y towards higher columns)
    $P scripts/gripper.py close                     # close (separate script)
    $P scripts/pick.py --cell C7 lift        # lift straight up (slow)
    $P scripts/pick.py --cell C7 check       # look at the cell again: is the soil gone (plug pulled out)?
    $P scripts/pick.py --cell C7 lower       # put it back down (to the grasp pose, slow)
    $P scripts/gripper.py open
    $P scripts/pick.py --cell C7 retreat     # back to the pre-grasp
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("phase", choices=["observe", "pregrasp", "approach", "nudge", "lift", "check", "lower", "retreat"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--observe-distance", type=float, default=0.34)
    ap.add_argument("--plug-dz-mm", type=float, default=13.0, help="fallback only: plug top above the tray top when no soil was measured")
    ap.add_argument("--lift-mm", type=float, default=20.0)
    ap.add_argument("--dx-mm", type=float, default=0.0, help="nudge: horizontal, along the approach")
    ap.add_argument("--dy-mm", type=float, default=0.0, help="nudge: along the closing axis")
    ap.add_argument("--dz-mm", type=float, default=0.0, help="nudge: up")
    ap.add_argument("--max-line-dev-mm", type=float, default=8.0, help="the moves at the plant stay this close to their straight line")
    ap.add_argument("--dynamics", type=float, default=None, help="one factor for all moves (default: robot.dynamics / fine_dynamics in the config)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    fine = "fine" if args.dynamics is None else min(args.dynamics, 0.03)
    # the wrist housings are ~6 cm in radius around the checked joint centres: over the tray the user's own setting
    # (tested by hand), over the disc and the table the full clearance

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T, make_T

    T_FC, _ = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
    tray_top = float(grid["z_top"])
    disc_file = PROJECT_DIR / "calibration" / "disc.json"
    # wrist and camera clearances are over what lies under them (tray top, disc top or table), as in place.py
    out = PROJECT_DIR / "experiments" / "grasp"
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"{args.cell}_side_plan.json"
    from seedling.world import World

    from seedling.disc import load_disc

    disc_now = load_disc(disc_file)  # None when the last picture saw it gone
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
        from seedling.grasp import observe_pose as _op

        return _op(cfg, grid, args.cell, T_FC, tray_top, q_from, elevations=(65, 55, 70, 75, 80),
                   distance=args.observe_distance, plug_dz=args.plug_dz_mm / 1000)

    def look(tag):
        """Observation pose for this cell (camera ~0.34 m away, looking down), picture, soil surface of the cell."""
        import time

        from seedling.calibration import flange_pose
        from seedling.camera import ZedCamera
        from seedling.plugsense import measure_plug

        c = np.array(grid["cells"][args.cell], float)
        pl = None
        # the observation pose is always worked out as from the overview pose (as when the plant was chosen: run.py, pick.py plan
        # --from-observe), whatever the arm comes from (run.py goes from one plant's hole straight to the next plant,
        # 2026-11-05); the way there from where the arm is is planned below
        q_ref = np.array(cfg["tray"]["overview_q"], float)
        if plan_file.exists():
            try:
                old_ = json.loads(plan_file.read_text())
                # (the program, plan.py, picks it near the cell's stage: any observation pose made for this cell is used)
                if ("observe_q" in old_ and (old_.get("program") or float(np.degrees(np.abs(np.array(old_["observe_from"]) - q_ref)).max()) < 0.5)
                        and np.linalg.norm(np.array(old_["observe_cell_xyz"]) - np.array(grid["cells"][args.cell], float)) < 0.003):
                    from seedling.wrist import ObservationPlan, path_clearance

                    q_o = np.array(old_["observe_q"])
                    z_, part_ = path_clearance(q_ref, q_o, T_FC, cfg)
                    pl = ObservationPlan(True, q_o, None, old_["observe_desc"] + " (worked out when the plant was chosen)",
                                         float(np.degrees(np.abs(q_o - q_ref)).max()), z_, part_)
            except Exception:  # noqa: BLE001
                pl = None
        if pl is None:
            pl = observe_pose(q_ref)
        if not pl.ok:
            print("no safe observation pose:", pl.tried[-3:])
            return None
        print(f"{tag}: observation pose {pl.desc}, max joint change {pl.max_joint_step_deg:.0f} deg, lowest {pl.lowest_part} z={pl.lowest_z:.3f}")
        # from where the arm is (the overview, or just after letting go the last plant in the disc) to the observation
        # pose: the straight joint move when it is clear of everything (seedling.world: the plants in the tray and in
        # the disc too), else straight up first and a path (seedling.planner.lift_then_plan, as goto_overview.py does)
        from seedling.planner import lift_then_plan
        from seedling.world import World

        disc_o = json.loads(disc_file.read_text()) if disc_file.exists() else None
        from seedling.disc import taken_holes

        taken_o = taken_holes(disc_o) if disc_o is not None and "holes" in disc_o else None
        w_o = World(cfg, T_FC, grid, disc_o, tray_plants=list(grid.get("planted") or grid.get("occupied", [])), taken_holes=taken_o)
        steps_o, msg_o = lift_then_plan(robot.q(), pl.q, w_o, cfg, time_s=6.0)
        if steps_o is None:
            print(f"no safe way to the observation pose ({msg_o})")
            return None
        if len(steps_o) > 1:
            print(f"{tag}: to the observation pose {msg_o}, {len(steps_o)} moves")
        for name_o, _qa, qb_o in steps_o:
            r = robot.move_joints(qb_o, dynamics=args.dynamics, description=f"observe {args.cell}: {name_o}")
            if not r.ok:
                print("motion failed:", r.reason)
                return None
        import os

        if os.environ.get("SEEDLING_SIM_SOIL_DZ"):
            # the whole-run simulation (scripts/80): the close-up "measures" the soil this much off the overview's value
            # (per cell, uniformly in [lo, hi] from the seed): no camera
            import zlib

            if os.environ.get("SEEDLING_SIM_REALTIME"):
                time.sleep(0.5 + 0.35)  # settling, then 5 frames averaged (real time simulated)
            lo_, hi_, seed_ = os.environ["SEEDLING_SIM_SOIL_DZ"].split(",")
            u_ = (zlib.crc32(f"{args.cell}:{seed_}".encode()) % 10000) / 10000.0
            z_ov_ = (grid.get("soil_z_overview") or {}).get(args.cell) or tray_top + 0.013
            z_m = float(z_ov_) + float(lo_) + u_ * (float(hi_) - float(lo_))
            print(f"{tag}: (simulated) soil z {z_m:.3f} ({(z_m - tray_top) * 1000:+.0f} mm vs the tray top)")
            return {"q": robot.q().tolist(), "measure": {"ok": True, "soil_z": z_m, "n": 0, "shift_mm": [0.0, 0.0]},
                    "stem": None, "leaf_cover": 0.0, "overview": z_ov_,
                    "overview_cell": [float(v) for v in grid["cells"][args.cell]]}  # for this tray fit (grasp.load_observation)
        time.sleep(0.5)
        cam = ZedCamera(cfg)
        try:
            f = cam.grab(n_average=5)
        finally:
            cam.close()
        T_BC = flange_pose(robot) @ T_FC
        f.meta["q"] = robot.q().tolist()
        f.save(out / f"{args.cell}_{tag}", save_png=False, compress=False)  # read again by plan / plan.py at once; the archive copy below is compressed
        # every close-up kept (the file above is overwritten per cell): training data for a stem fine-tune model
        (out / "closeups").mkdir(exist_ok=True)
        from seedling.camera import save_in_background

        save_in_background([(f, out / "closeups" / f"{args.cell}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}", False)])
        # the cell again from the close-up (the walls of the cells around it, seedling/closeup.py): recorded always,
        # grasped at only with grasp.closeup_xy: apply (then the plug is measured around that centre too)
        from seedling.closeup import grasp_xy, refine_cell

        try:
            cu = refine_cell(f, T_BC, grid["cells"], args.cell, cfg)
        except Exception as e:  # noqa: BLE001  (a record, never a reason to stop)
            cu = {"ok": False, "status": f"error: {e!r}", "shift_mm": [0.0, 0.0]}
        print(f"{tag}: close-up re-location: {cu['status']}; shift {cu.get('shift_mm')} mm"
              + (f", tray top {cu['z_top']:.3f} (overview {cu['z_overview']:.3f})" if cu.get("z_top") else ""))
        xy_, why_ = grasp_xy({"closeup_xy": cu}, cfg)
        if xy_ is not None:
            c = np.array([xy_[0], xy_[1], c[2]])
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
        return {"q": robot.q().tolist(), "measure": m, "stem": stem, "leaf_cover": cover, "overview": (grid.get("soil_z_overview") or {}).get(args.cell),
                "closeup_xy": cu, "T_BC": T_BC.tolist(), "overview_cell": [float(v) for v in grid["cells"][args.cell]]}

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

    if not plan_file.exists():
        print("no plan: plan it first (scripts/plan.py grasp --cell ...; the program's plan, 2026-11-16: the old per-plant planner removed)")
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
        import os

        from seedling.gripper import make_gripper

        if not os.environ.get("SEEDLING_SIM_STATE"):  # (the whole-run simulation has no gripper)
            g = make_gripper(cfg, sim=False, step_mode=False)
            g.open()
            g.detach(settle_s=0.0)  # (no wait here, as before)
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
