#!/usr/bin/env python
"""The side tool (robot.side_tool, 2026-11-16) measured on the robot. READ ONLY: no motion is ever commanded.

pivot (default): the open fingers' centre put on ONE fixed sharp point (a screw tip on the table) by hand, from several
hand headings (the user guides the arm, Desk's guiding mode / the pilot buttons); Enter reads the joints each time.
The point's position in the flange frame then follows (least squares, all the poses meeting at one point): the arm's
direction in the flange plane (arm_direction_deg), how far out (reach_m) and how far below the face (below_m; only seen
when the hand is also tilted between poses, else the configured value is kept). 1 deg of direction is 4.5 mm sideways
at 259 mm: measured, not judged by eye.

weight: the arm still, nothing touching the tool: the external force the robot estimates (its model has Desk's
end-effector mass) gives how much heavier the tool is and, flange down, where its weight sits sideways; printed with the
values for Desk's end-effector settings.

    .venvs/pipeline/bin/python scripts/tool_pivot.py              # pivot, 5+ headings, Enter each, q to finish
    .venvs/pipeline/bin/python scripts/tool_pivot.py --weight
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import time

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR

G_ACC = 9.81


def solve_pivot(T_BF_list, below_m: float | None = None):
    """(p_F: the point in the flange frame, c_B: the point in the base frame, residuals m, z observed). All poses put
    p_F on c_B: R_i p + t_i = c. Rotations about one axis only (the flange kept vertical) leave p along that axis
    unknown: then p_z = below_m."""
    A = np.vstack([np.hstack([T[:3, :3], -np.eye(3)]) for T in T_BF_list])
    b = np.concatenate([-T[:3, 3] for T in T_BF_list])
    s = np.linalg.svd(A, compute_uv=False)
    z_seen = s[-1] > 0.05 * s[0]
    if z_seen or below_m is None:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        A2 = np.delete(A, 2, axis=1)
        x2 = np.linalg.lstsq(A2, b - A[:, 2] * below_m, rcond=None)[0]
        x = np.insert(x2, 2, below_m)
    p, c = x[:3], x[3:]
    res = [float(np.linalg.norm(T[:3, :3] @ p + T[:3, 3] - c)) for T in T_BF_list]
    return p, c, res, bool(z_seen)


def weight_from_readings(readings, m_ee: float, c_ee, arm_direction_deg: float) -> dict:
    """the tool's mass and centre of mass (flange frame) from readings with the flange down and the arm turned between
    them (each: F, tau base frame, arm_dir_B): (mass x lever) along / across the arm and the estimate's offset (base
    frame, the same in all) by least squares; the mass from the mean Fz. Sign: the wrench the robot exerts"""
    A, y, fz = [], [], []
    for r_ in readings:
        ub = np.asarray(r_["arm_dir_B"], float)
        t_ = np.asarray(r_["tau"], float)
        A.append(np.column_stack([ub, [-ub[1], ub[0]], np.eye(2)]))
        y.append(np.array([-t_[1], t_[0]]) / G_ACC)  # the unmodelled (mass x lever), base frame xy
        fz.append(float(r_["F"][2]))
    x, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)
    dm = float(np.mean(fz)) / G_ACC
    m_tot = m_ee + dm
    th_ = np.radians(arm_direction_deg)
    u_F = np.array([np.cos(th_), np.sin(th_)])
    mc_h = m_ee * np.asarray(c_ee, float)[:2] + x[0] * u_F + x[1] * np.array([-u_F[1], u_F[0]])
    c_tot = np.array([*(mc_h / m_tot), 0.02])  # height along the flange axis not seen (flange down): ~2 cm out
    return {"m_total": float(m_tot), "c_total": c_tot.tolist(), "x": x.tolist(), "dm": dm, "fz": fz}


def read_q(robot, n: int = 10) -> np.ndarray:
    qs = []
    for _ in range(n):
        qs.append(robot.q())
        time.sleep(0.02)
    return np.median(qs, axis=0)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--weight", action="store_true", help="the tool's extra weight and where it sits (arm still)")
    ap.add_argument("--min-poses", type=int, default=4)
    ap.add_argument("--add", action="store_true", help="read one pose into the session (.tmp/pivot_session.json)")
    ap.add_argument("--undo", action="store_true", help="drop the session's last pose")
    ap.add_argument("--solve", action="store_true", help="conclude from the session's poses and save")
    ap.add_argument("--reset", action="store_true", help="start a new session (with --add: this pose the first)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.robot import FrankyRobot
    from seedling.transforms import inv_T

    load_T_FC(cfg)  # the joint offsets estimated with the arm (kinematics.JOINT_OFFSETS), as every plan uses them
    robot = FrankyRobot(cfg, step_mode=True)
    st = cfg["robot"].get("side_tool") or {}
    out_dir = PROJECT_DIR / "experiments" / "tool"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.weight:
        s = robot.robot.state
        print("hold still: averaging the robot's external force estimate for 3 s (nothing may touch the tool)")
        W = []
        t_end = time.time() + 3.0
        while time.time() < t_end:
            W.append(np.array(robot.robot.state.O_F_ext_hat_K, float))
            time.sleep(0.02)
        W = np.median(W, axis=0)
        F, tau = W[:3], W[3:]
        q = robot.q()
        T_BF = kinematics.fk_flange(q)
        m_ee = float(s.m_ee)
        c_ee = np.array(s.F_x_Cee, float)
        print(f"Desk's end effector now: mass {m_ee:.3f} kg, centre of mass in the flange frame {np.round(c_ee, 4).tolist()} m")
        print(f"external force (base frame) {np.round(F, 2).tolist()} N, torque {np.round(tau, 3).tolist()} Nm")
        tilt = np.degrees(np.arccos(min(1.0, abs(T_BF[2, 2]))))
        T_BE = kinematics.fk(q, robot.F_T_EE)
        d = (T_BE @ robot.T_EG)[:3, 3] - T_BE[:3, 3]
        u = d[:2] / max(1e-9, np.linalg.norm(d[:2]))  # the arm's direction (base frame, level)
        rec = {"q": q.tolist(), "F": F.tolist(), "tau": tau.tolist(), "m_ee": m_ee, "c_ee": c_ee.tolist(), "arm_dir_B": u.tolist(),
               "tilt_deg": float(tilt), "t": time.time()}
        (out_dir / f"weight_{stamp}.json").write_text(json.dumps(rec, indent=1))
        if tilt > 15:
            print(f"the flange is tilted {tilt:.0f} deg from vertical: read with the flange pointing down")
            return 1
        # Sign: taken as the wrench the robot exerts (a tool heavier than Desk's model reads +z). As the libfranka docs
        # word it (acting on the robot) the side tool came out lighter than the Franka Hand with its weight behind the
        # flange; this way 0.9 kg, 13 cm out along the arm (2026-11-16, two arm headings 138 deg apart)
        # The estimate is offset by ~2 N / ~2 Nm at rest (one reading alone said 1.1 m out): readings with the arm turned
        # about the flange axis between them are combined, the offset (the same in the base frame) solved for
        prev = []
        for f_ in sorted(out_dir.glob("weight_*.json")):
            r_ = json.loads(f_.read_text())
            if "arm_dir_B" in r_ and abs(r_.get("m_ee", -1) - m_ee) < 1e-6 and time.time() - r_.get("t", 0) < 3 * 3600 \
                    and r_.get("tilt_deg", 90) <= 15:
                prev.append(r_)
        angles = [np.degrees(np.arccos(np.clip(np.dot(r_["arm_dir_B"], u), -1, 1))) for r_ in prev]
        if len(prev) < 2 or max(angles) < 90:
            print(f"reading saved; turn the arm about the flange axis by 90 deg or more (joint 7, flange still down) and read again "
                  f"({len(prev)} reading(s) usable now, the widest {max(angles):.0f} deg apart)")
            return 0
        w = weight_from_readings(prev, m_ee, c_ee, float(st.get("arm_direction_deg", 0.0)))
        m_tot, c_tot, x, dm, fz = w["m_total"], np.array(w["c_total"]), w["x"], w["dm"], w["fz"]
        rec.update(m_total=m_tot, c_total=c_tot.tolist(), moment_along_arm=float(x[0]), moment_across=float(x[1]))
        (out_dir / f"weight_{stamp}.json").write_text(json.dumps(rec, indent=1))
        print(f"{len(prev)} readings, arm headings up to {max(angles):.0f} deg apart: the tool {dm:+.2f} kg against Desk's model "
              f"(Fz {np.round(fz, 2).tolist()} N), its weight moment {x[0]:+.3f} kg m along the arm, {x[1]:+.3f} across "
              f"(estimate offset {np.round(x[2:] * G_ACC, 2).tolist()} Nm)")
        print(f"for Desk (end effector): mass {m_tot:.2f} kg, centre of mass (flange frame) {np.round(c_tot, 3).tolist()} m, "
              f"{np.linalg.norm(c_tot[:2]) * 100:.0f} cm from the flange axis (+-0.3 kg)")
        return 0

    if args.add or args.undo or args.solve or args.reset:
        # one step per call (the poses kept in .tmp/pivot_session.json), for a session driven from outside a terminal
        sf = PROJECT_DIR / ".tmp" / "pivot_session.json"
        qs = [] if args.reset or not sf.exists() else json.loads(sf.read_text())["q"]
        if args.undo and qs:
            qs.pop()
        if args.add:
            q = read_q(robot)
            qs.append(q.tolist())
            T = kinematics.fk_flange(q)
            print(f"pose {len(qs)}: flange at {np.round(T[:3, 3], 3).tolist()}, turned "
                  f"{np.degrees(np.arctan2(T[1, 0], T[0, 0])):+.0f} deg, tilted {np.degrees(np.arccos(min(1.0, abs(T[2, 2])))):.0f} deg")
        sf.parent.mkdir(exist_ok=True)
        sf.write_text(json.dumps({"q": qs}))
        T_list = [kinematics.fk_flange(np.array(q_)) for q_ in qs]
        if len(T_list) >= 2:
            p, c, res, z_seen = solve_pivot(T_list, float(st.get("below_m", 0.048)))
            print(f"{len(T_list)} poses: point in the flange frame {np.round(p * 1000, 1).tolist()} mm "
                  f"(direction {np.degrees(np.arctan2(p[1], p[0])):.2f} deg, reach {np.hypot(p[0], p[1]) * 1000:.1f} mm, "
                  f"below {p[2] * 1000:.1f} mm{'' if z_seen else ' (configured)'}); misfit per pose "
                  f"{np.round(np.array(res) * 1000, 1).tolist()} mm")
        else:
            print(f"{len(T_list)} pose(s)")
        if not args.solve:
            return 0
        if len(T_list) < args.min_poses:
            print(f"only {len(T_list)} poses: nothing concluded")
            return 1
    else:
        T_list, qs = interactive(robot, kinematics, args, st)
    if len(T_list) < args.min_poses:
        print(f"only {len(T_list)} poses: nothing concluded")
        return 1
    p, c, res, z_seen = solve_pivot(T_list, float(st.get("below_m", 0.048)))
    th = float(np.degrees(np.arctan2(p[1], p[0])))
    reach = float(np.hypot(p[0], p[1]))
    print(f"\nthe open fingers' centre in the flange frame: {np.round(p * 1000, 1).tolist()} mm "
          f"(misfit per pose {np.round(np.array(res) * 1000, 1).tolist()} mm, rms {np.sqrt(np.mean(np.square(res))) * 1000:.1f} mm)")
    print(f"  arm_direction_deg: {th:.2f}   reach_m: {reach:.4f}   below_m: {p[2]:.4f}"
          + ("" if z_seen else " (not seen: the poses only turned about the flange axis; the configured value kept)"))
    print(f"  (configured: {st.get('arm_direction_deg')} / {st.get('reach_m')} / {st.get('below_m')})")
    rec = {"q": qs, "p_flange_m": p.tolist(), "point_base_m": c.tolist(), "residual_m": res, "below_seen": z_seen,
           "arm_direction_deg": th, "reach_m": reach, "below_m": float(p[2])}
    f = out_dir / f"pivot_{stamp}.json"
    f.write_text(json.dumps(rec, indent=1))
    print(f"saved {f.relative_to(PROJECT_DIR)}; put the three values into configs/pipeline.yaml robot.side_tool")
    return 0


def interactive(robot, kinematics, args, st):
    print("Desk: guiding mode on. Put the OPEN fingers' centre on one fixed sharp point; hold still; Enter.\n"
          f"Then turn the hand to another heading (45-90 deg further; tilt it a little now and then) and again; "
          f"{args.min_poses}+ poses. 'u' undoes the last, 'q' finishes.")
    T_list, qs = [], []
    while True:
        cmd = input(f"[{len(T_list)} poses] Enter = read, u = undo, q = finish: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "u":
            if T_list:
                T_list.pop()
                qs.pop()
            continue
        q = read_q(robot)
        T = kinematics.fk_flange(q)
        T_list.append(T)
        qs.append(q.tolist())
        yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        tilt = np.degrees(np.arccos(min(1.0, abs(T[2, 2]))))
        print(f"  pose {len(T_list)}: flange at {np.round(T[:3, 3], 3).tolist()}, turned {yaw:+.0f} deg, tilted {tilt:.0f} deg")
        if len(T_list) >= 2:
            p, c, res, z_seen = solve_pivot(T_list, float(st.get("below_m", 0.048)))
            print(f"  so far: point in the flange frame {np.round(p * 1000, 1).tolist()} mm; misfit per pose "
                  f"{np.round(np.array(res) * 1000, 1).tolist()} mm")
    return T_list, qs



if __name__ == "__main__":
    raise SystemExit(main())
