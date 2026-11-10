#!/usr/bin/env python
"""WARNING (2026-11-06): never leave the pin-brake gripper closed while touching points (the motor nearly burned out).

Correct the wrist camera's mounting (T_flange_cam) from hand-guided touch points: the closed fingers' meeting
point (the pointer) is guided by hand exactly over a known spot the overview located (the disc centre, a hole, a tray
cell), its position read from the joints, and the mounting is fitted so that the overview puts those spots where the
pointer really was.

    .venvs/pipeline/bin/python scripts/81_point_check.py A1 --height-mm 30   # the pointer over cell A1 (then guide it by hand)
    .venvs/pipeline/bin/python scripts/82_touch_calib.py record A1           # read the joints: one touch point
    .venvs/pipeline/bin/python scripts/82_touch_calib.py list
    .venvs/pipeline/bin/python scripts/82_touch_calib.py solve               # fit, print the residuals (writes nothing)
    .venvs/pipeline/bin/python scripts/82_touch_calib.py solve --write       # a new mounting file + config points to it

The located spots are taken from the overview record (experiments/tray/<id>/) the touch belongs to: the last locate
(calibration/tray_grid.json, calibration/disc.json) when recorded. The pointer is EMPTY_CLOSE_FWD ahead of the open gap
centre (closed on nothing the fingers turn fully: 27 mm, configs gripper.pin_brake.close_forward_m notes).
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import time

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR

EMPTY_CLOSE_FWD = 0.029  # measured 2026-11-06: closed meeting point 179 mm, open gap centre 150 mm from the flange face
FILE = PROJECT_DIR / "calibration" / "touch_points.jsonl"


def located(name: str) -> tuple[list, str]:
    """the overview's base-frame point for a spot name, and which file it came from"""
    grid = json.loads((PROJECT_DIR / "calibration" / "tray_grid.json").read_text())
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    n = name.upper()
    if n in ("CENTRE", "CENTER"):
        return [float(disc["centre"][0]), float(disc["centre"][1]), float(disc["z_top"])], "disc"
    if n.startswith("H") and n[1:].isdigit():
        h = disc["holes"][int(n[1:]) - 1]
        return [float(h[0]), float(h[1]), float(disc["z_top"])], "disc"
    c = grid["cells"][n]
    return [float(c[0]), float(c[1]), float(grid["z_top"])], "tray"


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("action", choices=["record", "list", "solve"])
    ap.add_argument("name", nargs="?", help="record: the spot (centre, H1..H6, a cell)")
    ap.add_argument("--q", default=None, help="record: these joints (JSON) instead of reading the robot")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.config import ee_to_contact, flange_to_ee

    T_FC, meta = load_T_FC(cfg)
    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    grid = json.loads((PROJECT_DIR / "calibration" / "tray_grid.json").read_text())
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())

    if args.action == "record":
        if not args.name:
            print("record NAME")
            return 1
        if args.q:
            q = np.array(json.loads(args.q), float)
        else:
            from seedling.robot import FrankyRobot

            q = FrankyRobot(cfg, step_mode=True).q()
        T = kinematics.fk(q, F) @ G
        a = T[:3, 2]
        pointer = T[:3, 3] + a * EMPTY_CLOSE_FWD
        p_loc, src = located(args.name)
        q_ov = (grid if src == "tray" else disc).get("q") or grid.get("q")
        rec = {"name": args.name.upper(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "q": q.tolist(),
               "pointer": pointer.tolist(), "located": p_loc, "from": src, "q_overview": q_ov,
               "calibration_id": meta.get("calibration_id"), "tilt_deg": float(np.degrees(np.arcsin(abs(a[2]))))}
        with FILE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        d = (pointer[:2] - np.array(p_loc[:2])) * 1000
        print(f"{rec['name']}: pointer {np.round(pointer, 4).tolist()}, located {np.round(p_loc, 4).tolist()}: "
              f"pointer - located dx {d[0]:+.1f} dy {d[1]:+.1f} mm ({np.hypot(*d):.1f}); hand {rec['tilt_deg']:.1f} deg from flat")
        return 0

    recs = [json.loads(line) for line in FILE.read_text().splitlines() if line.strip()] if FILE.exists() else []
    if args.action == "list":
        for r in recs:
            d = (np.array(r["pointer"][:2]) - np.array(r["located"][:2])) * 1000
            print(f"{r['time']} {r['name']:>6}: dx {d[0]:+6.1f} dy {d[1]:+6.1f} mm  (located {np.round(r['located'], 4).tolist()})")
        return 0

    # solve: T_FC' = T_FC @ exp(delta); each located spot back into the camera frame of its overview (as the camera saw
    # it), then out with the corrected mounting; its xy must land on the pointer (the pointer's height is not the spot's)
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from seedling.transforms import apply_T, inv_T, make_T

    if len(recs) < 3:
        print(f"{len(recs)} touch points: at least 3 (spread over the tray and the disc)")
        return 1
    rows = []
    for r in recs:
        T_BF = kinematics.fk_flange(np.array(r["q_overview"], float))
        p_cam = apply_T(inv_T(T_BF @ T_FC), np.array(r["located"], float))
        Tq = kinematics.fk(np.array(r["q"], float), F) @ G
        rows.append((T_BF, p_cam, np.array(r["pointer"][:2], float), r["name"], Tq[:3, 2][:2].copy()))

    def corrected(x):
        return T_FC @ make_T(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])

    def resid(x):
        Tn = corrected(x)
        res = [(apply_T(T_BF @ Tn, p)[:2] - (ptr + ax * x[6])) for T_BF, p, ptr, _, ax in rows]
        # a weak pull toward no change: only what the points ask for
        return np.r_[np.concatenate(res), x * 0.2]

    before = np.array([np.linalg.norm(apply_T(T_BF @ T_FC, p)[:2] - ptr) for T_BF, p, ptr, _, _ in rows]) * 1000
    sol = least_squares(resid, np.zeros(7))
    Tn = corrected(sol.x)
    after = np.array([np.linalg.norm(apply_T(T_BF @ Tn, p)[:2] - (ptr + ax * sol.x[6])) for T_BF, p, ptr, _, ax in rows]) * 1000
    # leave-one-out: each point predicted by the fit of the others
    loo = []
    for i in range(len(rows)):
        keep = [rw for j, rw in enumerate(rows) if j != i]

        def rr(x, keep=keep):
            Tk = corrected(x)
            return np.r_[np.concatenate([apply_T(T_BF @ Tk, p)[:2] - (ptr + ax * x[6]) for T_BF, p, ptr, _, ax in keep]), x * 0.2]
        if len(keep) >= 2:
            xk = least_squares(rr, np.zeros(7)).x
            T_BF, p, ptr, _, ax = rows[i]
            loo.append(float(np.linalg.norm(apply_T(T_BF @ corrected(xk), p)[:2] - (ptr + ax * xk[6])) * 1000))
    print(f"{len(rows)} touch points; mounting change: rotation {np.degrees(np.linalg.norm(sol.x[:3])):.2f} deg, "
          f"shift {np.linalg.norm(sol.x[3:6]) * 1000:.1f} mm; the pointer {(EMPTY_CLOSE_FWD + sol.x[6]) * 1000:.1f} mm ahead of the gap centre")
    for (_, _, _, n, _), b, a_, l in zip(rows, before, after, loo + [np.nan] * (len(rows) - len(loo))):
        print(f"  {n:>6}: {b:5.1f} mm before -> {a_:4.1f} mm after (left out of the fit: {l:4.1f} mm)")
    if not args.write:
        return 0
    import yaml

    src = PROJECT_DIR / cfg["camera"]["T_flange_cam_file"]
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    data["T_flange_cam"] = [[float(v) for v in row] for row in Tn]
    data["calibration_id"] = f"{data.get('calibration_id', '')}+touch_{stamp}"
    data["touch_fix"] = {"n_points": len(rows), "before_mm": [round(float(v), 1) for v in before],
                         "after_mm": [round(float(v), 1) for v in after], "leave_one_out_mm": [round(v, 1) for v in loo],
                         "previous_file": str(src.relative_to(PROJECT_DIR))}
    dst = PROJECT_DIR / "calibration" / f"T_flange_cam.touch_{stamp}.yaml"
    dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    cp = PROJECT_DIR / "configs" / "pipeline.yaml"
    txt = cp.read_text(encoding="utf-8")
    old = f"T_flange_cam_file: {cfg['camera']['T_flange_cam_file']}"
    if old not in txt:
        print("config line not found; set camera.T_flange_cam_file to", dst.relative_to(PROJECT_DIR))
        return 1
    cp.write_text(txt.replace(old, f"T_flange_cam_file: {dst.relative_to(PROJECT_DIR)}", 1), encoding="utf-8")
    print("written", dst.relative_to(PROJECT_DIR), "and the config points to it (the previous file kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
