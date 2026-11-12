#!/usr/bin/env python
"""Replay the close-up re-location (seedling/closeup.py, what pick.py observe runs) on the recorded close-ups, offline.
Each experiments/grasp/closeups/<cell>_observe_<time>.npz is paired with the overview fit used then (the last
located experiments/tray/<id> before it, that day) and that fit's camera calibration (the close-up's pose from its
own joint angles); one line per close-up, a summary, and optionally overlays (the overview's cell walls red, the
close-up's green, drawn on the close-up at the tray top it measured).

    .venvs/pipeline/bin/python tools/replay_closeup.py                     # all close-ups -> experiments/replay/closeup_<time>/
    .venvs/pipeline/bin/python tools/replay_closeup.py --only 20261106 C9 --overlays
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[], help="close-up name prefixes (a cell, e.g. C9) or days / times (20261106)")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    import numpy as np

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.camera import Frame
    from seedling.closeup import refine_cell, tray_axes
    from seedling.config import load_config
    from seedling.transforms import apply_T, inv_T

    cfg = load_config()
    dq = np.asarray(load_T_FC(cfg)[1].get("joint_offsets_rad") or np.zeros(7), float)
    out = Path(args.out) if args.out else ROOT / "experiments" / "replay" / f"closeup_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    ovs = []
    for d in sorted((ROOT / "experiments" / "tray").iterdir()):
        j = d / "tray_grid.json"
        if j.exists():
            s = json.loads(j.read_text())
            if s.get("ok"):
                ovs.append((d.name, s))
    rows = []
    for f in sorted((ROOT / "experiments" / "grasp" / "closeups").glob("*.npz")):
        cell, t = f.stem.split("_observe_")
        if args.only and not any(f.stem.startswith(o) or t.startswith(o) for o in args.only):
            continue
        c_ = [o for o in ovs if o[0] < t and o[0][:8] == t[:8]]
        if not c_:
            print(f"{f.stem}: no located overview before it that day")
            continue
        oid, s = c_[-1]
        fr = Frame.load(f)
        T_FC = inv_T(kinematics.fk_flange(np.array(s["q"]), offsets=dq)) @ np.array(s["T_BC"])
        T_BC = kinematics.fk_flange(np.array(fr.meta["q"]), offsets=dq) @ T_FC
        t0 = time.perf_counter()
        r = refine_cell(fr, T_BC, s["cells"], cell, cfg)
        r.update(id=f.stem, overview=oid, cell=cell, calibration=s.get("calibration"), t_s=round(time.perf_counter() - t0, 2))
        rows.append(r)
        print(f"{f.stem:32s} {'ok ' if r['ok'] else 'NO '} shift {str(r['shift_mm']):16s} tray top {r.get('z_top')} "
              f"(overview {r.get('z_overview')})  yaw {r.get('yaw_deg')}  seen {r.get('seen')}  [{r['status'][:60]}]", flush=True)
        if args.overlays and r.get("z_top"):
            img = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR).copy()
            er, ec, p = tray_axes(s["cells"])
            zt, K = r["z_top"], fr.intr.K()

            def lines(cc, colr, er_, ec_):
                for k in range(-3, 3):
                    for a_, b_ in ((er_, ec_), (ec_, er_)):
                        pts = np.array([[*(cc + (k + 0.5) * p * a_ - 2.5 * p * b_), zt], [*(cc + (k + 0.5) * p * a_ + 2.5 * p * b_), zt]])
                        pc = apply_T(inv_T(T_BC), pts)
                        uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
                        cv2.line(img, tuple(map(int, uv[0])), tuple(map(int, uv[1])), colr, 2)

            lines(np.array(s["cells"][cell][:2]), (0, 0, 255), er, ec)
            a = np.radians(r["yaw_deg"])
            Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            lines(np.array(r["xy"]), (0, 255, 0), Rz @ er, Rz @ ec)
            cv2.putText(img, f"{cell}: red overview, green close-up, shift {r['shift_mm']} mm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(str(out / f"{f.stem}.jpg"), cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    ok = [r for r in rows if r["ok"]]
    if ok:
        sh = np.array([r["shift_mm"] for r in ok])
        n = np.hypot(*sh.T)
        print(f"{len(ok)}/{len(rows)} re-located; close-up minus overview: median {np.median(n):.1f} mm, 90% {np.percentile(n, 90):.1f} mm, "
              f"mean vector {np.round(sh.mean(0), 1).tolist()} mm (base x, y)")
    print(f"results {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
