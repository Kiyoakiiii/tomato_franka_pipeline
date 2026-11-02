#!/usr/bin/env python
"""Run the perception chain on saved frames (no hardware).

Inputs may be capture .npz files, legacy dataset folders (stem_rgb.png + stem_depth_mm.npy),
directories containing either, or ``--synthetic N`` rendered scenes (ground-truth masks; reports
3D localisation error against the known stem axis).

    .venvs/pipeline/bin/python scripts/11_perceive_offline.py "data/raw/tomato images" --limit 20
    .venvs/pipeline/bin/python scripts/11_perceive_offline.py --synthetic 20
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import csv
import json
import time
from pathlib import Path

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def expand_inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir() and (p / "stem_rgb.png").exists():
            out.append(p)
        elif p.is_dir():
            out += sorted(q for q in p.iterdir() if (q.is_dir() and (q / "stem_rgb.png").exists()) or q.suffix == ".npz")
        else:
            out.append(p)
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--synthetic", type=int, default=0, help="render N random synthetic scenes instead")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-calib", action="store_true", help="ignore calibration/T_base_cam.yaml (camera-frame results only)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.camera import load_frame_any
    from seedling.perception import observe, save_observation
    from seedling.transforms import apply_T

    out = Path(args.out) if args.out else PROJECT_DIR / "experiments" / "perception" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.synthetic:
        from seedling.synthetic import GTSegmenter, default_sim_camera, default_world, render

        cfg["grasp"]["table_z_m"] = 0.05
        seg = GTSegmenter()
        T_BC, intr = default_sim_camera()
        for i in range(args.synthetic):
            world = default_world(i, 1)
            frame = render(world, T_BC, intr, seed=i)
            obs = observe(None, seg, cfg, T_BC, frame=frame)
            save_observation(obs, out, f"synthetic_{i:03d}", cfg, T_BC, save_frame=False)
            s = world.stems[0]
            for g in obs.geoms:
                for sg in g.segments:
                    if not sg.ok:
                        continue
                    p = apply_T(T_BC, np.array(sg.p_C))
                    r = p - s.base_B
                    axis_err = np.linalg.norm(r - (r @ s.dir_B) * s.dir_B)
                    d = apply_T(T_BC, np.array(sg.dir_C)) - T_BC[:3, 3]
                    ang = np.degrees(np.arccos(min(1.0, abs(d @ s.dir_B) / np.linalg.norm(d))))
                    rows.append({"frame": i, "segment": sg.seg_index, "axis_err_mm": axis_err * 1000, "tangent_err_deg": ang,
                                 "diam_err_mm": (sg.diameter_m - 2 * s.radius) * 1000, "rms_mm": sg.rms_m * 1000})
        if rows:
            e = np.array([r["axis_err_mm"] for r in rows])
            a = np.array([r["tangent_err_deg"] for r in rows])
            dd = np.array([r["diam_err_mm"] for r in rows])
            print(f"synthetic: {len(rows)} ok segments over {args.synthetic} scenes")
            print(f"  axis position error mm: median {np.median(e):.2f}  p95 {np.percentile(e, 95):.2f}  max {e.max():.2f}")
            print(f"  tangent error deg:      median {np.median(a):.2f}  p95 {np.percentile(a, 95):.2f}")
            print(f"  diameter error mm:      median {np.median(dd):+.2f}")
    else:
        from seedling.segmentation import StemSegmenter

        seg = StemSegmenter(cfg)
        T_BC = None
        if not args.no_calib:
            try:
                from seedling.calibration import load_T_BC

                T_BC, _ = load_T_BC(cfg)
            except FileNotFoundError:
                print("no camera extrinsics: reporting camera-frame geometry only")
        items = expand_inputs(args.inputs)
        if args.limit:
            items = items[: args.limit]
        for i, p in enumerate(items):
            frame = load_frame_any(p)
            obs = observe(None, seg, cfg, T_BC, frame=frame)
            name = p.stem if p.suffix else p.name
            save_observation(obs, out, name, cfg, T_BC, save_frame=False)
            segs = [s for g in obs.geoms for s in g.segments]
            ok = [s for s in segs if s.ok]
            best = max(ok, key=lambda s: s.score) if ok else None
            rows.append({
                "input": str(p), "instances": len(obs.instances), "top_conf": round(max([x.conf for x in obs.instances], default=0), 3),
                "segments": len(segs), "ok_segments": len(ok),
                "median_valid_fraction": round(float(np.median([s.valid_fraction for s in segs])), 3) if segs else "",
                "best_depth_m": round(best.p_C[2], 4) if best else "", "best_diameter_mm": round(best.diameter_m * 1000, 2) if best else "",
                "best_rms_mm": round(best.rms_m * 1000, 2) if best else "", "reject": obs.rejection_summary()[:120],
                "feasible_grasps": sum(c.feasible for c in obs.candidates),
            })
            print(f"[{i+1}/{len(items)}] {name}: inst={rows[-1]['instances']} ok_seg={len(ok)}/{len(segs)} {rows[-1]['reject']}")
        n_ok = sum(1 for r in rows if r["ok_segments"])
        print(f"\n{n_ok}/{len(rows)} frames have at least one valid 3D stem segment")
    if rows:
        with (out / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "run.json").write_text(json.dumps({"inputs": args.inputs, "synthetic": args.synthetic, "config": cfg["_path"]}, indent=1))
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
