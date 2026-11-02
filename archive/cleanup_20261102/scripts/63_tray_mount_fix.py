#!/usr/bin/env python
"""After the camera was taken off and put back: correct its mounting (T_flange_cam) from a tray-consistency run
(61_tray_consistency.py, ideally with --close views too). The tray grid is the target (seedling/traycalib.py);
the arm's joint offsets are kept. Validation: every view in turn is left out, the correction and the tray are
fitted on the others, and the left-out view's cells are compared with that tray (mm on the tray plane).

    .venvs/pipeline/bin/python scripts/63_tray_mount_fix.py experiments/tray_consistency/<id>
    .venvs/pipeline/bin/python scripts/63_tray_mount_fix.py experiments/tray_consistency/<id> --save
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
from pathlib import Path

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def plane_points(v, T_FC, z):
    """The view's cell pixels intersected with the plane z through the mounting T_FC (base frame xy)."""
    from seedling import kinematics

    T_BC = kinematics.fk_flange(v["q"]) @ T_FC
    rays = (T_BC[:3, :3] @ (np.linalg.inv(v["K"]) @ np.column_stack([v["uv"], np.ones(len(v["uv"]))]).T)).T
    o = T_BC[:3, 3]
    return (o + rays * ((z - o[2]) / rays[:, 2])[:, None])[:, :2]


def deviation_mm(v, T_FC, fit, pitch):
    from seedling.traycalib import tray_points

    model = tray_points(fit["tray"], v["names"], pitch, fit["sign"])
    return np.linalg.norm(plane_points(v, T_FC, fit["tray"][3]) - model[:, :2], axis=1) * 1000


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("run", help="directory of a 61_tray_consistency.py run")
    ap.add_argument("--margin-px", type=float, default=30.0)
    ap.add_argument("--whole-only", action="store_true", help="use only the views that show the whole tray")
    ap.add_argument("--calib", default=None, help="the calibration file the run's grids were fitted with (default: the current one)")
    ap.add_argument("--no-depth", action="store_true", help="do not use the dense depth of the tray top (height constraint)")
    ap.add_argument("--save", action="store_true", help="write the corrected mounting (the old file is kept as a backup)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling import kinematics
    from seedling.calibration import load_T_FC, save_T_FC
    from seedling.camera import Frame
    from seedling.config import project_path
    from seedling.traycalib import cell_pixels, fit_mounting

    T_FC, meta = load_T_FC(cfg, args.calib)
    pitch = float(cfg["tray"]["model"]["pitch_m"])
    cols = int(cfg["tray"]["model"]["cols"])
    run = Path(args.run)
    views = []
    for r in json.loads((run / "views.json").read_text()):
        if not r.get("ok") or not r.get("cells") or (args.whole_only and r.get("close")):
            continue
        f = Frame.load(run / f"view{r['view']:02d}.npz")
        q = np.array(f.meta["q"])
        names, uv = cell_pixels(r["cells"], kinematics.fk_flange(q) @ T_FC, f.intr.K())
        H, W = f.rgb.shape[:2]
        keep = (uv[:, 0] > args.margin_px) & (uv[:, 1] > args.margin_px) & (uv[:, 0] < W - args.margin_px) & (uv[:, 1] < H - args.margin_px)
        # dense stereo points on the tray top (camera frame) near the fitted top plane: they fix the height
        T_BC = kinematics.fk_flange(q) @ T_FC
        okd = np.isfinite(f.xyz).all(-1)
        Pc = f.xyz[okd]
        Pb = (T_BC[:3, :3] @ Pc.T).T + T_BC[:3, 3]
        cxy = np.array([c[:2] for c in r["cells"].values()])
        lo, hi = cxy.min(0) - 0.01, cxy.max(0) + 0.01
        near = (np.abs(Pb[:, 2] - r["z_top"]) < 0.008) & (Pb[:, 0] > lo[0]) & (Pb[:, 0] < hi[0]) & (Pb[:, 1] > lo[1]) & (Pb[:, 1] < hi[1])
        plane_C = Pc[near][:: max(1, int(near.sum() // 3000))]
        views.append({"view": r["view"], "close": bool(r.get("close")), "q": q, "K": f.intr.K(), "cells": r["cells"],
                      "names": [n for n, k in zip(names, keep) if k], "uv": uv[keep], "plane_C": plane_C if not args.no_depth else None})
    n_close = sum(v["close"] for v in views)
    print(f"{len(views)} views ({n_close} close), {sum(len(v['names']) for v in views)} cell observations")
    if len(views) < 4:
        print("too few views")
        return 1

    def report(name, T, fit, vs):
        d = {v["view"]: deviation_mm(v, T, fit, pitch) for v in vs}
        allv = np.concatenate(list(d.values()))
        print(f"{name}: cell deviation from the fitted tray median {np.median(allv):.2f} mm, 95% {np.percentile(allv, 95):.2f}, max {allv.max():.2f}")
        return d

    fit0 = fit_mounting(views, T_FC, pitch, cols, fit_mount=False)
    report("current mounting", T_FC, fit0, views)
    fit = fit_mounting(views, T_FC, pitch, cols)
    print(f"correction: {fit['delta_rot_deg']:.2f} deg {np.round(fit['delta_rotvec_deg'], 2).tolist()} (camera axes), "
          f"{np.round(fit['delta_trans_mm'], 1).tolist()} mm; reprojection {fit['rms_px_before']:.2f} -> {fit['rms_px_after']:.2f} px")
    report("corrected (all views fitted)", fit["T_FC"], fit, views)
    print("leave one view out (correction + tray fitted on the others):")
    loo = {"whole": [], "close": []}
    for k, v in enumerate(views):
        rest = views[:k] + views[k + 1:]
        fk = fit_mounting(rest, T_FC, pitch, cols)
        d = deviation_mm(v, fk["T_FC"], fk, pitch)
        loo["close" if v["close"] else "whole"].append(d)
        print(f"  view {v['view']:2d} ({'close' if v['close'] else 'whole'}, joint 7 {v['q'][6]:+.2f}): median {np.median(d):.2f} mm, max {d.max():.2f} "
              f"[others: {fk['delta_rot_deg']:.2f} deg, {np.round(fk['delta_trans_mm'], 1).tolist()} mm]")
    summary = {}
    for kind, ds in loo.items():
        if ds:
            a = np.concatenate(ds)
            summary[kind] = {"median_mm": float(np.median(a)), "p95_mm": float(np.percentile(a, 95)), "max_mm": float(a.max())}
            print(f"LEFT-OUT {kind} views: median {np.median(a):.2f} mm, 95% {np.percentile(a, 95):.2f}, max {a.max():.2f}")
    out = {"run": str(run), "correction_rot_deg": fit["delta_rot_deg"], "correction_rotvec_deg": fit["delta_rotvec_deg"].tolist(),
           "correction_trans_mm": fit["delta_trans_mm"].tolist(), "rms_px": [fit["rms_px_before"], fit["rms_px_after"]], "leave_one_out": summary}
    (run / "mount_fix.json").write_text(json.dumps(out, indent=1))
    if args.save:
        old = project_path(cfg["camera"]["T_flange_cam_file"])
        backup = old.with_name(f"T_flange_cam.{meta.get('calibration_id', 'old')}.yaml")
        if not backup.exists():
            backup.write_text(old.read_text())
        new_meta = {k: v for k, v in meta.items() if k != "T_flange_cam"}
        new_meta["calibration_id"] = f"{meta.get('calibration_id')}+trayfix_{run.name}"
        new_meta["tray_mount_fix"] = out
        print("saved", save_T_FC(cfg["camera"]["T_flange_cam_file"], fit["T_FC"], new_meta), "(backup", backup.name + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
