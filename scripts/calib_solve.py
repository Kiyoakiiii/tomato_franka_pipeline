#!/usr/bin/env python
"""Wrist-camera calibration, step 2: solve the camera pose on the flange, with held-out validation.

All markers seen in a sample are used together (corner reprojection, one unknown base pose per marker). Every
k-th sample (default every 4th) is left out of the fit and used only to check how well the calibration predicts
the markers there. That held-out number, not the fit residual, is the accuracy estimate:
  lateral mm   = held-out corner reprojection error converted to mm at the marker distance (what matters for
                 locating a stem in the image)
  PnP 3-D mm   = marker centre from that sample's own PnP vs the fitted position (includes PnP depth noise,
                 so it is pessimistic for small markers)

    .venvs/pipeline/bin/python scripts/calib_solve.py calibration/sessions/wrist_<id> --save
    .venvs/pipeline/bin/python scripts/calib_solve.py --synthetic     # self-test, no hardware
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--max-reproj-px", type=float, default=1.5, help="drop single-marker detections with a worse corner fit")
    ap.add_argument("--save", action="store_true", help="write camera.T_flange_cam_file")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="only these marker ids (ignore stray detections)")
    ap.add_argument("--marker-mm", type=float, default=None, help="override the marker side used at collection (e.g. measured later)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import (load_samples, predict_marker_errors, save_T_FC, solve_eye_in_hand_markers,
                                      synthetic_wrist_samples)
    from seedling.transforms import apply_T, rotation_angle_deg

    truth = None
    if args.synthetic:
        samples, truth, _ = synthetic_wrist_samples()
    else:
        samples = load_samples(args.session)
    L = args.marker_mm / 1000.0 if args.marker_mm else float(samples[0]["marker_length_m"])
    BF, OBS, KS, n_drop = [], [], [], 0
    for s in samples:
        obs = {}
        for k, v in s["markers"].items():
            if args.ids is not None and int(k) not in args.ids:
                continue
            if v.get("reproj_px", 0) <= args.max_reproj_px:
                obs[int(k)] = np.array(v["corners_px"])
            else:
                n_drop += 1
        BF.append(np.array(s["T_BF"]))
        OBS.append(obs)
        KS.append(np.array(s["K"]))
    if n_drop:
        print(f"dropped {n_drop} marker detections with reprojection error > {args.max_reproj_px} px")
    idx = np.array([i for i in range(len(samples)) if OBS[i]])
    hold = idx[(np.arange(len(idx)) % args.holdout_every) == args.holdout_every - 1] if args.holdout_every > 1 else np.array([], int)
    fit = np.setdiff1d(idx, hold)
    print(f"{len(idx)} samples with markers: {len(fit)} fit, {len(hold)} held out; marker {L * 1000:.1f} mm")
    res = solve_eye_in_hand_markers([BF[i] for i in fit], [OBS[i] for i in fit], [KS[i] for i in fit], L)
    T_FC, T_BM = res["T_FC"], res["T_BM"]
    ps = np.array(res["per_sample_rms_px"])
    print(f"markers used: {res['ids']} (reference {res['ref_id']}); fit corner reprojection rms {res['fit_rms_px']:.2f} px, "
          f"worst sample {ps.max():.2f} px ({samples[fit[int(np.argmax(ps))]]['tag']})")
    print(f"  (pose-only initial guess differs from the refined one by {rotation_angle_deg(res['init_T_FC'][:3, :3], T_FC[:3, :3]):.2f} deg, "
          f"{np.linalg.norm(res['init_T_FC'][:3, 3] - T_FC[:3, 3]) * 1000:.1f} mm)")
    val = None
    if len(hold):
        e = [predict_marker_errors(T_FC, T_BM, BF[i], OBS[i], KS[i], L) for i in hold]
        px = np.concatenate([x["px"] for x in e])
        lat = np.concatenate([x["lateral_mm"] for x in e])
        pos = np.concatenate([x["pnp_pos_mm"] for x in e])
        val = {"n_samples": int(len(hold)), "n_markers": int(len(px)), "px_median": float(np.median(px)), "px_max": float(px.max()),
               "lateral_mm_median": float(np.median(lat)), "lateral_mm_max": float(lat.max()),
               "pnp_pos_mm_median": float(np.median(pos)), "pnp_pos_mm_max": float(pos.max())}
        print(f"HELD-OUT validation ({val['n_samples']} samples, {val['n_markers']} marker views):")
        print(f"  corner reprojection: median {val['px_median']:.2f} px, max {val['px_max']:.2f} px")
        print(f"  lateral error at the marker: median {val['lateral_mm_median']:.2f} mm, max {val['lateral_mm_max']:.2f} mm")
        print(f"  PnP 3-D centre error: median {val['pnp_pos_mm_median']:.2f} mm, max {val['pnp_pos_mm_max']:.2f} mm")
        verdict = "good" if val["lateral_mm_max"] < 2 else "usable, check" if val["lateral_mm_max"] < 4 else "poor - recollect"
        print(f"  -> {verdict}")
    stereo = [(np.array(v["depth_center_C"])[2] - np.array(v["T_CM"])[2][3]) * 1000
              for s in samples for v in s["markers"].values() if v.get("depth_center_C")]
    if stereo:
        print(f"stereo depth minus PnP depth at the markers (mm): median {np.median(stereo):+.1f}  "
              "(ZED depth bias, or a wrong --marker-mm if large)")
    if len(T_BM) > 1:
        c = {k: T[:3, 3] for k, T in T_BM.items()}
        ks = sorted(c)
        print("marker spacing on the sheet (mm, fitted): " + ", ".join(
            f"{a}-{b} {np.linalg.norm(c[a] - c[b]) * 1000:.1f}" for i, a in enumerate(ks) for b in ks[i + 1:]))
    eul = Rotation.from_matrix(T_FC[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\ncamera on the flange: position {np.round(T_FC[:3, 3] * 1000, 1).tolist()} mm, rotation xyz {np.round(eul, 1).tolist()} deg")
    print("  (sanity check against the physical mount: flange z = out of the flange, camera z = viewing direction)")
    for k, T in T_BM.items():
        print(f"marker {k} in base: {np.round(T[:3, 3], 4).tolist()} m")
    if truth is not None:
        p_C = np.array([0.02, -0.03, 0.45])
        err = np.linalg.norm(apply_T(BF[0] @ T_FC, p_C) - apply_T(BF[0] @ truth, p_C)) * 1000
        print(f"[synthetic] vs truth: rotation {rotation_angle_deg(T_FC[:3, :3], truth[:3, :3]):.3f} deg, "
              f"point 0.45 m in front of the camera mapped within {err:.2f} mm")
    if args.save and not args.synthetic:
        sid = Path(args.session).name
        meta = {"calibration_id": f"wrist_{sid}", "session": str(args.session), "n_fit": int(len(fit)), "fit_rms_px": res["fit_rms_px"],
                "heldout_validation": val, "marker_length_m": L,
                "T_base_marker_at_calibration": {str(k): T.tolist() for k, T in T_BM.items()},
                "flange_to_ee_used": json.loads((Path(args.session) / "collect_args.json").read_text()).get("flange_to_ee")}
        path = save_T_FC(cfg["camera"]["T_flange_cam_file"], T_FC, meta)
        (Path(args.session) / "solution.json").write_text(json.dumps({**meta, "T_flange_cam": T_FC.tolist()}, indent=1))
        print("saved", path)
    elif not args.synthetic:
        print("(not saved; add --save when the held-out error is acceptable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
