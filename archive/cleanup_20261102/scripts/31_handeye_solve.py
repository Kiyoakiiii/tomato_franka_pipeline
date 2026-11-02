#!/usr/bin/env python
"""Solve T_base_cam from collected samples, with held-out validation.

Every k-th sample (default 4th) is held out from the fit and only used to report the
validation error. The fit residual is reported separately and is not an accuracy claim.

    .venvs/pipeline/bin/python scripts/31_handeye_solve.py calibration/sessions/<id> [--save]
    .venvs/pipeline/bin/python scripts/31_handeye_solve.py --synthetic   # self-test with simulated data
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _common import base_parser, config_from_args



def synthetic_samples(n: int = 24, noise_px_m: float = 0.0015, seed: int = 0):
    from seedling.synthetic import look_at
    from seedling.transforms import inv_T, make_T

    rng = np.random.default_rng(seed)
    T_BC = look_at(np.array([0.55, -0.55, 0.45]), np.array([0.5, 0.0, 0.15]))
    T_EM = make_T(Rotation.from_euler("xyz", [90, 0, 30], degrees=True).as_matrix(), [0.0, 0.05, -0.02])
    samples = []
    for _ in range(n):
        T_BE = make_T(Rotation.from_euler("xyz", [180 + rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-40, 40)], degrees=True).as_matrix(),
                      [0.5 + rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 0.25 + rng.uniform(-0.05, 0.05)])
        T_CM = inv_T(T_BC) @ T_BE @ T_EM
        T_CM[:3, 3] += rng.normal(0, noise_px_m, 3)
        T_CM[:3, :3] = T_CM[:3, :3] @ Rotation.from_rotvec(rng.normal(0, np.radians(0.3), 3)).as_matrix()
        samples.append({"tag": f"s{_}", "T_BE": T_BE.tolist(), "T_CM": T_CM.tolist(), "depth_center_C": None})
    return samples, T_BC


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--save", action="store_true", help="write camera.T_base_cam_file")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import load_samples, residuals_per_sample, save_T_BC, solve_eye_to_hand
    from seedling.transforms import apply_T, inv_T, rotation_angle_deg

    truth = None
    if args.synthetic:
        samples, truth = synthetic_samples()
    else:
        samples = load_samples(args.session)
    T_BE = [np.array(s["T_BE"]) for s in samples]
    T_CM = [np.array(s["T_CM"]) for s in samples]
    idx = np.arange(len(samples))
    hold = idx[(idx % args.holdout_every) == args.holdout_every - 1] if args.holdout_every > 1 else np.array([], int)
    fit = np.setdiff1d(idx, hold)
    print(f"{len(samples)} samples: {len(fit)} fit, {len(hold)} held out")
    res = solve_eye_to_hand([T_BE[i] for i in fit], [T_CM[i] for i in fit])
    T_BC, T_EM = res["T_BC"], res["T_EM"]
    fr = np.array(res["fit_residual_mm"])
    print(f"fit residual (marker origin, mm): rms {res['fit_rms_mm']:.2f}  median {np.median(fr):.2f}  max {fr.max():.2f}")
    worst = np.argsort(fr)[::-1][:3]
    print("  worst fit samples:", [(samples[fit[i]]["tag"], round(float(fr[i]), 2)) for i in worst])
    val = None
    if len(hold):
        hv = residuals_per_sample(T_BC, T_EM, [T_BE[i] for i in hold], [T_CM[i] for i in hold]) * 1000
        # signed per-axis error in base frame
        o = np.zeros(3)
        dv = np.array([apply_T(T_BC @ T_CM[i], o) - apply_T(T_BE[i] @ T_EM, o) for i in hold]) * 1000
        val = {"n": int(len(hold)), "median_mm": float(np.median(hv)), "max_mm": float(hv.max()), "rms_mm": float(np.sqrt(np.mean(hv**2))),
               "mean_signed_xyz_mm": dv.mean(0).round(2).tolist()}
        print(f"HELD-OUT validation (mm): median {val['median_mm']:.2f}  rms {val['rms_mm']:.2f}  max {val['max_mm']:.2f}  mean signed xyz {val['mean_signed_xyz_mm']}")
    stereo = [(np.array(s["depth_center_C"])[2] - np.array(s["T_CM"])[2][3]) * 1000 for s in samples if s.get("depth_center_C")]
    if stereo:
        print(f"stereo depth minus PnP depth at marker centre (mm): median {np.median(stereo):+.1f}  (diagnostic for ZED depth bias)")
    print("T_base_cam =\n", np.round(T_BC, 4))
    print("camera position in base [m]:", np.round(T_BC[:3, 3], 4).tolist())
    if truth is not None:
        p_C = apply_T(inv_T(truth), np.array([0.5, 0.0, 0.10]))  # a tray point seen by the camera
        werr = np.linalg.norm(apply_T(T_BC, p_C) - np.array([0.5, 0.0, 0.10])) * 1000
        print(f"[synthetic] vs truth: camera origin {np.linalg.norm(T_BC[:3,3]-truth[:3,3])*1000:.2f} mm, rotation {rotation_angle_deg(T_BC[:3,:3], truth[:3,:3]):.3f} deg, "
              f"mapped tray point {werr:.2f} mm")
    if args.save and not args.synthetic:
        sid = Path(args.session).name
        meta = {"calibration_id": f"handeye_{sid}", "session": str(args.session), "n_fit": int(len(fit)), "fit_rms_mm": res["fit_rms_mm"],
                "heldout_validation": val, "T_ee_marker": T_EM.tolist()}
        path = save_T_BC(cfg["camera"]["T_base_cam_file"], T_BC, meta)
        print("saved", path)
        (Path(args.session) / "solution.json").write_text(json.dumps({**meta, "T_base_cam": T_BC.tolist()}, indent=1))
    elif not args.synthetic:
        print("(not saved; add --save when the held-out error is acceptable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
