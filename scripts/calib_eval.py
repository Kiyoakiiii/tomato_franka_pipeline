#!/usr/bin/env python
"""Wrist-camera calibration, step 3: test a calibration on sessions it was not fitted to, optionally refit on several.

The sheet may lie anywhere in a test session (e.g. on the raised tray next to the seedlings, seen obliquely from
the working start pose): with T_FC held fixed the marker poses are fitted on all samples but one, and that one
sample is predicted (leave-one-out). A wrong T_FC cannot explain views from different directions with a single
marker pose, so this held-out error is the accuracy of the calibration under the test session's conditions.

    # how good is the current calibration at the working pose (session collected with calib_collect.py from the start pose)?
    .venvs/pipeline/bin/python scripts/calib_eval.py calibration/sessions/wrist_<new>
    # fit on all sessions together; validate by leaving each session out in turn; --save to use it
    .venvs/pipeline/bin/python scripts/calib_eval.py calibration/sessions/wrist_<old> calibration/sessions/wrist_<new> --refit
    .venvs/pipeline/bin/python scripts/calib_eval.py --synthetic      # self-test, no hardware

Also prints, per session, the stereo disparity offset that makes the ZED dense depth agree with the marker PnP
distances (camera.disparity_offset_px).
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _common import base_parser, config_from_args

DEFAULT_BASELINE_M = 0.11991  # ZED 2i serial 31983704 (older sessions did not record it)


def load_session(path, si, ids, max_reproj_px, L_override):
    """Rows with the reported joint angles; T_BF is filled in per calibration (its joint offsets)."""
    from seedling.calibration import load_samples

    samples = load_samples(path)
    L = L_override or float(samples[0]["marker_length_m"])
    rows = []
    for s in samples:
        # marker keys made unique per session (the same sheet ids lie elsewhere in another session)
        obs = {si * 100 + int(k): np.array(v["corners_px"]) for k, v in s["markers"].items()
               if (ids is None or int(k) in ids) and v.get("reproj_px", 0) <= max_reproj_px}
        if obs:
            rows.append({"tag": s["tag"], "q": np.array(s["q"]), "obs": obs, "K": np.array(s["K"]), "raw": s})
    return rows, L


def with_T_BF(rows, dq):
    from seedling import kinematics

    return [{**r, "T_BF": kinematics.fk_flange(r["q"], dq)} for r in rows]


def loo(T_FC, dq, rows, L):
    from seedling.calibration import leave_one_out_errors

    rows = with_T_BF(rows, dq)
    return leave_one_out_errors(T_FC, [r["T_BF"] for r in rows], [r["obs"] for r in rows], [r["K"] for r in rows], L), rows


def summarise(errs, rows, T_FC):
    from seedling.calibration import view_elevation_deg

    px = np.concatenate([e["px"] for e in errs]) if errs else np.array([])
    lat = np.concatenate([e["lateral_mm"] for e in errs]) if errs else np.array([])
    pos = np.concatenate([e["pnp_pos_mm"] for e in errs]) if errs else np.array([])
    elev = [view_elevation_deg(r["T_BF"] @ T_FC) for r in rows]
    worst = int(np.argmax([max(e["lateral_mm"], default=0) for e in errs]))
    return {"n_samples": len(rows), "n_marker_views": int(len(px)),
            "px_median": float(np.median(px)), "px_max": float(px.max()),
            "lateral_mm_median": float(np.median(lat)), "lateral_mm_p90": float(np.percentile(lat, 90)), "lateral_mm_max": float(lat.max()),
            "pnp_pos_mm_median": float(np.median(pos)),
            "elevation_deg": [round(min(elev), 1), round(max(elev), 1)], "worst_sample": rows[worst]["tag"]}


def print_summary(name, s):
    verdict = "good" if s["lateral_mm_p90"] < 2 else "usable" if s["lateral_mm_p90"] < 4 else "poor"
    print(f"  {name}: {s['n_samples']} samples ({s['n_marker_views']} marker views), camera looking down "
          f"{s['elevation_deg'][0]:.0f}-{s['elevation_deg'][1]:.0f} deg")
    print(f"    held-out corner error median {s['px_median']:.2f} px, max {s['px_max']:.2f} px")
    print(f"    held-out lateral error median {s['lateral_mm_median']:.2f} mm, 90% {s['lateral_mm_p90']:.2f} mm, "
          f"max {s['lateral_mm_max']:.2f} mm (worst {s['worst_sample']})  -> {verdict}")


def stereo_offset(rows, fx_default=None):
    """Disparity offset (px) that makes the dense depth at each marker centre equal its PnP distance."""
    offs = []
    for r in rows:
        meta = r["raw"].get("camera_meta", {})
        fB = r["K"][0, 0] * float(meta.get("baseline_m", DEFAULT_BASELINE_M))
        cur = float(meta.get("disparity_offset_px", 0.0))
        for v in r["raw"]["markers"].values():
            if v.get("depth_center_C") and np.isfinite(v["depth_center_C"][2]):
                offs.append(cur + fB / v["depth_center_C"][2] - fB / v["T_CM"][2][3])
    return offs


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--calib", default=None, help="calibration file to test (default camera.T_flange_cam_file)")
    ap.add_argument("--refit", action="store_true", help="also fit one calibration on all given sessions")
    ap.add_argument("--save", action="store_true", help="with --refit: write it (the previous file is kept as a backup)")
    ap.add_argument("--joint-offsets", type=int, nargs="*", default=[5],
                    help="with --refit: robot joints (1-7) whose reported angle gets a fitted constant offset; none: plain")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--max-reproj-px", type=float, default=1.5)
    ap.add_argument("--marker-mm", type=float, default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.calibration import load_T_FC, save_T_FC, solve_eye_in_hand_joint_offsets
    from seedling.config import project_path
    from seedling.transforms import rotation_angle_deg

    if args.synthetic:
        return synthetic_selftest()
    if not args.sessions:
        ap.error("give at least one session directory")
    L_over = args.marker_mm / 1000.0 if args.marker_mm else None
    sessions = []
    for si, p in enumerate(args.sessions):
        rows, L = load_session(p, si, args.ids, args.max_reproj_px, L_over)
        sessions.append((Path(p).name, rows, L))
    Ls = {round(L, 5) for _, _, L in sessions}
    if len(Ls) > 1:
        raise SystemExit(f"sessions use different marker sizes {Ls}; this tool assumes one sheet")
    L = sessions[0][2]

    T_cur, meta = load_T_FC(cfg, args.calib)
    dq_cur = np.array(meta.get("joint_offsets_rad") or np.zeros(7))
    print(f"calibration under test: {meta.get('calibration_id')} (fitted on {meta.get('session')}); "
          f"joint offsets {np.round(np.degrees(dq_cur), 3).tolist()} deg")
    report = {"calibration": meta.get("calibration_id"), "sessions": {}}
    for name, rows, _ in sessions:
        errs, rows = loo(T_cur, dq_cur, rows, L)
        s = summarise(errs, rows, T_cur)
        fitted_on = meta.get("session") or []
        if name in {Path(x).name for x in ([fitted_on] if isinstance(fitted_on, str) else fitted_on)}:
            name += " (its own fitting data: optimistic)"
        print_summary(name, s)
        offs = stereo_offset(rows)
        if offs:
            s["disparity_offset_px"] = float(np.median(offs))
            print(f"    stereo disparity offset {np.median(offs):.1f} px (spread {np.percentile(offs, 10):.1f}..{np.percentile(offs, 90):.1f}; "
                  f"config {float(cfg['camera'].get('disparity_offset_px', 0)):.1f})")
        report["sessions"][name] = s

    if args.refit:
        joints = tuple(j - 1 for j in args.joint_offsets)
        if any(j in (0, 6) for j in joints):
            raise SystemExit("joints 1 and 7 cannot be identified from marker views (they act like moving the sheet / the mount)")

        def fit(rows):
            return solve_eye_in_hand_joint_offsets([r["q"] for r in rows], [r["obs"] for r in rows], [r["K"] for r in rows], L, joints)

        all_rows = [r for _, rows, _ in sessions for r in rows]
        res = fit(all_rows)
        T_new, dq_new = res["T_FC"], res["joint_offsets_rad"]
        print(f"\nREFIT on all {len(all_rows)} samples, joint offsets on joints {list(args.joint_offsets)}: fit rms {res['fit_rms_px']:.2f} px")
        print(f"  joint offsets (added to the reported angles) {np.round(np.degrees(dq_new), 3).tolist()} deg")
        eul = Rotation.from_matrix(T_new[:3, :3]).as_euler("xyz", degrees=True)
        print(f"  camera on the flange: position {np.round(T_new[:3, 3] * 1000, 1).tolist()} mm, rotation xyz {np.round(eul, 1).tolist()} deg")
        print(f"  vs the tested calibration: rotation {rotation_angle_deg(T_cur[:3, :3], T_new[:3, :3]):.2f} deg, "
              f"position {np.linalg.norm(T_cur[:3, 3] - T_new[:3, 3]) * 1000:.1f} mm")
        report["refit"] = {"T_flange_cam": T_new.tolist(), "joint_offsets_rad": dq_new.tolist(), "fit_rms_px": res["fit_rms_px"], "validation": {}}
        if len(sessions) > 1:
            print("  validation, each session left out of the fit in turn (offsets and T_FC from the other sessions):")
            for k, (name, rows, _) in enumerate(sessions):
                r_k = fit([r for j, (_, rs, _) in enumerate(sessions) if j != k for r in rs])
                errs, rows_k = loo(r_k["T_FC"], r_k["joint_offsets_rad"], rows, L)
                s = summarise(errs, rows_k, r_k["T_FC"])
                print_summary(f"{name} (not in the fit; offsets there {np.round(np.degrees(r_k['joint_offsets_rad'][list(joints)]), 2).tolist()} deg)", s)
                report["refit"]["validation"][name] = s
        else:
            print("  (only one session: no independent validation of the refit; collect a second one)")
        if args.save:
            path = project_path(cfg["camera"]["T_flange_cam_file"])
            backup = path.with_name(f"{path.stem}.{meta.get('calibration_id', 'previous')}{path.suffix}")
            if path.exists() and not backup.exists():
                shutil.copy2(path, backup)
                print("previous calibration kept as", backup)
            T_BM = res["T_BM"]
            new_meta = {"calibration_id": "wrist_multi_" + "+".join(n.replace("wrist_", "") for n, _, _ in sessions),
                        "session": [str(p) for p in args.sessions], "n_fit": len(all_rows), "fit_rms_px": res["fit_rms_px"],
                        "heldout_validation": report["refit"]["validation"] or None, "marker_length_m": L,
                        # markers of the LAST session (ids as printed on the sheet): the sheet as it lies now
                        "T_base_marker_at_calibration": {str(k % 100): T.tolist() for k, T in T_BM.items() if k // 100 == len(sessions) - 1},
                        "flange_to_ee_used": meta.get("flange_to_ee_used"),
                        "joint_offsets_rad": dq_new.tolist(),
                        "joint_offsets_note": "added to the reported joint angles before forward kinematics (kinematics.JOINT_OFFSETS)"}
            print("saved", save_T_FC(cfg["camera"]["T_flange_cam_file"], T_new, new_meta))
    out = Path(args.sessions[-1]) / "eval.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print("report:", out)
    return 0


def synthetic_selftest() -> int:
    """True calibration -> small held-out error; the same data with a 1 deg wrong calibration -> clearly larger."""
    from seedling.calibration import leave_one_out_errors, synthetic_wrist_samples
    from seedling.transforms import make_T

    samples, T_true, _ = synthetic_wrist_samples(3)
    rows = [{"tag": s["tag"], "T_BF": np.array(s["T_BF"]), "obs": {int(k): np.array(v["corners_px"]) for k, v in s["markers"].items()},
             "K": np.array(s["K"]), "raw": s} for s in samples if s["markers"]]
    for label, T in (("true", T_true), ("1 deg off", T_true @ make_T(Rotation.from_euler("x", 1, degrees=True).as_matrix()))):
        errs = leave_one_out_errors(T, [r["T_BF"] for r in rows], [r["obs"] for r in rows], [r["K"] for r in rows], 0.04)
        print_summary(label, summarise(errs, rows, T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
