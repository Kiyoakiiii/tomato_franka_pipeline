"""Correct the wrist camera's mounting (T_flange_cam) from views of the tray, after the camera was taken off and
put back. The tray is a target of known geometry (rows x cols cells at a known pitch, on a plane): in every view
the cell centres of the fitted grid give their pixels. One tray pose (x, y, yaw, height, pitch scale; plane
assumed level) and one small correction of the camera mounting (rotation + translation, in the camera frame)
are fitted so that the model grid reprojects onto those pixels in all views. The robot's joint offsets are
kept: they belong to the arm, not to the mounting.

The pixels come from grids fitted with the old mounting; a small mounting error mostly shifts the grid as a
whole in the top view, so the pixels it projects back to are still where the cells are in the image.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import kinematics
from .transforms import apply_T, inv_T, make_T


def cell_pixels(cells: dict, T_BC: np.ndarray, K: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = sorted(cells)
    P = apply_T(inv_T(T_BC), np.array([cells[k] for k in names], float))
    uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
    return names, uv


def _ij(names: list[str]) -> np.ndarray:
    return np.array([(ord(n[0]) - ord("A"), int(n[1:]) - 1) for n in names], float)  # (row, column)


def tray_points(t: np.ndarray, names: list[str], pitch: float, sign: int) -> np.ndarray:
    """t = (x0, y0, yaw, z, scale): A1 at (x0, y0, z); columns along yaw, rows to the left (sign +1) or right."""
    x0, y0, yaw, z, s = t
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    v = sign * np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    ij = _ij(names)
    return np.array([x0, y0, z]) + (pitch * s) * (ij[:, 1:2] * u + ij[:, 0:1] * v)


def _tray_guess(cells: dict, cols: int) -> tuple[np.ndarray, int]:
    A1, Ae, F1 = (np.array(cells[k], float) for k in ("A1", f"A{cols}", "F1"))
    u = (Ae - A1)[:2]
    yaw = float(np.arctan2(u[1], u[0]))
    sign = 1 if np.dot((F1 - A1)[:2], [-np.sin(yaw), np.cos(yaw)]) > 0 else -1
    return np.array([A1[0], A1[1], yaw, A1[2], 1.0]), sign


def fit_mounting(views: list[dict], T_FC: np.ndarray, pitch: float, cols: int = 12, fit_mount: bool = True,
                 prior_deg: float = 1.0, prior_mm: float = 10.0, fit_scale: bool = False, depth_px_per_mm: float = 2.0) -> dict:
    """views: dicts with q (7,), K (3,3), cells (dict of the view's fit), names, uv (N,2).
    Returns the corrected T_FC, the tray parameters and the pixel residuals before / after.
    The correction has a weak prior (prior_deg / prior_mm per pixel of residual). The pitch is held fixed by
    default: with a free scale, moving the camera along its axis and shrinking the tray explain the same pixels.
    A view may carry "plane_C" (N,3): dense stereo points on the tray top in the camera frame. Their height
    through the mounting must equal the tray's (weight depth_px_per_mm): the grid pixels alone barely tell
    a camera moved along its axis from a tray at another height (2026-10-22: 13 mm along the axis, the
    table then came out 1.4 cm too high)."""
    guess, sign = _tray_guess(views[0]["cells"], cols)
    T_BF = [kinematics.fk_flange(np.asarray(v["q"], float)) for v in views]

    def mount(d):
        return T_FC @ make_T(Rotation.from_rotvec(d[:3]).as_matrix(), d[3:6])

    def unpack(x):
        d, t = (x[:6], x[6:]) if fit_mount else (np.zeros(6), x)
        return d, (t if fit_scale else np.append(t, 1.0))

    def residuals(x):
        d, t = unpack(x)
        Tm = mount(d)
        out = []
        for v, TB in zip(views, T_BF):
            P = apply_T(inv_T(TB @ Tm), tray_points(t, v["names"], pitch, sign))
            uv = (v["K"] @ (P / P[:, 2:3]).T).T[:, :2]
            out.append((uv - v["uv"]).ravel())
            if v.get("plane_C") is not None and len(v["plane_C"]):
                zb = apply_T(TB @ Tm, v["plane_C"])[:, 2]
                out.append(np.array([(np.median(zb) - t[3]) * 1000 * depth_px_per_mm]))
        r = np.concatenate(out)
        if fit_mount:
            r = np.concatenate([r, np.degrees(d[:3]) / prior_deg, d[3:6] * 1000 / prior_mm])
        return r

    g = guess if fit_scale else guess[:4]
    x0 = np.concatenate([np.zeros(6), g]) if fit_mount else g
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    d, t = unpack(sol.x)

    def rms(r):  # pixel residuals only (the depth terms are interleaved per view)
        px, i = [], 0
        for v in views:
            m = 2 * len(v["names"])
            px.append(r[i:i + m])
            i += m + (1 if v.get("plane_C") is not None and len(v["plane_C"]) else 0)
        return float(np.sqrt(np.mean(np.sum(np.concatenate(px).reshape(-1, 2) ** 2, axis=1))))

    return {"T_FC": mount(d), "delta_rot_deg": float(np.degrees(np.linalg.norm(d[:3]))), "delta_rotvec_deg": np.degrees(d[:3]),
            "delta_trans_mm": d[3:6] * 1000, "tray": t, "sign": sign,
            "rms_px_before": rms(residuals(x0)), "rms_px_after": rms(residuals(sol.x))}
