"""Where the substrate of one cell actually is: its surface height and centre, from the dense stereo points of
peat-coloured pixels around the cell centre. Works whether the plug stands above the tray, level with it or
sunk below its rim (only the soil itself is used, not the tray). The upper part of the peat points is the top
surface (a raised plug also shows its sides, lower down).
"""
from __future__ import annotations

import numpy as np

from .camera import Frame
from .stemline import _mask, substrate_mask
from .transforms import apply_T


def measure_plug(frame: Frame, T_BC: np.ndarray, cell_xy, cfg: dict, search_r: float = 0.018, max_shift: float = 0.012,
                 min_points: int = 60) -> dict:
    """search_r: well inside the plug (neighbouring plugs start ~25 mm from the cell centre). The height is the
    most populated 2 mm height band of the soil points (the top surface), refined by their median; the centre
    of the visible soil is reported but leaves hide one side, so the caller should keep the cell centre."""
    ok = np.isfinite(frame.xyz).all(-1)
    soil = substrate_mask(frame.rgb, cfg["stemline"]) & ~_mask(frame.rgb, cfg["stemline"]) & ok
    P = apply_T(T_BC, frame.xyz[soil])
    c = np.asarray(cell_xy, float)[:2]
    near = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < search_r
    Q = P[near]
    if len(Q) < min_points:
        return {"ok": False, "status": f"too little soil in view ({len(Q)} points; covered by leaves or out of view)", "n": int(len(Q))}
    h, e = np.histogram(Q[:, 2], bins=np.arange(Q[:, 2].min(), Q[:, 2].max() + 0.004, 0.002))
    hs = np.convolve(h, [1, 2, 1], mode="same")
    z_top = float(e[int(np.argmax(hs))] + 0.001)
    top = Q[np.abs(Q[:, 2] - z_top) < 0.005]
    if len(top) < min_points // 2:
        return {"ok": False, "status": "no flat top surface", "n": int(len(Q))}
    z = float(np.median(top[:, 2]))
    centre = top[:, :2].mean(0)
    shift = centre - c
    if np.linalg.norm(shift) > max_shift:  # one-sided visibility pulls the mean: keep the cell centre, report it
        centre = c
        note = f"centre shift {np.linalg.norm(shift) * 1000:.0f} mm not used (> {max_shift * 1000:.0f} mm)"
    else:
        note = ""
    return {"ok": True, "status": "ok" + (f" ({note})" if note else ""), "soil_z": round(z, 4),
            "centre": [round(float(centre[0]), 4), round(float(centre[1]), 4)],
            "shift_mm": [round(float(shift[0]) * 1000, 1), round(float(shift[1]) * 1000, 1)],
            "n": int(len(Q)), "n_top": int(len(top)), "spread_mm": round(float(np.percentile(top[:, 2], 90) - np.percentile(top[:, 2], 10)) * 1000, 1)}
