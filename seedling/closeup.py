"""Close-up re-location of the cell to grasp. The overview fit (seedling/overview.py) puts every cell to a few mm
from 0.8-1 m away; grasping goes by those coordinates. The close-up (scripts/pick.py observe, ~0.35-0.42 m)
sees the cell walls around the chosen cell at 2-3x the resolution and from another pose: the walls of the cells
around it (up to 5 x 5) are matched again there, in a window smaller than half a cell around the overview cells, so
the close-up can move the grasp point by mm but never onto the next cell.

Same idea as traygrid.locate_tray (a top view of the tray top plane matched with a template of dark cells and
bright walls), but local and in the tray's own axes (any tray yaw), with leaves masked out, and the plane height
from the close-up's own depth on the walls."""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Frame
from .transforms import apply_T


def _rc(name: str) -> tuple[int, int]:
    return ord(name[0]) - ord("A"), int(name[1:]) - 1


def tray_axes(cells: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """unit vectors along the rows (A -> B) and the columns (1 -> 2) in base xy, and the pitch (from the cells' spread)"""
    rows = max(_rc(k)[0] for k in cells) + 1
    cols = max(_rc(k)[1] for k in cells) + 1
    P = lambda k: np.array(cells[k][:2], float)
    er = (P(f"{chr(ord('A') + rows - 1)}1") - P("A1")) / (rows - 1)
    ec = (P(f"A{cols}") - P("A1")) / (cols - 1)
    pitch = 0.5 * (np.linalg.norm(er) + np.linalg.norm(ec))
    return er / np.linalg.norm(er), ec / np.linalg.norm(ec), float(pitch)


def _template(r0: int, c0: int, rows: int, cols: int, n: int, p: float, wall: float, rim: float, res: float):
    """axis-aligned local template around cell (r0, c0): 1 on walls and rim, 0 in the cells; mask 0 off the tray.
    Pixel (i, j) is at v = (i - h) * res along the rows, u = (j - h) * res along the columns from the cell centre."""
    half = (n + 0.5) * p + rim
    h = int(round(half / res))
    s = np.arange(-h, h + 1) * res
    V, U = np.meshgrid(s, s, indexing="ij")
    # position in tray cell units, from the tray's corner (row 0, column 0 edge)
    fr = V / p + r0 + 0.5
    fc = U / p + c0 + 0.5
    on = (fr >= -rim / p) & (fr <= rows + rim / p) & (fc >= -rim / p) & (fc <= cols + rim / p)
    dr = np.abs(fr - np.round(fr)) * p  # distance to the nearest wall line (m)
    dc = np.abs(fc - np.round(fc)) * p
    inside = (fr > 0) & (fr < rows) & (fc > 0) & (fc < cols)
    wall_px = (dr < wall / 2) | (dc < wall / 2) | ~inside
    t = np.where(wall_px, 1.0, 0.0).astype(np.float32)
    return t, on.astype(np.uint8), h


def _wall_z(P: np.ndarray, z0: float, lo: float = -0.02, hi: float = 0.035) -> float | None:
    """the tray top: the highest well-populated height peak of the plain points in [z0 + lo, z0 + hi] (the cell
    bottoms 4-5 cm lower form the other peak; the overview's z0 can be ~1 cm off)"""
    z = P[:, 2][(P[:, 2] > z0 + lo) & (P[:, 2] < z0 + hi)]
    if len(z) < 300:
        return None
    h, e = np.histogram(z, bins=np.arange(z0 + lo, z0 + hi + 0.002, 0.002))
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4
    pk = [i for i in range(len(hs)) if hs[i] >= 0.3 * hs.max() and (i == 0 or hs[i] >= hs[i - 1]) and (i == len(hs) - 1 or hs[i] >= hs[i + 1])]
    k = max(pk)
    sel = z[np.abs(z - (e[k] + 0.001)) < 0.004]
    return float(np.median(sel))


def refine_cell(frame: Frame, T_BC: np.ndarray, cells: dict, cell: str, cfg: dict, window_m: float = 0.018,
                yaw_deg: float = 3.0, n_around: int = 2, res: float = 0.001, band_m: float = 0.005, min_score: float = 0.15,
                min_seen: float = 0.3) -> dict:
    """The cell's centre from the close-up's depth: the points at the tray-top height (the wall tops; plants and peat
    left out) form the wall pattern of the cells around it (n_around each way, those on the tray), which is matched
    within window_m of the overview cells (and yaw within yaw_deg). Purely 3D: no plane to project the picture onto,
    so the cell bottoms' bright features and a wrong top height cannot pull it. Returns ok, status, xy (the refined
    centre, base m), shift_mm (refined - overview, base xy), z_top (measured here), yaw_deg (vs the overview), score,
    contrast (the peak above the window's spread), on_edge, seen (share of the template area with depth).

    82 close-ups of 2026-11-02..06: 79 matched (3 had no depth at all); the same cell seen again from another pose
    lands within ~1 mm (rms), as does the overview; the two differ by 1-10 mm after the 11-06 calibration (5-16 mm
    before it): a bias between the two views, not noise."""
    from .stemline import _mask, substrate_mask

    m = cfg["tray"]["model"]
    rows, cols = int(m["rows"]), int(m["cols"])
    wall, rim = float(m["wall_m"]), float(m["rim_m"])
    er, ec, p = tray_axes(cells)
    r0, c0 = _rc(cell)
    c = np.array(cells[cell][:2], float)
    z0 = float(cells[cell][2])
    ok = np.isfinite(frame.xyz).all(-1)
    if ok.sum() < 1000:
        return {"ok": False, "status": "no depth in the close-up", "xy": c.tolist(), "shift_mm": [0.0, 0.0], "z_overview": round(z0, 4)}
    Pall = apply_T(T_BC, frame.xyz[ok])
    plain = (~_mask(frame.rgb, cfg["stemline"]) & ~substrate_mask(frame.rgb, cfg["stemline"]))[ok]
    half = (n_around + 0.5) * p + rim + window_m + 0.003
    d = Pall[:, :2] - c
    near = (np.abs(d @ er) < half) & (np.abs(d @ ec) < half)
    P, Pp = Pall[near], Pall[near & plain]
    zw = _wall_z(Pp, z0)
    base = {"xy": c.tolist(), "shift_mm": [0.0, 0.0], "z_overview": round(z0, 4)}
    if zw is None:
        return {**base, "ok": False, "status": "no tray top in the close-up depth around the cell"}
    W = Pp[np.abs(Pp[:, 2] - zw) < band_m]  # wall tops (and the rim)
    t, tm, h = _template(r0, c0, rows, cols, n_around, p, wall, rim, res)
    hw = int(round(window_m / res))
    n = 2 * (h + hw) + 1

    def raster(Q, er_, ec_):
        dq = Q[:, :2] - c
        i = np.round((dq @ er_) / res).astype(int) + h + hw
        j = np.round((dq @ ec_) / res).astype(int) + h + hw
        k = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        img = np.zeros((n, n), np.float32)
        np.add.at(img, (i[k], j[k]), 1.0)
        return img

    best = None
    for dd in np.arange(-yaw_deg, yaw_deg + 1e-9, 0.25):
        a = np.radians(dd)
        Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        er_, ec_ = Rz @ er, Rz @ ec
        wimg = cv2.GaussianBlur(raster(W, er_, ec_), (0, 0), 1.5)
        seen = cv2.dilate((raster(P, er_, ec_) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        I = np.minimum(wimg / (np.percentile(wimg[wimg > 0], 90) + 1e-9), 1.0) if (wimg > 0).any() else wimg
        fill = float(np.median(I[seen])) if seen.any() else 0.0
        I = np.where(seen, I, fill).astype(np.float32)  # no depth there (out of view, or no texture): neutral
        R = np.nan_to_num(cv2.matchTemplate(I, t, cv2.TM_CCOEFF_NORMED, mask=tm), nan=-1, posinf=-1, neginf=-1)
        _, v, _, (x, y) = cv2.minMaxLoc(R)
        if best is None or v > best[0]:
            cov = float(seen[hw:hw + t.shape[0], hw:hw + t.shape[1]][tm > 0].mean())
            best = (float(v), dd, R, (x, y), er_, ec_, cov)
    v, dd, R, (x, y), er_, ec_, cov = best
    dx = dy = 0.0
    if 0 < x < R.shape[1] - 1 and 0 < y < R.shape[0] - 1:  # sub-pixel peak
        a_, b_, c_ = R[y, x - 1], R[y, x], R[y, x + 1]
        dx = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if (a_ - 2 * b_ + c_) else 0.0
        a_, b_, c_ = R[y - 1, x], R[y, x], R[y + 1, x]
        dy = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) if (a_ - 2 * b_ + c_) else 0.0
    dv, du = float(y + dy - hw) * res, float(x + dx - hw) * res  # the template's offset = the cell's (rows, columns)
    xy = c + dv * er_ + du * ec_
    on_edge = bool(x in (0, R.shape[1] - 1) or y in (0, R.shape[0] - 1))
    contrast = float((v - np.median(R)) / (R.std() + 1e-9))
    shift = (xy - c) * 1000
    out = {**base, "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
           "shift_mm": [round(float(shift[0]), 1), round(float(shift[1]), 1)],
           "shift_rows_cols_mm": [round(dv * 1000, 1), round(du * 1000, 1)], "z_top": round(zw, 4), "wall_points": int(len(W)),
           "yaw_deg": round(float(dd), 2), "score": round(v, 3), "contrast": round(contrast, 1), "on_edge": on_edge,
           "seen": round(cov, 2)}
    if on_edge:
        out.update(ok=False, status=f"best match at the edge of the {window_m * 1000:.0f} mm window: the overview may be a whole cell off")
    elif abs(dd) >= yaw_deg - 1e-6:
        out.update(ok=False, status=f"best match at the edge of the yaw range ({dd:+.1f} deg): camera pose doubtful")
    elif cov < min_seen or v < min_score:
        out.update(ok=False, status=f"too little of the cells around seen (depth on {cov * 100:.0f}% of them, match {v:.2f})")
    else:
        out.update(ok=True, status=f"ok: moved {np.hypot(*shift):.1f} mm")
    return out


def grasp_xy(obs: dict | None, cfg: dict, cell_xyz=None) -> tuple[list | None, str]:
    """where to grasp from the close-up re-location saved with the observation (pick.py observe: obs["closeup_xy"]): the
    refined cell centre when grasp.closeup_xy is "apply" and the re-location is sound, else None (the overview cell
    stays); and a line saying which and why. "log" (the default) only records it: not yet checked against the robot.
    cell_xyz: the cell as located now; a close-up made for another fit of the tray (moved since) is not used"""
    g = cfg.get("grasp", {})
    mode = g.get("closeup_xy", "log")
    cu = (obs or {}).get("closeup_xy")
    if not cu:
        return None, "no close-up re-location: the overview cell"
    sh = float(np.hypot(*cu.get("shift_mm", [0.0, 0.0])))
    if cell_xyz is not None and obs.get("overview_cell") is not None and             float(np.hypot(*(np.array(obs["overview_cell"][:2]) - np.array(cell_xyz[:2])))) > 0.003:
        return None, "the close-up was made for another tray fit: the overview cell"
    if not cu.get("ok"):
        return None, f"close-up re-location not usable ({cu.get('status')}): the overview cell"
    if mode != "apply":
        return None, f"close-up puts the cell {cu['shift_mm']} mm from the overview's (only logged: grasp.closeup_xy {mode})"
    lim = float(g.get("closeup_max_shift_mm", 15.0))
    if sh > lim:
        return None, f"close-up moves the cell {sh:.1f} mm, more than {lim:.0f}: the overview cell kept"
    return list(cu["xy"]), f"grasping at the close-up's cell centre: moved {cu['shift_mm']} mm from the overview's"
