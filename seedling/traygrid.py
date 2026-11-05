"""Tray cell positions from one picture: the cell layout is fixed (rows x cols, pitch), only the tray's place on
the table changes (x, y, yaw). The left image is resampled onto the tray's top plane as a 1 mm top view
(the plane height from the dense depth), and a template of the cell grid (dark cells, bright walls) is matched
over yaw and position. The whole tray must be in the picture: a partly visible grid matches equally well one
cell further, so a fit whose tray footprint leaves the picture is refused.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Frame
from .transforms import apply_T, inv_T


@dataclass
class TrayGrid:
    ok: bool
    status: str
    cells: dict | None = None  # "A1": [x, y, z] (base, m) of the cell centre on the tray top plane
    z_top: float | None = None
    yaw_deg: float | None = None
    score: float = 0.0
    outside_px: float = 0.0  # share of the fitted tray footprint not covered by the picture
    extra: dict | None = None

    def summary(self) -> dict:
        return {"ok": self.ok, "status": self.status, "z_top": self.z_top, "yaw_deg": self.yaw_deg, "score": round(self.score, 4),
                "outside": round(self.outside_px, 4), "A1": None if not self.cells else self.cells["A1"], **(self.extra or {})}


def tray_top_z(frame: Frame, T_BC: np.ndarray, lo: float, hi: float) -> float | None:
    """Most frequent height of the dense points in [lo, hi] m: the tray top (cell walls)."""
    ok = np.isfinite(frame.xyz).all(-1)
    z = apply_T(T_BC, frame.xyz[ok])[:, 2]
    z = z[(z > lo) & (z < hi)]
    if len(z) < 1000:
        return None
    h, e = np.histogram(z, bins=np.arange(lo, hi + 0.002, 0.002))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    # seen from above the cell interiors form a lower peak: take the highest peak of at least half the largest
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.5 * hs.max()]
    k = max(peaks) if peaks else int(np.argmax(h))
    sel = z[(z >= e[k] - 0.002) & (z < e[k + 1] + 0.002)]
    return float(np.median(sel))


def tray_top_candidates(frame: Frame, T_BC: np.ndarray, lo: float, hi: float, min_frac: float = 0.15, n: int = 3) -> list:
    """Height peaks in [lo, hi] that could be the tray top, highest first. With many empty cells in view the
    cell bottoms (4-5 cm lower; the black wall tops get little depth) can outnumber the wall tops by more than
    two to one, so the half-maximum rule of tray_top_z picks the cell bottoms: the grid is then fitted at each
    candidate and the best fit kept."""
    ok = np.isfinite(frame.xyz).all(-1)
    z = apply_T(T_BC, frame.xyz[ok])[:, 2]
    ha, ea = np.histogram(z[(z > -0.10) & (z < hi)], bins=np.arange(-0.10, hi + 0.004, 0.004))
    table = float(ea[int(np.argmax(ha))] + 0.002)  # the overview is mostly table
    z = z[(z > lo) & (z < hi)]
    if len(z) < 1000:
        return []
    h, e = np.histogram(z, bins=np.arange(lo, hi + 0.002, 0.002))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= min_frac * hs.max()]
    out = []
    for k in sorted(peaks, key=lambda i: -hs[i]):
        zc = float(np.median(z[(z >= e[k] - 0.002) & (z < e[k + 1] + 0.002)]))
        if all(abs(zc - o) > 0.015 for o in out):
            out.append(zc)
        if len(out) == n:
            break
    # a tray is at least ~3.5 cm tall: a peak closer above the table is cell bottoms (or something flat on it),
    # unless nothing else is left (a close view where the tray fills the picture)
    high = [c for c in out if c > table + 0.035]
    return sorted(high or out, reverse=True)


def z_scan_candidates(frame: Frame, T_BC: np.ndarray, cfg: dict, step: float = 0.005, n: int = 3) -> list:
    """Tray top heights to fit at: the grid template matched on the top view at every height of the search range
    (step 5 mm, half-size pictures, at the tray's angle found once), the n best local maxima of the match. The
    height histogram alone (tray_top_candidates) missed the tray top once the range reached the leaves (2026-11-04:
    a raised tray needs a range up to 0.40 m, and leaf planes at 0.16-0.22 m crowded out the top at 0.147)."""
    m = cfg["tray"]["model"]
    lo, hi = m["top_z_search_m"]
    rows, cols = int(m["rows"]), int(m["cols"])
    p, wall, rim = float(m["pitch_m"]) * 1000, float(m["wall_m"]) * 1000, float(m["rim_m"]) * 1000
    xr, yr = m["search_x_m"], m["search_y_m"]
    t0 = _template(rows, cols, p, wall, rim)
    t2 = cv2.resize(t0, (t0.shape[1] // 2, t0.shape[0] // 2), interpolation=cv2.INTER_AREA)
    hist = tray_top_candidates(frame, T_BC, lo, hi, n=6)

    def half(z):
        I, _, _, _ = top_view(frame, T_BC, z, xr, yr, res=0.002)
        return cv2.GaussianBlur(I, (0, 0), 0.5)

    # the angle: the coarse sweep at the histogram peaks (the angle barely depends on the height)
    ang0 = max((_match(half(z), t2, a)[0], a) for z in (hist[:3] or [(lo + hi) / 2]) for a in np.arange(-90.0, 90.0, 2.0))[1]
    zs = np.arange(lo, hi + 1e-9, step)
    sc = np.array([max(_match(half(z), t2, a)[0] for a in (ang0 - 2.0, ang0, ang0 + 2.0)) for z in zs])
    peaks = [i for i in range(len(zs)) if (i == 0 or sc[i] >= sc[i - 1]) and (i == len(zs) - 1 or sc[i] >= sc[i + 1])]
    # each refined to the dense depth there (median within 6 mm): the grid is periodic, and a fit 2-3 mm off the
    # true top plane was shifted by a whole cell on some pictures (the scan step alone is not enough)
    ok = np.isfinite(frame.xyz).all(-1)
    zd = apply_T(T_BC, frame.xyz[ok])[:, 2]
    out = []
    for i in sorted(peaks, key=lambda i: -sc[i]):
        near = zd[np.abs(zd - zs[i]) < 0.006]
        z = float(np.median(near)) if len(near) >= 1000 else float(zs[i])
        if all(abs(z - o) > 0.01 for o in out):
            out.append(z)
        if len(out) == n:
            break
    return out


def top_view(frame: Frame, T_BC: np.ndarray, z: float, x_range, y_range, res: float = 0.001):
    """Grey image I[ix, iy] of the plane z = const (x along rows, y along columns) and a validity mask."""
    xs = np.arange(x_range[0], x_range[1], res)
    ys = np.arange(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    Pc = apply_T(inv_T(T_BC), P)
    uv = frame.intr.project(Pc)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    valid = (Pc[:, 2] > 0.05) & (uv[:, 0] >= 1) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 2)
    U = np.where(valid, uv[:, 0], -10).astype(np.float32).reshape(X.shape)
    V = np.where(valid, uv[:, 1], -10).astype(np.float32).reshape(X.shape)
    I = cv2.remap(gray, U, V, cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    return I, valid.reshape(X.shape), xs, ys


def _template(rows: int, cols: int, p: float, wall: float, rim: float) -> np.ndarray:
    t = np.ones((int(round(rows * p + 2 * rim)), int(round(cols * p + 2 * rim))), np.float32)
    for r in range(rows):
        for c in range(cols):
            a, b = rim + r * p + wall / 2, rim + c * p + wall / 2
            t[int(round(a)):int(round(a + p - wall)), int(round(b)):int(round(b + p - wall))] = 0
    return t


def _match(I: np.ndarray, t0: np.ndarray, ang: float, window=None):
    """window: (function M -> expected template location (col, row), half size px): best match only inside it."""
    M = cv2.getRotationMatrix2D((t0.shape[1] / 2, t0.shape[0] / 2), ang, 1.0)
    ca, sa = abs(np.cos(np.radians(ang))), abs(np.sin(np.radians(ang)))
    W2, H2 = int(t0.shape[1] * ca + t0.shape[0] * sa) + 2, int(t0.shape[0] * ca + t0.shape[1] * sa) + 2
    M[:, 2] += [(W2 - t0.shape[1]) / 2, (H2 - t0.shape[0]) / 2]
    t = cv2.warpAffine(t0, M, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=0)
    m = cv2.warpAffine(np.ones_like(t0), M, (W2, H2), flags=cv2.INTER_NEAREST, borderValue=0)
    if t.shape[0] >= I.shape[0] or t.shape[1] >= I.shape[1]:
        return -1.0, M, (0.0, 0.0), m
    # where the picture is blank (outside the camera view) the normalised correlation divides by zero: inf, not a match
    R = np.nan_to_num(cv2.matchTemplate(I, t, cv2.TM_CCORR_NORMED, mask=m), nan=-1, posinf=-1, neginf=-1)
    if window is not None:
        c0, half = window[0](M), window[1]
        W = np.full_like(R, -1)
        x0, x1 = int(max(c0[0] - half, 0)), int(min(c0[0] + half + 1, R.shape[1]))
        y0, y1 = int(max(c0[1] - half, 0)), int(min(c0[1] + half + 1, R.shape[0]))
        if x0 >= x1 or y0 >= y1:
            return -1.0, M, (0.0, 0.0), m
        W[y0:y1, x0:x1] = R[y0:y1, x0:x1]
        R = W
    _, v, _, (x, y) = cv2.minMaxLoc(R)
    dx = dy = 0.0
    if 0 < x < R.shape[1] - 1 and 0 < y < R.shape[0] - 1:  # sub-pixel peak
        a, b, c = R[y, x - 1], R[y, x], R[y, x + 1]
        dx = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
        a, b, c = R[y - 1, x], R[y, x], R[y + 1, x]
        dy = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return float(v), M, (x + dx, y + dy), m


def locate_tray(frame: Frame, T_BC: np.ndarray, cfg: dict, prior: dict | None = None, window_m: float = 0.02,
                z_top: float | None = None, z_hint: float | None = None, fast_min_score: float = 0.66) -> TrayGrid:
    """prior: cells of an earlier fit; then only yaw within 1 deg and shifts within window_m of it are searched,
    so a close view that shows only part of the tray is located without whole-cell ambiguity (the shift
    against the prior is then the disagreement between the two views)."""
    m = cfg["tray"]["model"]
    rows, cols = int(m["rows"]), int(m["cols"])
    p, wall, rim = float(m["pitch_m"]) * 1000, float(m["wall_m"]) * 1000, float(m["rim_m"]) * 1000
    if prior is None and z_top is None and z_hint is not None:
        # fast path: the tray top near where it was last time (the dense depth there, within 3 cm); a clear fit there is
        # taken, else the full search below (2026-11-04: the full height scan took ~6 s of every locate)
        ok_ = np.isfinite(frame.xyz).all(-1)
        zd = apply_T(T_BC, frame.xyz[ok_])[:, 2]
        near = zd[np.abs(zd - z_hint) < 0.03]
        if len(near) >= 2000:
            h_, e_ = np.histogram(near, bins=np.arange(z_hint - 0.03, z_hint + 0.0301, 0.002))
            zc = float(np.median(near[np.abs(near - (e_[int(np.argmax(h_))] + 0.001)) < 0.006]))
            g = locate_tray(frame, T_BC, cfg, z_top=zc)
            if g.ok and g.score >= fast_min_score:
                g.extra = {**(g.extra or {}), "z_candidates": [round(zc, 4)], "fast_path": True}
                return g
    if prior is None and z_top is None:
        cands = z_scan_candidates(frame, T_BC, cfg)
        if len(cands) > 1:  # fit at each; a passing fit beats a failing one, then the better match
            fits = [locate_tray(frame, T_BC, cfg, z_top=zc) for zc in cands]
            best = max(fits, key=lambda g: (g.ok, g.score))
            best.extra = {**(best.extra or {}), "z_candidates": [round(c, 4) for c in cands],
                          "candidate_fits": [(round(c, 4), g.ok, round(float(g.score), 3)) for c, g in zip(cands, fits)]}
            return best
    z = z_top if z_top is not None else tray_top_z(frame, T_BC, *m["top_z_search_m"])
    if z is None:
        return TrayGrid(False, "no tray top plane in the dense depth")
    xr, yr = m["search_x_m"], m["search_y_m"]
    if prior is not None:
        z = float(np.mean([c[2] for c in prior.values()]))  # same plane as the prior: a pure xy comparison
    I, valid, xs, ys = top_view(frame, T_BC, z, xr, yr)
    I = cv2.GaussianBlur(I, (0, 0), 1.0)
    t0 = _template(rows, cols, p, wall, rim)
    want = None
    if prior is not None:
        A1, Ae = np.array(prior["A1"][:2]), np.array(prior[f"A{cols}"][:2])
        if abs(Ae[0] - A1[0]) > abs(Ae[1] - A1[1]):
            raise NotImplementedError("prior search: only for a tray with its long side across the robot's reach")
        yaw0 = float(np.degrees(np.arctan2(*(A1 - Ae)[::-1]))) - 90.0
        angles = np.arange(-yaw0 - 1.0, -yaw0 + 1.0 + 1e-6, 0.1)
        win = window_m * 1000.0

        def want(M):  # template location that puts template cell (row 0, last column) on the prior A1
            tc = M @ np.array([rim + (cols - 0.5) * p, rim + 0.5 * p, 1.0])
            return np.array([(A1[1] - ys[0]) * 1000.0 - tc[0], (A1[0] - xs[0]) * 1000.0 - tc[1]])
    else:
        # any orientation (the grid looks the same turned by 180 deg): 2 deg, then 0.5 deg, then 0.1 deg
        angles = np.arange(-90.0, 90.0, 2.0)
    best = None
    if prior is None:
        # the 2 deg sweep on a half-size picture (16x less work: it took most of a locate's 13 s); only its best
        # angle is kept, the finer steps below are at full size
        I2 = cv2.resize(I, (I.shape[1] // 2, I.shape[0] // 2), interpolation=cv2.INTER_AREA)
        t2 = cv2.resize(t0, (t0.shape[1] // 2, t0.shape[0] // 2), interpolation=cv2.INTER_AREA)
        coarse = max((_match(I2, t2, ang)[0], ang) for ang in angles)
        best = (-np.inf, coarse[1], None, None, None)
    else:
        for ang in angles:
            v, M, loc, mk = _match(I, t0, ang, (want, win))
            if best is None or v > best[0]:
                best = (v, ang, M, loc, mk)
    if prior is None:
        for step, half in ((0.5, 2.0), (0.1, 0.5)):
            for ang in np.arange(best[1] - half, best[1] + half + 1e-6, step):
                v, M, loc, mk = _match(I, t0, ang)
                if v > best[0]:
                    best = (v, ang, M, loc, mk)
    v, ang, M, loc, mk = best
    # how much of the fitted tray lies outside the picture
    ys_i, xs_i = np.nonzero(mk)
    yy = np.clip(ys_i + int(round(loc[1])), 0, I.shape[0] - 1)
    xx = np.clip(xs_i + int(round(loc[0])), 0, I.shape[1] - 1)
    outside = float(1.0 - valid[yy, xx].mean())
    cells = {}
    centres = {}
    for r in range(rows):
        for c in range(cols):
            tc = M @ np.array([rim + (c + 0.5) * p, rim + (r + 0.5) * p, 1.0]) + np.array(loc)
            centres[(r, c)] = (float(xs[0] + tc[1] / 1000.0), float(ys[0] + tc[0] / 1000.0))
    # names: row A nearest the robot (smallest x), column 1 on the robot's left (largest y)
    # names: along each tray direction, from near the robot (small x) if that direction runs mostly along x,
    # else from the robot's left (large y)
    def order(idx, centre_of):
        pts = np.array([centre_of(j) for j in idx])
        span = pts.max(0) - pts.min(0)
        return sorted(idx, key=(lambda j: centre_of(j)[0]) if span[0] >= span[1] else (lambda j: -centre_of(j)[1]))

    rs = order(range(rows), lambda r: np.mean([centres[(r, c)] for c in range(cols)], axis=0))
    cs = order(range(cols), lambda c: np.mean([centres[(r, c)] for r in range(rows)], axis=0))
    for i, r in enumerate(rs):
        for j, c in enumerate(cs):
            x, y = centres[(r, c)]
            cells[f"{chr(ord('A') + i)}{j + 1}"] = [round(x, 4), round(y, 4), round(z, 4)]
    # every fitted cell must stand on the tray: the median dense height within 20 mm of its centre (cell bottom,
    # plug or plant) is above the table. A grid shifted onto the table (dark, cell-like in the image, e.g. when
    # the tray is cut off by the picture edge) has its cells at table height. Walls are no use: seen from
    # above, the black wall tops get almost no depth
    ok = np.isfinite(frame.xyz).all(-1)
    Pd = apply_T(T_BC, frame.xyz[ok])
    from scipy.spatial import cKDTree

    hz, ez = np.histogram(Pd[:, 2], bins=np.arange(-0.10, z - 0.03, 0.004))
    table_z = float(ez[int(np.argmax(hz))] + 0.002) if hz.sum() else z - 0.06
    tree = cKDTree(Pd[:, :2])
    # the plain (neither plant nor peat) points: on the tray these are the walls and cell bottoms, never above its
    # top; a grid shifted onto some higher dark object (a cloth, a box) has them well above the fitted top
    from .stemline import _mask, substrate_mask

    plain = (~_mask(frame.rgb, cfg["stemline"]) & ~substrate_mask(frame.rgb, cfg["stemline"]))[ok]

    def check_cells(cl):
        seen, off, above = 0, [], []
        for k, (x, y, _) in cl.items():
            idx = tree.query_ball_point([x, y], 0.02)
            if len(idx) < 20:
                continue
            seen += 1
            # a real cell has walls around it: even a tray standing right on the table (cell bottoms 1-2 cm above
            # it, drain holes showing the table) has its 90th-percentile height well above the table
            if np.percentile(Pd[idx, 2], 90) < table_z + 0.015:
                off.append(k)
            idx = np.array(idx)
            zp = Pd[idx[plain[idx]], 2]
            if len(zp) >= 20 and np.median(zp) > z + 0.04:  # raised plugs stand up to ~3 cm above the top
                above.append(k)
        return seen, off, above

    col = np.array(cells["A2"][:2]) - np.array(cells["A1"][:2])
    row = np.array(cells["B1"][:2]) - np.array(cells["A1"][:2])

    seen, off, above = check_cells(cells)
    shifted = None
    if prior is None and (len(off) >= 2 or seen < len(cells)):
        # the cell pattern repeats: the match can be whole cells off along a line (seen with the tray's far end
        # hanging over the table). Whole-cell shifts are tried; the one with no cell off the tray wins. Also when
        # cells have no depth: one cell along the long side off (2026-11-02, a person beside the tray, so no
        # outline cross-check) put the last row just past the tray end, where the rim within 20 mm hid that it
        # stands on nothing; 68 of 72 cells were seen, the right fit sees all
        best = (len(off), len(above), -seen, 0), (0, 0), (seen, off, above), cells
        for a in range(-3, 4):
            for b in range(-2, 3):
                if a == 0 and b == 0:
                    continue
                d = a * col + b * row
                cl = {k: [round(float(v[0] + d[0]), 4), round(float(v[1] + d[1]), 4), v[2]] for k, v in cells.items()}
                r = check_cells(cl)
                key = (len(r[1]), len(r[2]), -r[0], abs(a) + abs(b))
                if key < best[0]:
                    best = key, (a, b), r, cl
        if best[1] != (0, 0):
            shifted = best[1]
            seen, off, above = best[2]
            cells = best[3]
    a1_, a12 = np.array(cells["A1"][:2]), np.array(cells[f"A{cols}"][:2])
    yaw = float(np.degrees(np.arctan2(*(a12 - a1_)[::-1])))  # direction column 1 -> last (long side), about base z
    # tray top from the dense depth on the walls (midpoints between neighbouring cells), not the height-peak
    # guess: seen from above, the plug tops (1-2 cm higher) can form the higher peak
    mids = []
    for k_, (x_, y_, _) in cells.items():
        c_ = int(k_[1:])
        for nb in (f"{k_[0]}{c_ + 1}", f"{chr(ord(k_[0]) + 1)}{c_}"):
            if nb in cells:
                mids.append(0.5 * (np.array([x_, y_]) + np.array(cells[nb][:2])))
    okw = np.isfinite(frame.xyz).all(-1)
    Pw = apply_T(T_BC, frame.xyz[okw])
    from scipy.spatial import cKDTree as _T

    idx = sorted({i for lst in _T(Pw[:, :2]).query_ball_point(np.array(mids), 0.004) for i in lst})
    zw = Pw[idx, 2]
    zw = zw[np.abs(zw - z) < 0.03]
    if len(zw) > 200:
        z_wall = float(np.median(zw))
        if abs(z_wall - z) > 0.004:
            cells = {k_: [c[0], c[1], round(z_wall, 4)] for k_, c in cells.items()}
            z = z_wall
    g = TrayGrid(True, "ok", cells, z, round(yaw, 2), v, outside)
    g.extra = {"cells_at_table_height": off, "cells_above_tray_top": above, "cells_seen": seen, "table_z": round(table_z, 4),
               "shifted_cells": shifted}
    if seen < 0.5 * len(cells):
        g.ok, g.status = False, f"only {seen} of {len(cells)} cells have depth: the tray is not (fully) in the picture"
    elif len(above) > 0.08 * seen:
        g.ok, g.status = False, f"{len(above)} fitted cells lie on something above the tray top (e.g. {above[:4]}): grid shifted off the tray, or wrong tray-top height"
    elif len(off) > 0.10 * seen or len(off) >= 3:
        g.ok, g.status = False, f"{len(off)} fitted cells lie at table height (e.g. {off[:4]}): grid shifted off the tray, tray cut off?"
    elif prior is None and outside > float(m.get("max_outside", 0.01)):
        g.ok, g.status = False, f"tray not fully in the picture ({outside * 100:.0f}% of it outside): the grid may be off by whole cells"
    elif v < float(m.get("min_score", 0.5)):
        g.ok, g.status = False, f"weak grid match (score {v:.2f})"
    return g


def draw_cells(frame: Frame, T_BC: np.ndarray, cells: dict) -> np.ndarray:
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    T_CB = inv_T(T_BC)
    for k, p in cells.items():
        uv = frame.intr.project(apply_T(T_CB, np.array(p))).astype(int)
        cv2.drawMarker(img, tuple(int(a) for a in uv), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        if k[0] in "AF" or k[1:] in ("1",):
            cv2.putText(img, k, tuple(int(a) + 5 for a in uv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def occupied_cells(frame: Frame, T_BC: np.ndarray, cells: dict, z_top: float, cfg: dict,
                   plug_dz: float = 0.013, min_peat: float = 0.10) -> dict:
    """Cells holding a plug: peat colour on a ring 9-15 mm around the cell centre at plug-top height. The plug's
    rim usually shows beside the leaves; an empty cell is black plastic there, also under a neighbour's
    overhanging leaves (2026-10-22, top view: the 12 plugs 0.06-0.76, the 60 empty cells 0.00). Dense depth
    was tried first and is unreliable under leaves. {cell: peat fraction}.
    min_peat 0.10 since 2026-11-04 (0.03 before): seen at an angle (60 deg overview) the side of a plug covers the ring
    of the empty cell in front of it (0.03-0.08: B6-B8; the arm went for the empty B8). A plug hidden under leaves
    can read lower and is then found once the plants in front of it are gone: missed for now is safer than an
    empty cell taken for a plug."""
    from .stemline import substrate_mask

    soil = substrate_mask(frame.rgb, cfg["stemline"])
    T_CB = inv_T(T_BC)
    H, W = soil.shape
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    out = {}
    for k, (x, y, _) in cells.items():
        pts = np.array([[x + r * np.cos(a), y + r * np.sin(a), z_top + plug_dz] for r in (0.009, 0.012, 0.015) for a in ang])
        uv = np.round(frame.intr.project(apply_T(T_CB, pts))).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.mean() < 0.9:
            continue
        frac = float(soil[uv[inside, 1], uv[inside, 0]].mean())
        if frac >= min_peat:
            out[k] = round(frac, 2)
    return out


def locate_tray_outline(frame: Frame, T_BC: np.ndarray, rows: int, cols: int, cfg: dict, pitch: float | None = None,
                        res: float = 0.002, min_step_m: float = 0.02, top_z: float | None = None) -> dict:
    """Tray from its outline only (works when every cell holds a plant and no cell or plug is visible): the
    dense points higher than the surface the tray stands on, without plant-green ones, form the tray's
    footprint; its minimum-area rectangle gives the corners, size and yaw; the cells divide the rectangle into
    rows x cols around the rectangle's centre. With the tray type's pitch the cells are placed at that pitch
    from the centre (2026-10-22, 72-cell tray: within 2 mm median / 4 mm max of the grid-template fit); without
    it the rectangle is divided evenly, rim included (median 7 mm, max 12 mm off). Returns cells (named like
    locate_tray), corners, size, the rim height."""
    import cv2 as _cv2

    from .stemline import _mask

    ok = np.isfinite(frame.xyz).all(-1)
    green = _mask(frame.rgb, cfg["stemline"])
    P = apply_T(T_BC, frame.xyz[ok])
    G = green[ok]
    # the surface under the tray: the most frequent height below the highest 30 %
    z = P[:, 2]
    h, e = np.histogram(z[z < np.percentile(z, 70)], bins=np.arange(z.min(), np.percentile(z, 70) + 0.004, 0.004))
    support = float(e[int(np.argmax(h))] + 0.002)
    if top_z is not None:
        # the tray top known (the grid fit): only the points at its top face (wall tops, rims, plugs), so a stand the
        # tray is raised on does not count as tray (2026-11-04: the stand made the outline 550 x 324 mm, the check
        # was skipped and a grid one row off went through)
        high = (z > top_z - 0.012) & (z < top_z + 0.10) & ~G
    else:
        high = (z > support + min_step_m) & ~G
    Q = P[high]
    if len(Q) < 2000:
        return {"ok": False, "status": "no raised region above the support surface"}
    # height map of the raised, non-green points; largest connected blob = the tray
    lo = Q[:, :2].min(0)
    ij = np.floor((Q[:, :2] - lo) / res).astype(int)
    Hm = np.zeros(ij.max(0) + 1, np.uint8)
    Hm[ij[:, 0], ij[:, 1]] = 255
    Hm = _cv2.morphologyEx(Hm, _cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = _cv2.connectedComponentsWithStats(Hm)
    if n < 2:
        return {"ok": False, "status": "no tray blob"}
    k = 1 + int(np.argmax(st[1:, _cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8)
    cnts, _ = _cv2.findContours(blob, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=_cv2.contourArea)
    (cx, cy), (w, hgt), ang = _cv2.minAreaRect(cnt.astype(np.float32))
    box = _cv2.boxPoints(((cx, cy), (w, hgt), ang))  # in height-map pixels (i = x index, j = y index)
    corners = np.array([lo + np.array([b[1], b[0]]) * res + res / 2 for b in box])  # points are (col=j, row=i)
    # long side -> columns
    e01, e12 = corners[1] - corners[0], corners[2] - corners[1]
    if np.linalg.norm(e01) >= np.linalg.norm(e12):
        o, u, v = corners[0], e01, e12
    else:
        o, u, v = corners[1], e12, -e01
        o = corners[1]
        v = corners[0] - corners[1]
    L, Wd = np.linalg.norm(u), np.linalg.norm(v)
    # tray top: the wall-top height peak above the support (not a percentile: plugs and plants stand higher)
    rim_z = top_z if top_z is not None else tray_top_z(frame, T_BC, support + min_step_m, support + 0.25)
    if rim_z is None:
        rim_z = float(np.percentile(Q[:, 2], 50))
    centres = {}
    mid = corners.mean(0)
    uu, vv = u / L, v / Wd
    for r in range(rows):
        for c in range(cols):
            if pitch:
                p = mid + uu * (c - (cols - 1) / 2) * pitch + vv * (r - (rows - 1) / 2) * pitch
            else:
                p = o + u * (c + 0.5) / cols + v * (r + 0.5) / rows
            centres[(r, c)] = (float(p[0]), float(p[1]))

    def order(idx, centre_of):
        pts = np.array([centre_of(j) for j in idx])
        span = pts.max(0) - pts.min(0)
        return sorted(idx, key=(lambda j: centre_of(j)[0]) if span[0] >= span[1] else (lambda j: -centre_of(j)[1]))

    rs = order(range(rows), lambda r: np.mean([centres[(r, c)] for c in range(cols)], axis=0))
    cs = order(range(cols), lambda c: np.mean([centres[(r, c)] for r in range(rows)], axis=0))
    cells = {}
    for i, r in enumerate(rs):
        for j, c in enumerate(cs):
            x, y = centres[(r, c)]
            cells[f"{chr(ord('A') + i)}{j + 1}"] = [round(x, 4), round(y, 4), round(rim_z, 4)]
    status, okk = "ok", True
    if pitch:
        dl, dw = L - cols * pitch, Wd - rows * pitch  # the rims add a little
        if not (-0.01 <= dl <= 0.045 and -0.01 <= dw <= 0.045):
            okk, status = False, (f"outline {L * 1000:.0f} x {Wd * 1000:.0f} mm does not fit a {rows} x {cols} tray at "
                                  f"{pitch * 1000:.1f} mm ({cols * pitch * 1000:.0f} x {rows * pitch * 1000:.0f} mm + rims): not the tray alone")
    # the whole outline must be inside the picture (a cut tray gives a too small or shifted rectangle)
    T_CB = inv_T(T_BC)
    H_, W_ = frame.rgb.shape[:2]
    uv = np.array([frame.intr.project(apply_T(T_CB, np.array([cx_, cy_, support + min_step_m]))) for cx_, cy_ in corners])
    if okk and ((uv[:, 0] < 15).any() or (uv[:, 1] < 15).any() or (uv[:, 0] > W_ - 15).any() or (uv[:, 1] > H_ - 15).any()):
        okk, status = False, "tray outline touches the picture border: not fully in view"
    return {"ok": okk, "status": status, "cells": cells, "corners": corners.tolist(), "size_m": [round(L, 4), round(Wd, 4)],
            "pitch_m": [round(L / cols, 4), round(Wd / rows, 4)], "yaw_deg": round(float(np.degrees(np.arctan2(u[1], u[0]))), 2),
            "support_z": round(support, 4), "rim_z": round(rim_z, 4), "blob_px": int(st[k, _cv2.CC_STAT_AREA])}
