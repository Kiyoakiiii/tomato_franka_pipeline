"""Lower-stem centre line by triangulating the rectified left/right images of the stereo camera.

Why: dense stereo, including the ZED neural modes, smooths a 2-3 mm stem into the background behind it
(measured 2026-10-12: depth along the stem stays at the plug/tray depth), while the high-confidence pixels
that survive are correct but sparse. A stem is, however, a clean thin line in both images. Here:

1. left image: green mask -> skeleton of the plant -> start at the skeleton end lowest in the world
   (the root, judged with the camera pose) -> walk up to the first branch (leaf stalk) or a length cap
   -> the stem band -> sub-pixel centre line (PCA of the band pixels);
2. right image (rectified: same rows): the grey values of the band searched along the rows with a
   disparity that is affine along the image line (true for any 3-D line), maximising NCC; then, as an
   independent check, the disparity of short row windows is measured on their own and compared;
3. per image row: disparity from the two centre lines -> 3-D point; PCA -> 3-D stem line;
4. the stem base: the lowest part of a stem is often pale and soiled (not green), so from the lowest
   green pixel the stem is followed further down as a ridge brighter than the substrate on both sides.
   The root is on that base pixel's ray, at the stem line's depth; the substrate height from the dense
   depth (large textured surfaces are fine) is an independent check, and the fallback when the base is
   hidden (then the line is extended down to it).

Lines that run nearly parallel to the stereo baseline in the image (i.e. horizontal in the image) cannot
be triangulated this way and are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .camera import Frame
from .transforms import apply_T, unit


def green_mask(rgb: np.ndarray, exg_threshold: float, min_brightness: float, hue_deg=(45.0, 150.0), min_saturation: float = 0.2) -> np.ndarray:
    """Plant green: excess-green index plus a hue/saturation gate (wood, orange and grey are excluded)."""
    f = rgb.astype(np.float32)
    s = f.sum(-1) + 1e-6
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / s
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    m = ((exg > exg_threshold) & (s > min_brightness) & (hue >= hue_deg[0]) & (hue <= hue_deg[1]) & (sat >= min_saturation)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m > 0


def _mask(rgb: np.ndarray, c: dict) -> np.ndarray:
    return green_mask(rgb, c["exg_threshold"], c["min_brightness"], c.get("hue_deg", (45.0, 150.0)), c.get("min_saturation", 0.2))


@dataclass
class StemLine2D:
    centre: np.ndarray  # (2,) a point on the sub-pixel centre line (u, v)
    direction: np.ndarray  # (2,) unit, from the root end towards the top
    root_px: np.ndarray
    top_px: np.ndarray
    band: np.ndarray  # (N, 2) int pixels (u, v) of the stem band
    width_px: float
    path: np.ndarray  # skeleton path (u, v), root first
    curvature_px: float  # RMS distance of the skeleton path from the fitted line
    depth_hint_m: float = 0.0  # rough distance of the plant from the dense depth (search window only)


@dataclass
class StemLineEstimate:
    ok: bool
    status: str
    left: StemLine2D | None = None
    right_centre: np.ndarray | None = None
    right_direction: np.ndarray | None = None
    match_score: float = 0.0
    points_C: np.ndarray | None = None  # triangulated centre-line points (camera frame)
    root_B: np.ndarray | None = None  # stem line at the substrate (or lowest visible point if no substrate)
    top_B: np.ndarray | None = None
    dir_B: np.ndarray | None = None  # unit, upward
    grasp_B: np.ndarray | None = None
    tilt_deg: float | None = None
    diameter_m: float | None = None
    soil_z: float | None = None
    base_px: np.ndarray | None = None  # where the stem enters the substrate in the left image (pale base included)
    root_source: str = ""  # "base pixel" (seen) or "extended to the substrate" (base hidden)
    root_vs_soil_mm: float | None = None  # root (stem stereo depth) vs base ray on the dense-depth substrate plane
    disparity_resid_px: float | None = None  # RMS of independently matched row windows about the fitted disparity
    depth_resid_mm: float | None = None  # the same as depth along the camera ray
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        r = lambda a, k=4: None if a is None else np.round(np.asarray(a, float), k).tolist()  # noqa: E731
        return {"ok": self.ok, "status": self.status, "root_B": r(self.root_B), "top_B": r(self.top_B), "dir_B": r(self.dir_B, 3),
                "grasp_B": r(self.grasp_B), "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 1),
                "diameter_mm": None if self.diameter_m is None else round(self.diameter_m * 1000, 1),
                "soil_z": None if self.soil_z is None else round(self.soil_z, 4),
                "root_source": self.root_source,
                "root_vs_soil_mm": None if self.root_vs_soil_mm is None else round(self.root_vs_soil_mm, 1),
                "disparity_resid_px": None if self.disparity_resid_px is None else round(self.disparity_resid_px, 2),
                "depth_resid_mm": None if self.depth_resid_mm is None else round(self.depth_resid_mm, 1),
                "match_score": round(self.match_score, 3), **self.extra}


# ------------------------------------------------------------------ left image: the stem in 2-D -----
_NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _up_in_image(T_BC: np.ndarray, frame: Frame, uv: np.ndarray, depth: float) -> np.ndarray:
    """Image direction of world +z at pixel uv (depth along the ray)."""
    i = frame.intr
    p_C = np.array([(uv[0] - i.cx) / i.fx, (uv[1] - i.cy) / i.fy, 1.0]) * depth
    q_C = apply_T(np.linalg.inv(T_BC), apply_T(T_BC, p_C) + [0, 0, 0.01])
    return unit(i.project(q_C) - uv)


def prune_skeleton(sk: np.ndarray, min_branch_px: float, rounds: int = 3) -> np.ndarray:
    """Remove side branches shorter than ``min_branch_px`` (end point -> junction); skeletons of a slightly
    ragged mask (e.g. where the stem meets the soil) otherwise branch every few pixels."""
    sk = sk.copy()
    H, W = sk.shape
    for _ in range(rounds):
        ys, xs = np.nonzero(sk)
        pix = set(zip(xs.tolist(), ys.tolist()))

        def nbrs(p):
            return [(p[0] + dx, p[1] + dy) for dy, dx in _NBR if (p[0] + dx, p[1] + dy) in pix]

        removed = False
        for e in [p for p in pix if len(nbrs(p)) == 1]:
            if e not in pix:
                continue
            branch, prev, cur = [e], None, e
            while True:
                nx = [p for p in nbrs(cur) if p != prev]
                if len(nx) != 1:
                    break
                prev, cur = cur, nx[0]
                if len(nbrs(cur)) > 2:  # reached a junction (kept)
                    break
                branch.append(cur)
                if len(branch) > min_branch_px:
                    break
            if len(nbrs(cur)) > 2 and len(branch) <= min_branch_px:
                for p in branch:
                    pix.discard(p)
                    sk[p[1], p[0]] = False
                removed = True
        if not removed:
            break
    return sk


def find_stem_2d(mask: np.ndarray, T_BC: np.ndarray, frame: Frame, cfg: dict, approx_depth: float,
                 seed_px: np.ndarray | None = None, root_px: np.ndarray | None = None) -> tuple[StemLine2D | None, str]:
    """root_px: start the stem at the skeleton end nearest this pixel (several plants whose leaves touch form
    one green region; their lowest end is then not this plant's root)."""
    c = cfg["stemline"]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, "no green pixels"
    if root_px is not None:
        seed_px = root_px
    if seed_px is not None:
        ys, xs = np.nonzero(lab)
        k = int(lab[ys, xs][np.argmin((xs - seed_px[0]) ** 2 + (ys - seed_px[1]) ** 2)])
    else:  # the start pose looks at the plant: the sizeable green region nearest the image centre
        big = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= c["min_plant_area_px"]]
        if not big:
            return None, "no sizeable green region"
        ctr = np.array([mask.shape[1] / 2, mask.shape[0] / 2])
        k = min(big, key=lambda j: float(np.linalg.norm(cent[j] - ctr)))
    comp = lab == k
    if comp.sum() < 50:
        return None, "green region too small"
    # rough distance of this plant: dense depth on/around it (the plug and leaves are large, textured surfaces)
    x, y, w, h = stats[k, :4]
    if root_px is not None:  # the region may span several plants: the plug at this root only
        r = 40
        x, y, w, h = int(root_px[0]) - r, int(root_px[1]) - r, 2 * r, 2 * r
        x, y = max(x, 0), max(y, 0)
    zz = frame.xyz[y:y + h, x:x + w, 2]
    zz = zz[np.isfinite(zz)]
    if len(zz) > 50:
        approx_depth = float(np.percentile(zz, 30))
    dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(comp)
    half0 = float(np.percentile(dt[sk], 25)) if sk.any() else 1.0  # thin parts (stem), not the leaf blades
    sk = prune_skeleton(sk, float(np.clip(2.5 * half0 + 2, 6.0, 20.0)))
    ys, xs = np.nonzero(sk)
    pix = set(zip(xs.tolist(), ys.tolist()))

    def nbrs(p):
        return [(p[0] + dx, p[1] + dy) for dy, dx in _NBR if (p[0] + dx, p[1] + dy) in pix]

    ends = [p for p in pix if len(nbrs(p)) == 1]
    if not ends:
        return None, "no skeleton end"
    if root_px is not None:  # an end near it, else any skeleton pixel (a ragged base forks into short ends)
        dd = lambda p: float(np.hypot(p[0] - root_px[0], p[1] - root_px[1]))  # noqa: E731
        root = min(ends, key=dd)
        if dd(root) > 6:
            root = min(pix, key=dd)
        if dd(root) > 12:
            return None, "no skeleton at the given root"
    else:
        up = _up_in_image(T_BC, frame, np.array(cent[k]), approx_depth)
        root = min(ends, key=lambda p: float(np.dot(p, up)))  # lowest in the world
    f = frame.intr.fx
    max_len = c["max_stem_length_m"] * f / approx_depth
    # longest skeleton path from the root: the main stem; short side twigs (ragged mask edges, cotyledons) are passed
    parent, dist, queue = {root: None}, {root: 0.0}, [root]
    while queue:
        nxt = []
        for p in queue:
            for q in nbrs(p):
                if q not in dist:
                    dist[q] = dist[p] + float(np.hypot(q[0] - p[0], q[1] - p[1]))
                    parent[q] = p
                    nxt.append(q)
        queue = nxt
    far = max(dist, key=dist.get)
    chain = [far]
    while parent[chain[-1]] is not None:
        chain.append(parent[chain[-1]])
    chain.reverse()
    path, length = [root], 0.0
    for prev, cur in zip(chain[:-1], chain[1:]):
        length += float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
        if length > max_len:
            break
        path.append(cur)
    widths = np.array([float(dt[p[1], p[0]]) for p in path])
    ref = float(np.median(widths))  # stem half width along the lower stem
    # drop a thin twig at the root end (ragged mask where the stem meets the soil)
    k0 = int(np.argmax(widths >= 0.6 * ref))
    path, widths = path[k0:], widths[k0:]
    # cut where the path enters a leaf blade: the width stays well above the stem width for several pixels
    run = 6
    for i in range(10, len(path) - run):
        if np.all(widths[i:i + run] > 1.8 * ref + 1.0):
            path = path[:i]
            break
    length = float(np.sum(np.hypot(*np.diff(np.array(path, float), axis=0).T))) if len(path) > 1 else 0.0
    min_len = c["min_stem_length_m"] * f / approx_depth
    if length < min_len:
        return None, f"stem segment too short ({length:.0f} px < {min_len:.0f} px before a leaf blade)"
    full = np.array(path, float)
    trim = max(2, int(0.08 * len(full)))  # the root end touches soil, the top end widens into the branch
    P = full[trim:-trim] if len(full) > 2 * trim + 5 else full
    half = float(np.median(dt[P[:, 1].astype(int), P[:, 0].astype(int)]))
    tree = cKDTree(P)
    u0, v0 = (P.min(0) - half - 3).astype(int)
    u1, v1 = (P.max(0) + half + 4).astype(int)
    vv, uu = np.mgrid[max(v0, 0):min(v1, mask.shape[0]), max(u0, 0):min(u1, mask.shape[1])]
    cand = np.column_stack([uu.ravel(), vv.ravel()])
    cand = cand[comp[cand[:, 1], cand[:, 0]]]
    dist, idx = tree.query(cand)
    # within the stem's half width of the path, and not beyond its ends (no leaf pixels around the branch)
    band = cand[(dist <= half + 1.0) & (idx > 0) & (idx < len(P) - 1)]
    if len(band) < 20:
        return None, "stem band too small"
    centre = band.mean(0)
    w, V = np.linalg.eigh(np.cov((band - centre).T))
    d = V[:, -1]
    if np.dot(d, P[-1] - P[0]) < 0:
        d = -d
    rel = P - centre
    curv = float(np.sqrt(np.mean((rel[:, 0] * d[1] - rel[:, 1] * d[0]) ** 2)))
    if 2 * half * approx_depth / f > float(cfg["grasp"]["max_stem_diameter_m"]):
        return None, f"not a thin stem ({2 * half:.0f} px wide)"
    return StemLine2D(centre, d, full[0], full[-1], band.astype(int), 2 * half, full, curv, approx_depth), ""


# ------------------------------------------------------------------ right image: match and refit ---
def _line_u_at(centre, d, v):
    return centre[0] + (v - centre[1]) * d[0] / d[1]


def match_right(left: StemLine2D, frame: Frame, cfg: dict, approx_depth: float | None):
    """Find the stem band in the right image: disparity d(v) = d0 + d1 (v - v_c) (affine along the image line,
    exact for a 3-D line) maximising the normalised cross-correlation of grey values over the stem band and
    its flanks. Grey, not colour: a stem can look green in one camera and pale (specular) in the other.
    Coarse integer search, then a sub-pixel refinement. Returns (right centre, right direction, NCC, model)
    with model = (d0, d1, v_c, grey left, grey right) for the residual check, or None."""
    c = cfg["stemline"]
    f, B = frame.intr.fx, float(frame.meta["baseline_m"])
    gL = cv2.GaussianBlur(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
    gR = cv2.GaussianBlur(cv2.cvtColor(frame.rgb_right, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
    H, W = gR.shape
    S = left.band.astype(float)
    nrm = np.array([-left.direction[1], left.direction[0]])
    # the stem band only: the background beside a stem differs between the two views (parallax, leaves)
    P = S[(S[:, 0] >= 0) & (S[:, 0] < W - 1) & (S[:, 1] >= 0) & (S[:, 1] < H)]
    del nrm
    a = gL[P[:, 1].astype(int), np.round(P[:, 0]).astype(int)]
    a = (a - a.mean()) / (a.std() + 1e-6)
    rows = P[:, 1].astype(int)
    vc = left.centre[1]
    zlo, zhi = c["depth_range_m"]
    if approx_depth is not None and np.isfinite(approx_depth):
        zlo, zhi = max(zlo, approx_depth - 0.12), min(zhi, approx_depth + 0.12)

    def ncc(d0s, d1, interp=False):
        u = P[None, :, 0] - (np.asarray(d0s, float)[:, None] + d1 * (P[None, :, 1] - vc))
        ok = (u >= 0) & (u < W - 1)
        u = np.clip(u, 0, W - 1.001)
        if interp:
            u0 = np.floor(u).astype(int)
            t = u - u0
            b = gR[rows[None, :], u0] * (1 - t) + gR[rows[None, :], u0 + 1] * t
        else:
            b = gR[rows[None, :], np.round(u).astype(int)]
        b = np.where(ok, b, np.nan)
        b = (b - np.nanmean(b, 1, keepdims=True)) / (np.nanstd(b, 1, keepdims=True) + 1e-6)
        return np.nanmean(a[None, :] * b, 1)

    d0s = np.arange(np.floor(f * B / zhi), np.ceil(f * B / zlo) + 1, 1.0)
    off = float(frame.meta.get("disparity_offset_px", 0.0))
    d_root = _root_disparity(frame, left, f * B, off)  # the root sits on the substrate: dense depth there is reliable
    v_root = left.root_px[1]
    best = (-np.inf, 0.0, 0.0)
    for d1 in np.arange(-1.2, 1.2001, 0.04):
        cand = d0s
        if d_root is not None:  # d(v_root) = d0 + d1 (v_root - vc) within the root window
            cand = d0s[np.abs(d0s + d1 * (v_root - vc) - d_root) <= c["root_disparity_window_px"]]
            if len(cand) == 0:
                continue
        sc = ncc(cand, d1)
        j = int(np.nanargmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), float(cand[j]), float(d1))
    if not np.isfinite(best[0]):
        return None, None, 0.0, None
    for d1 in best[2] + np.arange(-0.06, 0.0601, 0.005):  # sub-pixel refinement
        d0f = best[1] + np.arange(-1.5, 1.501, 0.1)
        sc = ncc(d0f, d1, interp=True)
        j = int(np.nanargmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), float(d0f[j]), float(d1))
    sc, d0, d1 = best
    if sc < c["min_match_score"]:
        return None, None, sc, None
    # right centre line = left centre line shifted by d(v)
    v0, v1 = S[:, 1].min(), S[:, 1].max()
    pR0 = np.array([_line_u_at(left.centre, left.direction, v0) - (d0 + d1 * (v0 - vc)), v0])
    pR1 = np.array([_line_u_at(left.centre, left.direction, v1) - (d0 + d1 * (v1 - vc)), v1])
    dR = unit(pR1 - pR0)
    if np.dot(dR, left.direction) < 0:
        dR = -dR
    return (pR0 + pR1) / 2, dR, sc, (d0, d1, vc, gL, gR)


def window_disparity_residuals(left: StemLine2D, model, rows_per_window: int = 9, step: int = 4, max_shift_px: float = 3.0,
                               min_score: float = 0.5) -> np.ndarray:
    """Disparity of short row windows of the stem, each matched on its own (strip across the stem including
    its edges, shifts within +-max_shift_px of the fitted model), as residuals to the fitted affine model.
    A real stem gives residuals of a fraction of a pixel; a wrong match (repeated texture, a leaf edge) or a
    strongly curved stem does not."""
    d0, d1, vc, gL, gR = model
    H, W = gL.shape
    half = left.width_px / 2 + 4
    v_lo, v_hi = int(left.band[:, 1].min()), int(left.band[:, 1].max())
    shifts = np.arange(-max_shift_px, max_shift_px + 1e-9, 0.25)
    out = []
    for v0 in range(v_lo, v_hi - rows_per_window + 2, step):
        vs = np.arange(v0, v0 + rows_per_window)
        us = [_line_u_at(left.centre, left.direction, v) + np.arange(-np.ceil(half), np.ceil(half) + 1) for v in vs]
        if min(u.min() for u in us) < 0 or max(u.max() for u in us) > W - 2:
            continue
        a = np.concatenate([gL[v, np.round(u).astype(int)] for v, u in zip(vs, us)])
        a = (a - a.mean()) / (a.std() + 1e-6)
        sc = []
        for sh in shifts:
            b = []
            for v, u in zip(vs, us):
                r = _sample_row(gR, v, u - (d0 + d1 * (v - vc)) - sh)
                if r is None:
                    break
                b.append(r)
            if len(b) < len(vs):
                sc.append(-np.inf)
                continue
            b = np.concatenate(b)
            b = (b - b.mean()) / (b.std() + 1e-6)
            sc.append(float(np.mean(a * b)))
        sc = np.array(sc)
        j = int(np.argmax(sc))
        if sc[j] < min_score or j == 0 or j == len(sc) - 1:
            continue
        den = sc[j - 1] - 2 * sc[j] + sc[j + 1]
        out.append(shifts[j] + (0.5 * (sc[j - 1] - sc[j + 1]) / den * 0.25 if den < 0 else 0.0))
    return np.array(out)


def trace_base(gray: np.ndarray, left: StemLine2D, max_px: float, min_contrast: float) -> np.ndarray:
    """From the lowest green pixel follow the stem further down as a ridge brighter than both sides (a pale or
    soiled stem base against the darker substrate). Returns the last pixel on the ridge (= root_px if none)."""
    H, W = gray.shape
    w = left.width_px
    p, d = left.root_px.astype(float), -left.direction.astype(float)

    def contrast(c, d):
        n = np.array([-d[1], d[0]])

        def mean_at(offs):
            q = c[None, None, :] + np.arange(-2, 3)[:, None, None] * d + np.asarray(offs)[None, :, None] * n
            u, v = np.round(q[..., 0]).astype(int), np.round(q[..., 1]).astype(int)
            if u.min() < 0 or v.min() < 0 or u.max() >= W or v.max() >= H:
                return np.nan
            return float(gray[v, u].mean())

        centre = mean_at(np.linspace(-w / 4, w / 4, 3))
        sides = [mean_at(sg * np.array([w / 2 + 2, w / 2 + 3.5, w / 2 + 5])) for sg in (-1, 1)]
        return centre - max(sides)

    d_start = d.copy()
    cos_max = np.cos(np.radians(35.0))  # the base may curve, but the trace must keep going down the stem
    last, misses, travelled, first = p.copy(), 0, 0.0, True
    while travelled < max_px and misses < 2:
        n = np.array([-d[1], d[0]])
        # first step: re-centre across the whole stem width (the skeleton end can sit on a ragged edge)
        offs = np.arange(-w / 2, w / 2 + 0.01, 0.5) if first else np.arange(-2, 2.01, 0.5)
        cands = [p + d + o * n for o in offs]
        cons = [contrast(c, d) for c in cands]
        if not np.isfinite(cons).any():
            break
        j = int(np.nanargmax(cons))
        c = cands[j]
        if not first:
            d_new = unit(0.7 * d + 0.3 * unit(c - p))
            if d_new @ d_start >= cos_max:
                d = d_new
        travelled += float((c - p) @ d_start) if not first else 1.0
        p, first = c, False
        if cons[j] >= min_contrast:
            last, misses = p.copy(), 0
        else:
            misses += 1
    return last


def _root_disparity(frame: Frame, left: StemLine2D, fB: float, off: float) -> float | None:
    """Measured (uncorrected) disparity expected at the root pixel: dense depth of the substrate just around it."""
    u0, v0 = int(left.root_px[0]), int(left.root_px[1])
    r = int(max(6, 1.5 * left.width_px))
    z = frame.xyz[max(v0 - r, 0):v0 + r + 1, max(u0 - r, 0):u0 + r + 1, 2]
    z = z[np.isfinite(z)]
    if len(z) < 15:
        return None
    return fB / float(np.median(z)) + off


def _sample_row(img: np.ndarray, v: float, us: np.ndarray) -> np.ndarray | None:
    """Linear interpolation of one image row at sub-pixel columns."""
    W = img.shape[1]
    if us.min() < 0 or us.max() > W - 2:
        return None
    u0 = np.floor(us).astype(int)
    t = us - u0
    row = img[int(v)]
    return row[u0] * (1 - t) + row[u0 + 1] * t


# ------------------------------------------------------------------ 3-D ----------------------------
def triangulate_lines(left: StemLine2D, cR: np.ndarray, dR: np.ndarray, frame: Frame) -> np.ndarray:
    """3-D points (camera frame) along the stem, one per image row of the left band."""
    i = frame.intr
    B = float(frame.meta["baseline_m"])
    v = np.arange(int(left.band[:, 1].min()), int(left.band[:, 1].max()) + 1, dtype=float)
    uL = _line_u_at(left.centre, left.direction, v)
    uR = _line_u_at(cR, dR, v)
    disp = uL - uR - float(frame.meta.get("disparity_offset_px", 0.0))  # same stereo bias correction as the dense depth
    ok = disp > 1.0
    Z = i.fx * B / disp[ok]
    return np.column_stack([(uL[ok] - i.cx) * Z / i.fx, (v[ok] - i.cy) * Z / i.fy, Z])


def _soil_z(frame: Frame, T_BC: np.ndarray, root_px: np.ndarray, gmask: np.ndarray, r_in_px: float, r_out_px: float) -> float | None:
    """Substrate height right at the root from the dense depth: non-green pixels in a ring around the root
    pixel (the plug surface is not flat; its rim and the cell walls are lower)."""
    H, W = gmask.shape
    u0, v0 = int(root_px[0]), int(root_px[1])
    r = int(r_out_px) + 1
    vv, uu = np.mgrid[max(v0 - r, 0):min(v0 + r, H), max(u0 - r, 0):min(u0 + r, W)]
    d2 = (uu - u0) ** 2 + (vv - v0) ** 2
    m = (d2 >= r_in_px ** 2) & (d2 <= r_out_px ** 2) & ~gmask[vv, uu]
    X = frame.xyz[vv[m], uu[m]]
    X = X[np.isfinite(X).all(1)]
    if len(X) < 20:
        return None
    return float(np.median(apply_T(T_BC, X)[:, 2]))


def estimate_stem_line(frame: Frame, T_BC: np.ndarray, cfg: dict, seed_px: np.ndarray | None = None,
                       root_px: np.ndarray | None = None, gmask: np.ndarray | None = None) -> StemLineEstimate:
    c = cfg["stemline"]
    if frame.rgb_right is None or "baseline_m" not in frame.meta:
        return StemLineEstimate(False, "frame has no right image")
    i = frame.intr
    g = _mask(frame.rgb, c) if gmask is None else gmask
    gz = frame.xyz[..., 2][g]
    gz = gz[np.isfinite(gz)]
    approx = float(np.median(gz)) if len(gz) > 20 else float(np.mean(c["depth_range_m"]))
    left, why = find_stem_2d(g, T_BC, frame, cfg, approx, seed_px, root_px)
    if left is None:
        return StemLineEstimate(False, why)
    est = StemLineEstimate(False, "", left=left)
    if abs(left.direction[1]) < c["min_vertical_fraction"]:
        est.status = "stem nearly horizontal in the image (parallel to the stereo baseline); roll the camera"
        return est
    approx = left.depth_hint_m or approx
    cR, dR, sc, model = match_right(left, frame, cfg, approx)
    est.match_score = sc
    if cR is None:
        est.status = f"stem not found in the right image (score {sc:.2f})"
        return est
    est.right_centre, est.right_direction = cR, dR
    P_C = triangulate_lines(left, cR, dR, frame)
    if len(P_C) < 5:
        est.status = "triangulation failed"
        return est
    est.points_C = P_C
    Zm = float(np.median(P_C[:, 2]))
    fB = i.fx * float(frame.meta["baseline_m"])
    res = window_disparity_residuals(left, model)
    if len(res) >= 3:
        est.disparity_resid_px = float(np.sqrt(np.mean(res ** 2)))
        est.depth_resid_mm = est.disparity_resid_px * Zm ** 2 / fB * 1000
    P_B = apply_T(T_BC, P_C)
    # the path must start at the lower end in the world (a stem top touching another plant's plug is not a root);
    # P_C is ordered by image row
    k = max(2, len(P_B) // 5)
    z_first, z_last = float(np.median(P_B[:k, 2])), float(np.median(P_B[-k:, 2]))
    z_root, z_top = (z_first, z_last) if left.root_px[1] <= left.top_px[1] else (z_last, z_first)
    if z_root > z_top:
        est.status = f"path starts at the upper end of the stem ({(z_root - z_top) * 1000:.0f} mm higher)"
        return est
    m = P_B.mean(0)
    _, V = np.linalg.eigh(np.cov((P_B - m).T))
    d = V[:, -1] if V[2, -1] >= 0 else -V[:, -1]
    t = (P_B - m) @ d
    hi = m + d * t.max()
    est.dir_B, est.top_B = d, hi
    est.tilt_deg = float(np.degrees(np.arccos(np.clip(d[2], -1, 1))))
    est.diameter_m = left.width_px * Zm / i.fx
    px_per_m = i.fx / Zm
    gray = cv2.GaussianBlur(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
    base = trace_base(gray, left, c["max_root_extension_m"] * px_per_m, float(c.get("base_min_contrast", 20.0)))
    est.base_px = base
    est.soil_z = _soil_z(frame, T_BC, base, g, 0.003 * px_per_m, 0.010 * px_per_m)
    # root on the base pixel's ray, at the depth where that ray passes the stem line (closest approach)
    o = T_BC[:3, 3]
    ray = unit(T_BC[:3, :3] @ np.array([(base[0] - i.cx) / i.fx, (base[1] - i.cy) / i.fy, 1.0]))
    w0 = o - m
    b_ = float(ray @ d)
    s_ray = (b_ * float(d @ w0) - float(ray @ w0)) / max(1 - b_ ** 2, 1e-9)
    root, est.root_source = o + s_ray * ray, "base pixel"
    if est.soil_z is not None:
        if d[2] > 0.2 and root[2] - est.soil_z > float(c.get("max_root_above_soil_m", 0.008)) \
                and root[2] - est.soil_z < c["max_root_extension_m"]:  # base hidden (leaf, cell wall): extend the line
            root, est.root_source = root + d * (est.soil_z - root[2]) / d[2], "extended to the substrate"
        else:  # independent check: the same ray on the substrate plane from the dense depth
            hit = o + ray * (est.soil_z - o[2]) / ray[2]
            est.root_vs_soil_mm = float(np.linalg.norm(hit - root)) * 1000
    est.root_B = root
    # grasp point: on the fitted (visible) stem line, grasp_offset_m above the root along the stem
    along = min(float(c["grasp_offset_m"]), float((hi[2] - root[2]) / max(d[2], 1e-6)))
    est.grasp_B = m + d * (float((root - m) @ d) + along)
    est.extra = {"visible_length_mm": round(float(t.max() - t.min()) * 1000, 1), "curvature_px": round(left.curvature_px, 2),
                 "width_px": round(left.width_px, 1), "n_rows": int(len(P_C)), "approx_depth_m": round(approx, 3),
                 "base_below_green_px": round(float(np.linalg.norm(base - left.root_px)), 1),
                 "root_off_line_mm": round(float(np.linalg.norm((root - m) - d * ((root - m) @ d))) * 1000, 1)}
    if est.disparity_resid_px is not None and est.disparity_resid_px > float(c.get("max_disparity_resid_px", 1.0)):
        est.status = f"right-image match inconsistent along the stem (row windows off by {est.disparity_resid_px:.1f} px RMS)"
        return est
    est.ok, est.status = True, "ok"
    return est


# ------------------------------------------------------------------ several plants -------------------
def substrate_mask(rgb: np.ndarray, c: dict) -> np.ndarray:
    """Peat / soil: brown (yellowish in Lab b) and not green; the black or grey tray is bluish (b < 128)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    m = ((lab[..., 2] >= int(c.get("substrate_lab_b_min", 131))) & (lab[..., 1] >= 124)
         & (lab[..., 0] > 25) & (lab[..., 0] < 140)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0


def root_candidates(frame: Frame, T_BC: np.ndarray, cfg: dict, gmask: np.ndarray | None = None) -> list[np.ndarray]:
    """Pixels where a stem may enter the substrate: points of the green skeleton near soil that are the lowest
    in the world within a small radius (local minima; skeleton ends at a ragged stem base fork, so ends alone
    miss stems). Leaves lying on a plug give candidates too; the stem and stereo checks reject them."""
    c = cfg["stemline"]
    g = _mask(frame.rgb, c) if gmask is None else gmask
    zz = frame.xyz[..., 2]
    approx = float(np.nanmedian(zz[g])) if g.any() else float(np.mean(c["depth_range_m"]))
    px_per_m = frame.intr.fx / approx
    # the pale stem base is not green: the green skeleton may end some mm above the soil
    r = int(round(float(c.get("root_soil_gap_m", 0.015)) * px_per_m))
    soil = cv2.dilate(substrate_mask(frame.rgb, c).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = st[1:, cv2.CC_STAT_AREA] >= int(c.get("min_seedling_area_px", 300))  # specks of green on the plugs
    sk = prune_skeleton(skeletonize(g & big[lab]), 8.0)
    vs, us = np.nonzero(sk)
    if len(us) == 0:
        return []
    P = np.column_stack([us, vs]).astype(float)
    up = _up_in_image(T_BC, frame, P.mean(0), approx)
    h = P @ up  # height along the image direction of world up
    comp = lab[vs, us]
    tree = cKDTree(P)
    rad = 0.006 * px_per_m
    out = []
    for j in np.nonzero(soil[vs, us])[0]:
        nb = [k for k in tree.query_ball_point(P[j], rad) if comp[k] == comp[j]]
        if h[j] <= min(h[k] for k in nb) + 1e-6 and len(nb) >= 4:
            if all(np.linalg.norm(P[j] - o) > rad for o in out):
                out.append(P[j])
    return out


def estimate_all_stems(frame: Frame, T_BC: np.ndarray, cfg: dict, min_sep_m: float = 0.015) -> tuple[list[StemLineEstimate], list]:
    """Every plant in view: one estimate per stem root found in 3-D, duplicates (same root within min_sep_m)
    merged keeping the best right-image match. Also returns (root pixel, status) of the rejected candidates."""
    c = cfg["stemline"]
    g = _mask(frame.rgb, c)
    found, rejected = [], []
    for r in root_candidates(frame, T_BC, cfg, g):
        e = estimate_stem_line(frame, T_BC, cfg, root_px=r, gmask=g)
        if not e.ok:
            rejected.append((r, e.status))
            continue
        dup = [j for j, f in enumerate(found) if np.linalg.norm(f.root_B - e.root_B) < min_sep_m]
        if dup and found[dup[0]].match_score >= e.match_score:
            continue
        found = [f for j, f in enumerate(found) if j not in dup] + [e]
    found.sort(key=lambda e: (round(float(e.root_B[0]), 2), float(e.root_B[1])))
    return found, rejected


def estimate_cell_stem_line(frame: Frame, T_BC: np.ndarray, cfg: dict, plug_B: np.ndarray, radius_m: float = 0.022,
                            gmask: np.ndarray | None = None, candidates: list | None = None) -> tuple[StemLineEstimate | None, list]:
    """The stem of the plant in one known cell (tray grid): only root candidates near this plug's top are tried,
    and of the stems found the one whose root is nearest the plug centre (within radius_m, xy) is returned.
    Also returns (root pixel, status, 2-D stem or None) of every candidate tried."""
    c = cfg["stemline"]
    g = _mask(frame.rgb, c) if gmask is None else gmask
    cands = root_candidates(frame, T_BC, cfg, g) if candidates is None else candidates
    i = frame.intr
    T_CB = np.linalg.inv(T_BC)
    ctr = i.project(apply_T(T_CB, plug_B))
    px_per_m = i.fx / float(apply_T(T_CB, plug_B)[2])
    near = [r for r in cands if np.linalg.norm(r - ctr) <= (radius_m + float(c.get("root_soil_gap_m", 0.015))) * px_per_m]
    best, tried = None, []
    for r in sorted(near, key=lambda r: float(np.linalg.norm(r - ctr))):
        e = estimate_stem_line(frame, T_BC, cfg, root_px=r, gmask=g)
        tried.append((r, e.status, e.left))
        if not e.ok:
            continue
        d = float(np.linalg.norm((e.root_B - plug_B)[:2]))
        if d <= radius_m and (best is None or d < np.linalg.norm((best.root_B - plug_B)[:2])):
            best = e
    return best, tried
