"""Stem centreline extraction and local 3D line fitting (RANSAC + PCA).

The fitted line is the visible stem *surface*; with ``surface_to_axis`` the
centre is pushed back along the viewing ray by half the estimated diameter.
That correction is a model assumption to be checked against independent
references, not a guarantee of axis accuracy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .camera import Frame


@dataclass
class StemSegment:
    instance: int
    seg_index: int
    path_range: tuple[int, int]
    pixel_center: tuple[float, float]
    bend_deg: float
    ok: bool = False
    reject_reason: str = ""
    p_C: list[float] | None = None  # estimated axis point, camera frame (m)
    dir_C: list[float] | None = None  # unit tangent, camera frame, oriented upward in the image
    length_m: float | None = None
    diameter_m: float | None = None
    n_candidates: int = 0  # mask pixels assigned to the window
    n_valid: int = 0  # pixels with usable depth
    n_inliers: int = 0
    valid_fraction: float = 0.0
    inlier_ratio: float = 0.0
    rms_m: float | None = None
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstanceGeometry:
    instance: int
    conf: float
    path_uv: np.ndarray  # (N, 2) ordered centreline pixels, top -> bottom
    halfwidth_px: np.ndarray  # (N,)
    segments: list[StemSegment] = field(default_factory=list)
    reject_reason: str = ""


# ---------------------------------------------------------------- 2D -------------------------------
def clean_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == keep).astype(np.uint8)
    # fill holes
    inv = 1 - m
    n2, lab2 = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])).tolist())
    holes = ~np.isin(lab2, list(border)) & (inv > 0)
    return (m > 0) | holes


_NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bfs_far(start, pix: set) -> tuple[tuple[int, int], dict]:
    prev = {start: None}
    q = deque([start])
    last = start
    while q:
        cur = q.popleft()
        last = cur
        y, x = cur
        for dy, dx in _NBR:
            nb = (y + dy, x + dx)
            if nb in pix and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return last, prev


def longest_skeleton_path(mask: np.ndarray) -> np.ndarray:
    """Ordered (N, 2) [u, v] pixels along the longest skeleton path, from top (small v) to bottom."""
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    if len(ys) < 2:
        return np.zeros((0, 2), int)
    pix = set(zip(ys.tolist(), xs.tolist()))
    # start BFS from the topmost pixel to find one end, then the farthest from that end
    start = min(pix)
    a, _ = _bfs_far(start, pix)
    b, prev = _bfs_far(a, pix)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path = np.array([[x, y] for y, x in path], dtype=int)
    if path[0, 1] > path[-1, 1]:
        path = path[::-1]
    return path


def _principal_dir(pts: np.ndarray) -> np.ndarray:
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[0]


def bend_angle_deg(path_uv: np.ndarray) -> float:
    if len(path_uv) < 8:
        return 0.0
    h = len(path_uv) // 2
    d1 = _principal_dir(path_uv[:h].astype(float))
    d2 = _principal_dir(path_uv[h:].astype(float))
    return float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0, 1))))


# ---------------------------------------------------------------- 3D -------------------------------
def fit_line_ransac(pts: np.ndarray, threshold: float, iterations: int = 200, rng=None):
    """Return (point, unit_dir, inlier_mask) for a 3D line, or None."""
    n = len(pts)
    if n < 3:
        return None
    rng = rng or np.random.default_rng(0)
    best = None
    best_count = -1
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        L = np.linalg.norm(d)
        if L < 1e-4:
            continue
        d = d / L
        r = pts - pts[i]
        dist = np.linalg.norm(r - np.outer(r @ d, d), axis=1)
        inl = dist < threshold
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best = cnt, inl
    if best is None or best_count < 3:
        return None
    # PCA refinement on inliers (one re-selection pass)
    for _ in range(2):
        P = pts[best]
        c = P.mean(0)
        d = _principal_dir(P)
        r = pts - c
        dist = np.linalg.norm(r - np.outer(r @ d, d), axis=1)
        new = dist < threshold
        if new.sum() < 3:
            break
        best = new
    P = pts[best]
    c = P.mean(0)
    d = _principal_dir(P)
    return c, d / np.linalg.norm(d), best


def analyse_instance(frame: Frame, mask: np.ndarray, instance: int, conf: float, pcfg: dict) -> InstanceGeometry:
    m = clean_mask(mask)
    path = longest_skeleton_path(m)
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    geo = InstanceGeometry(instance, conf, path, dist[path[:, 1], path[:, 0]] if len(path) else np.zeros(0))
    L = int(pcfg["segment_length_px"])
    if len(path) < max(int(pcfg["segment_length_px"]) // 2, 8):
        geo.reject_reason = f"centreline too short ({len(path)} px)"
        return geo

    ys, xs = np.nonzero(m)
    owner = cKDTree(path).query(np.column_stack([xs, ys]), k=1)[1]
    interior = dist[ys, xs] >= 1.5
    xyz = frame.xyz[ys, xs]
    conf_map = frame.conf[ys, xs] if frame.conf is not None else None
    finite = np.isfinite(xyz).all(1)
    zmin, zmax = 0.05, 5.0
    usable = finite & (xyz[:, 2] > zmin) & (xyz[:, 2] < zmax)
    if conf_map is not None:
        usable &= conf_map <= float(pcfg["max_point_confidence"])

    step = int(pcfg["segment_step_px"])
    L = min(L, len(path))
    starts = list(range(0, max(1, len(path) - L + 1), step))
    rng = np.random.default_rng(0)
    for k, s in enumerate(starts):
        e = s + L
        sub = path[s:e]
        seg = StemSegment(instance, k, (s, e), tuple(float(v) for v in sub.mean(0)), bend_angle_deg(sub))
        geo.segments.append(seg)
        sel = (owner >= s) & (owner < e)
        seg.n_candidates = int(sel.sum())
        if seg.bend_deg > float(pcfg["max_bend_deg"]):
            seg.reject_reason = f"bend {seg.bend_deg:.0f} deg"
            continue
        use = sel & usable
        if (use & interior).sum() >= int(pcfg["min_support_points"]):
            use &= interior  # prefer interior pixels to avoid edge mixing with background
        seg.n_valid = int(use.sum())
        seg.valid_fraction = seg.n_valid / max(seg.n_candidates, 1)
        if seg.n_valid < int(pcfg["min_support_points"]):
            seg.reject_reason = f"insufficient valid depth ({seg.n_valid} pts)"
            continue
        P = xyz[use].astype(float)
        zmed = np.median(P[:, 2])
        P = P[np.abs(P[:, 2] - zmed) < float(pcfg["depth_jump_m"])]
        if len(P) < int(pcfg["min_support_points"]):
            seg.reject_reason = "depth discontinuity"
            continue
        fit = fit_line_ransac(P, float(pcfg["ransac_threshold_m"]), int(pcfg["ransac_iterations"]), rng)
        if fit is None:
            seg.reject_reason = "line fit failed"
            continue
        c, d, inl = fit
        seg.n_inliers = int(inl.sum())
        seg.inlier_ratio = seg.n_inliers / len(P)
        r = P[inl] - c
        t = r @ d
        resid = np.linalg.norm(r - np.outer(t, d), axis=1)
        seg.rms_m = float(np.sqrt(np.mean(resid**2)))
        seg.length_m = float(np.percentile(t, 97) - np.percentile(t, 3))
        if d[1] > 0:  # orient upward in the image (camera y points down)
            d = -d
        hw_px = float(np.median(geo.halfwidth_px[s:e]))
        seg.diameter_m = float(2 * hw_px * c[2] / frame.intr.fx)
        if pcfg.get("surface_to_axis", True):
            c = c + (seg.diameter_m / 2.0) * c / np.linalg.norm(c)
        seg.p_C, seg.dir_C = c.tolist(), d.tolist()
        if seg.inlier_ratio < float(pcfg["min_inlier_ratio"]):
            seg.reject_reason = f"inlier ratio {seg.inlier_ratio:.2f}"
        elif seg.rms_m > float(pcfg["max_rms_m"]):
            seg.reject_reason = f"rms {seg.rms_m*1000:.1f} mm"
        elif seg.length_m < float(pcfg["min_length_m"]):
            seg.reject_reason = f"length {seg.length_m*1000:.0f} mm"
        else:
            seg.ok = True
            seg.score = (
                min(seg.valid_fraction, 1.0)
                + seg.inlier_ratio
                + min(seg.length_m / 0.03, 1.0)
                - seg.rms_m / float(pcfg["max_rms_m"])
                - seg.bend_deg / float(pcfg["max_bend_deg"]) * 0.5
            )
    return geo


def analyse_frame(frame: Frame, instances, pcfg: dict) -> list[InstanceGeometry]:
    return [analyse_instance(frame, inst.mask, inst.index, inst.conf, pcfg) for inst in instances]
