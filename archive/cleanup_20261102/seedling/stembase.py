"""The stem where it leaves the substrate, in one known cell: only a short piece just above the soil is used,
so leaves and leaf stalks (higher up) do not disturb it.

1. soil height: dense depth of the peat on this plug's top;
2. search volume: a short vertical cylinder on the plug (radius base_radius_m around the cell centre,
   base_z_m above the soil), projected into the left image;
3. the image is turned so that world-up points up at the plug; a white top-hat across the rows keeps narrow
   bright structures (a pale or green stem against dark peat); runs of stem width give centre points per
   row; a RANSAC line through them (steep enough) is the stem piece;
4. right image: the band is matched (stemline.match_right), triangulated row by row, and the 3-D points must
   lie inside the cylinder (a stem of another plant in front of or behind it is rejected here);
5. 3-D line through the points: its intersection with the soil plane is the root, its direction the stem's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Frame
from .stemline import StemLine2D, match_right, substrate_mask, triangulate_lines, window_disparity_residuals, _up_in_image
from .transforms import apply_T, inv_T, unit


@dataclass
class StemBase:
    ok: bool
    status: str
    soil_z: float | None = None
    root_B: np.ndarray | None = None  # stem at the soil surface (base frame)
    dir_B: np.ndarray | None = None  # unit, upward
    tilt_deg: float | None = None
    points_B: np.ndarray | None = None
    root_off_centre_mm: float | None = None
    match_score: float = 0.0
    disparity_resid_px: float | None = None
    left: StemLine2D | None = None
    extra: dict = field(default_factory=dict)

    def at_height(self, h: float) -> np.ndarray | None:
        """Point of the stem line h above the soil."""
        if self.root_B is None:
            return None
        return self.root_B + self.dir_B * h / max(self.dir_B[2], 1e-6)

    def summary(self) -> dict:
        r = lambda a, k=4: None if a is None else np.round(np.asarray(a, float), k).tolist()  # noqa: E731
        return {"ok": self.ok, "status": self.status, "soil_z": None if self.soil_z is None else round(self.soil_z, 4),
                "root_B": r(self.root_B), "dir_B": r(self.dir_B, 3), "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 1),
                "root_off_centre_mm": None if self.root_off_centre_mm is None else round(self.root_off_centre_mm, 1),
                "match_score": round(self.match_score, 3),
                "disparity_resid_px": None if self.disparity_resid_px is None else round(self.disparity_resid_px, 2), **self.extra}


def plug_soil_z(frame: Frame, T_BC: np.ndarray, centre_xy, z_guess: float, cfg: dict, radius: float = 0.02) -> float | None:
    """Median height of the dense points on peat-coloured pixels over this plug (within radius of the centre,
    within 3 cm of the guess)."""
    soil = substrate_mask(frame.rgb, cfg["stemline"])
    ok = soil & np.isfinite(frame.xyz).all(-1)
    P = apply_T(T_BC, frame.xyz[ok])
    m = (np.hypot(P[:, 0] - centre_xy[0], P[:, 1] - centre_xy[1]) < radius) & (np.abs(P[:, 2] - z_guess) < 0.03)
    if m.sum() < 30:
        return None
    return float(np.median(P[m, 2]))


def find_stem_base(frame: Frame, T_BC: np.ndarray, cfg: dict, centre_B: np.ndarray) -> StemBase:
    c = cfg["stembase"]
    i = frame.intr
    xy = np.asarray(centre_B[:2], float)
    soil_z = plug_soil_z(frame, T_BC, xy, float(centre_B[2]), cfg)
    est = StemBase(False, "", soil_z=soil_z)
    if soil_z is None:
        soil_z = float(centre_B[2])
        est.extra["soil_from"] = "cell model"
    R = float(c["base_radius_m"])
    h0, h1 = c["base_z_m"]
    T_CB = inv_T(T_BC)
    # search volume in the image
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    vol = np.array([[xy[0] + R * np.cos(a), xy[1] + R * np.sin(a), soil_z + h] for h in (h0, h1) for a in ang])
    pc = apply_T(T_CB, vol)
    if (pc[:, 2] < 0.1).any():
        est.status = "plug not in front of the camera"
        return est
    uv = i.project(pc)
    H, W = frame.rgb.shape[:2]
    ctr_px = i.project(apply_T(T_CB, np.array([xy[0], xy[1], soil_z + 0.5 * (h0 + h1)])))
    depth = float(apply_T(T_CB, np.array([xy[0], xy[1], soil_z]))[2])
    px_per_m = i.fx / depth
    up = _up_in_image(T_BC, frame, ctr_px, depth)
    # rotate the image about the search centre so that world-up points up (-v)
    rot = float(np.degrees(np.arctan2(up[0], -up[1])))  # angle of `up` from the image's -v axis
    M = cv2.getRotationMatrix2D((float(ctr_px[0]), float(ctr_px[1])), -rot, 1.0)
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    G = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    uvr = (M @ np.column_stack([uv, np.ones(len(uv))]).T).T
    u0, u1 = int(np.floor(uvr[:, 0].min())) - 2, int(np.ceil(uvr[:, 0].max())) + 2
    v0, v1 = int(np.floor(uvr[:, 1].min())), int(np.ceil(uvr[:, 1].max()))
    if u0 < 5 or v0 < 5 or u1 > W - 5 or v1 > H - 5:
        est.status = "search volume at the image border"
        return est
    wmin = max(2.0, float(c["min_width_m"]) * px_per_m)
    wmax = float(c["max_width_m"]) * px_per_m
    pad = int(3 * wmax) + 4
    patch = cv2.GaussianBlur(G[v0:v1, u0 - pad:u1 + pad], (0, 0), 0.8)
    k = max(3, int(round(2.2 * wmax)) | 1)
    tophat = patch - cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    roi = np.zeros(G.shape, np.uint8)
    cv2.fillConvexPoly(roi, cv2.convexHull(uvr.astype(np.int32)), 1)
    roi = roi[v0:v1, u0 - pad:u1 + pad] > 0
    pts = []
    for r in range(tophat.shape[0]):
        row = (tophat[r] > float(c["min_contrast"])) & roi[r]
        j = 0
        while j < len(row):
            if row[j]:
                k2 = j
                while k2 < len(row) and row[k2]:
                    k2 += 1
                width = k2 - j
                if wmin <= width <= wmax:
                    seg = tophat[r, j:k2]
                    pts.append((j + float((seg * np.arange(width)).sum() / seg.sum()), r, width))
                j = k2
            else:
                j += 1
    est.extra["ridge_points"] = len(pts)
    if len(pts) < 6:
        est.status = "no stem-like ridge just above the soil (base hidden or no plant)"
        return est
    P2 = np.array(pts, float)
    # RANSAC: x = a + b * r, |b| <= tan(max_tilt_image)
    rng = np.random.default_rng(0)
    bmax = np.tan(np.radians(float(c["max_image_tilt_deg"])))
    best, n_rows = None, tophat.shape[0]
    for _ in range(300):
        s = P2[rng.choice(len(P2), 2, replace=False)]
        if s[0, 1] == s[1, 1]:
            continue
        b = (s[1, 0] - s[0, 0]) / (s[1, 1] - s[0, 1])
        if abs(b) > bmax:
            continue
        a = s[0, 0] - b * s[0, 1]
        inl = np.abs(P2[:, 0] - (a + b * P2[:, 1])) <= 1.5
        span = len(np.unique(P2[inl, 1]))
        score = span - 0.02 * abs(a + b * n_rows / 2 - (ctr_px[0] - (u0 - pad)))  # long, then near the centre
        if best is None or score > best[0]:
            best = (score, inl, span)
    # rows a vertical stem piece h0..h1 above the soil covers in the turned image (foreshortened by the view)
    pv = i.project(apply_T(T_CB, np.array([[xy[0], xy[1], soil_z + h0], [xy[0], xy[1], soil_z + h1]])))
    rows_expected = float(np.linalg.norm(pv[1] - pv[0]))
    est.extra["rows_expected"] = round(rows_expected, 1)
    if best is None or best[2] < float(c["min_rows_fraction"]) * rows_expected:
        est.status = f"no straight stem piece just above the soil ({0 if best is None else best[2]} rows, {rows_expected:.0f} expected)"
        return est
    inl = best[1]
    # keep the continuous run of rows nearest the soil (stray points on leaves above also fit the line), it
    # must be long enough and reach down to the soil (else bits of leaf above a hidden base were joined)
    gap_max = max(3.0, 0.15 * rows_expected)
    rr = np.sort(np.unique(P2[inl, 1]))
    runs, start = [], 0
    for j in range(1, len(rr) + 1):
        if j == len(rr) or rr[j] - rr[j - 1] > gap_max:
            runs.append(rr[start:j])
            start = j
    soil_row = float((M @ np.array([*pv[0], 1.0]))[1]) - v0  # row of the stem at h0 above the soil (turned image)
    run = max(runs, key=lambda r: (r.max() >= soil_row - max(4.0, 0.3 * rows_expected), r.max() - r.min()))
    if run.max() - run.min() + 1 < float(c["min_rows_fraction"]) * rows_expected:
        est.status = f"no continuous stem piece just above the soil ({run.max() - run.min() + 1:.0f} rows, {rows_expected:.0f} expected)"
        return est
    if soil_row - run.max() > max(4.0, 0.3 * rows_expected):
        est.status = f"stem piece does not reach the soil ({soil_row - run.max():.0f} rows short): base hidden?"
        return est
    inl = inl & (P2[:, 1] >= run.min()) & (P2[:, 1] <= run.max())
    A = np.column_stack([np.ones(inl.sum()), P2[inl, 1]])
    a, b = np.linalg.lstsq(A, P2[inl, 0], rcond=None)[0]
    width = float(np.median(P2[inl, 2]))
    rows = np.unique(P2[inl, 1]).astype(int)
    # back to image coordinates
    Minv = cv2.invertAffineTransform(M)

    def to_img(x, r):
        q = np.array([x + u0 - pad, r + v0, 1.0])
        return Minv @ q

    band = []
    for r in rows:
        xc = a + b * r
        for dx in np.arange(-width / 2, width / 2 + 0.01, 1.0):
            band.append(to_img(xc + dx, r))
    band = np.unique(np.round(np.array(band)).astype(int), axis=0)
    top, bottom = to_img(a + b * rows.min(), rows.min()), to_img(a + b * rows.max(), rows.max())
    d2 = unit(top - bottom)
    path = np.array([to_img(a + b * r, r) for r in rows[::-1]])
    left = StemLine2D(centre=(top + bottom) / 2, direction=d2, root_px=bottom, top_px=top, band=band, width_px=width,
                      path=path, curvature_px=0.0, depth_hint_m=depth)
    est.left = left
    if abs(d2[1]) < cfg["stemline"]["min_vertical_fraction"]:
        est.status = "stem piece nearly horizontal in the image"
        return est
    cR, dR, sc, model = match_right(left, frame, cfg, depth)
    est.match_score = sc
    if cR is None:
        est.status = f"stem piece not found in the right image (score {sc:.2f})"
        return est
    P_C = triangulate_lines(left, cR, dR, frame)
    if len(P_C) < 5:
        est.status = "triangulation failed"
        return est
    P_B = apply_T(T_BC, P_C)
    res = window_disparity_residuals(left, model, rows_per_window=7, step=3)
    if len(res) >= 2:
        est.disparity_resid_px = float(np.sqrt(np.mean(res ** 2)))
    # the piece must be inside the search cylinder (not a stem of another plant in front / behind)
    r_xy = np.hypot(P_B[:, 0] - xy[0], P_B[:, 1] - xy[1])
    inside = (r_xy <= R + 0.006) & (P_B[:, 2] >= soil_z - 0.006) & (P_B[:, 2] <= soil_z + h1 + 0.01)
    est.extra["inside_fraction"] = round(float(inside.mean()), 2)
    if inside.mean() < 0.7:
        med = np.median(P_B, axis=0)
        est.status = (f"the matched piece is not on this plug ({np.hypot(med[0] - xy[0], med[1] - xy[1]) * 1000:.0f} mm from the centre, "
                      f"{(med[2] - soil_z) * 1000:+.0f} mm above the soil)")
        return est
    P_B = P_B[inside]
    m = P_B.mean(0)
    _, V = np.linalg.eigh(np.cov((P_B - m).T))
    d = V[:, -1] if V[2, -1] >= 0 else -V[:, -1]
    if d[2] < np.cos(np.radians(float(c["max_tilt_deg"]))):
        est.status = f"stem piece tilted {np.degrees(np.arccos(d[2])):.0f} deg (max {c['max_tilt_deg']})"
        return est
    root = m + d * (soil_z - m[2]) / d[2]
    est.points_B, est.dir_B, est.root_B = P_B, d, root
    est.tilt_deg = float(np.degrees(np.arccos(np.clip(d[2], -1, 1))))
    est.root_off_centre_mm = float(np.hypot(root[0] - xy[0], root[1] - xy[1]) * 1000)
    est.extra.update(width_mm=round(width / px_per_m * 1000, 1), rows=int(len(rows)), n_points=int(len(P_B)))
    if est.disparity_resid_px is not None and est.disparity_resid_px > float(cfg["stemline"].get("max_disparity_resid_px", 1.0)):
        est.status = f"right-image match inconsistent ({est.disparity_resid_px:.1f} px)"
        return est
    est.ok, est.status = True, "ok"
    return est
