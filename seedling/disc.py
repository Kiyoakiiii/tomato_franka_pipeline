"""The transplant disc (3D printed, red): six holes of 49.5 mm on a ~57 mm circle, 170 mm across, holes 55 mm
deep with a closed bottom. Located from one picture:

- the hole interiors are the saturated red-orange blobs (the flat top face is lit pale and fails that test);
- the outline of each visible hole interior is its rim circle on the top plane (looking in at an angle, the
  far inner wall shows from the rim down and the near rim hides the rest), so the outline pixels are cast onto
  the top plane and a circle is fitted;
- the top plane height is the dense depth on the top face right around the holes;
- the disc centre is the circle through the hole centres (radius fitted with 4+ holes seen, the nominal one with 3;
  user said ~52 mm, measured 57 mm on the disc: 57 is the nominal now);
  the hole sizes, their distance to that centre and their spacing (60 deg) are the checks;
- a hole counts as free when its red interior is seen (a plug in it shows soil and leaves instead);
- as the disc fills, fewer than 3 clean hole rims remain (leaves hang over the holes): then the disc is found from
  its outer rim (the top face outline on the top plane, a 170 mm circle) and its turn from the top face itself (the
  phase at which the six 49.5 mm hole circles hold the fewest top-face points); a hole is free when its inside is
  mostly red. (2026-10-28: the disc was turned 14 deg by hand mid-run; a prior-position check could not tell.)
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Frame
from .transforms import apply_T

DISC = {"hole_d": 0.0495, "hole_r": 0.057, "hole_depth": 0.055, "outer_d": 0.170, "n_holes": 6}


def _ray_plane(frame: Frame, T_BC: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    Kinv = np.linalg.inv(frame.intr.K())
    d = (T_BC[:3, :3] @ (Kinv @ np.column_stack([uv, np.ones(len(uv))]).T)).T
    o = T_BC[:3, 3]
    t = (z - o[2]) / d[:, 2]
    return o + t[:, None] * d


def _fit_circle(xy: np.ndarray):
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    c, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    r = float(np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2))
    res = np.hypot(*(xy - c[:2]).T) - r
    return c[:2], r, res


def _from_outer_rim(frame, T_BC, XYZ, hsv, red, ok, z_top, g, prior, n_good) -> dict:
    """centre from the outer rim, turn from the top face, free holes from the red inside (see the module notes)"""
    col = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 50)
    with np.errstate(invalid="ignore"):
        top = ok & col & (np.abs(XYZ[..., 2] - z_top) < 0.006)
    m = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return {"ok": False, "status": f"{n_good} clean hole rims and no disc top face at z {z_top:.3f}"}
    k = 1 + int(np.argmax(st[1:, 4]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # the outline, not its hull: leaves lying over the rim bite dents into it (they are outliers to a circle of the
    # known radius, fitted robustly: the visible arcs agree, 2026-10-28 18:13 the hull gave 81 mm)
    uv = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    xy = _ray_plane(frame, T_BC, uv, z_top)[:, :2]
    R0 = g["outer_d"] / 2
    c, _, _ = _fit_circle(xy)
    for tol in (0.012, 0.008, 0.005, 0.003, 0.003):
        d = np.hypot(*(xy - c).T)
        inl = np.abs(d - R0) < tol
        for _ in range(10):  # Gauss-Newton, radius fixed
            dv = xy[inl] - c
            L = np.hypot(*dv.T)
            step, *_ = np.linalg.lstsq(-dv / L[:, None], -(L - R0), rcond=None)
            c = c + step
            if np.linalg.norm(step) < 1e-6:
                break
    d = np.hypot(*(xy - c).T)
    inl = np.abs(d - R0) < 0.003
    rms = float(np.sqrt(np.mean((d[inl] - R0) ** 2))) if inl.any() else 1.0
    # the inliers must cover the circle well (not one short arc)
    ang = np.degrees(np.arctan2(*(xy[inl] - c).T[::-1])) if inl.any() else np.array([])
    cover = len(np.unique((ang // 10).astype(int))) * 10
    if inl.mean() < 0.2 or rms > 0.002 or cover < 180:
        return {"ok": False, "status": f"{n_good} clean hole rims; outer rim: {inl.mean() * 100:.0f}% of the outline on a "
                f"{R0 * 2000:.0f} mm circle over {cover} deg, rms {rms * 1000:.1f} mm (disc not in full view?)"}
    # turn: the six hole circles at the phase that leaves the fewest top-face points inside them
    P = XYZ[ok]
    P = P[(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < g["outer_d"] / 2 - 0.003) & (np.abs(P[:, 2] - z_top) < 0.003)]
    r_h = float(prior.get("hole_circle_r_mm", g["hole_r"] * 1000)) / 1000
    scores = []
    for phi in np.arange(0.0, 60.0, 0.5):
        a = np.radians(phi + 60.0 * np.arange(6))
        H = c + r_h * np.column_stack([np.cos(a), np.sin(a)])
        d = np.min(np.hypot(P[:, 0, None] - H[None, :, 0], P[:, 1, None] - H[None, :, 1]), axis=1)
        scores.append((float(np.mean(d < g["hole_d"] / 2 - 0.004)), phi))
    frac, phi = min(scores)
    if frac > 0.3:
        return {"ok": False, "status": f"{n_good} clean hole rims; no turn of the disc leaves its holes clear of the top face"}
    # numbering kept: the same hole keeps its number when the disc is turned by less than 30 deg
    p0 = float(prior.get("phase_deg", phi))
    phi = p0 + ((phi - p0 + 30) % 60 - 30)
    a = np.radians(phi + 60.0 * np.arange(6))
    H = c + r_h * np.column_stack([np.cos(a), np.sin(a)])
    K, T_CB = frame.intr.K(), np.linalg.inv(T_BC)
    red_frac = []
    for hc in H:
        ring = np.array([[hc[0] + 0.018 * np.cos(t), hc[1] + 0.018 * np.sin(t), z_top] for t in np.linspace(0, 2 * np.pi, 40)])
        pc = apply_T(T_CB, ring)
        px = (K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.int32)
        mm = np.zeros(red.shape, np.uint8)
        cv2.fillPoly(mm, [px], 1)
        red_frac.append(float(red[mm > 0].mean()) if mm.any() else 0.0)
    free = [i + 1 for i, f_ in enumerate(red_frac) if f_ > 0.5]
    return {"ok": True, "status": f"outer rim ({inl.mean() * 100:.0f}% of the outline over {cover} deg, rms {rms * 1000:.1f} mm) and top face (turn {phi:.1f} deg, "
            f"{frac * 100:.0f}% top points in the holes); {n_good} clean hole rims", "centre": [float(c[0]), float(c[1])],
            "z_top": round(z_top, 4), "z_top_depth": round(z_top, 4), "free_holes": free, "phase_deg": round(float(phi), 1),
            "hole_circle_r_mm": round(r_h * 1000, 1), "holes_seen": [], "hole_radius_mm": [], "rim_rms_mm": [],
            "red_fraction": [round(v, 2) for v in red_frac], "holes": H.tolist(), "hole_bottom_z": round(z_top - g["hole_depth"], 4),
            "from_outer_rim": True}


def locate_disc(frame: Frame, T_BC: np.ndarray, area=(0.30, 0.95, -0.50, 0.50), max_top_z: float = 0.45, geom: dict | None = None,
                prior: dict | None = None) -> dict:
    g = {**DISC, **(geom or {})}
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    ok = np.isfinite(frame.xyz).all(-1)
    XYZ = np.full(frame.xyz.shape, np.nan)
    XYZ[ok] = apply_T(T_BC, frame.xyz[ok])
    x0, x1, y0, y1 = area
    with np.errstate(invalid="ignore"):
        inarea = ok & (XYZ[..., 0] > x0) & (XYZ[..., 0] < x1) & (XYZ[..., 1] > y0) & (XYZ[..., 1] < y1) & (XYZ[..., 2] < max_top_z)
    red = ((hsv[..., 0] < 20) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 100)
    # hole interiors often lack depth (deep, steep walls): the area test is per blob (its pixels with depth), not per pixel
    m0 = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the rim between a hole and the disc's red side wall is ~3 mm (a few pixels): eroded apart, each blob is grown
    # back inside the red mask before its outline is taken
    m = cv2.erode(m0, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    blobs = []
    for k in range(1, n):
        w, hgt, a = st[k, 2], st[k, 3], st[k, 4]
        if a < 300 or max(w, hgt) > 2.5 * min(w, hgt):
            continue
        if a < 0.5 * w * hgt:
            # filled enough? the eroded blob of a free hole whose bottom is darker fills less of its box (2026-11-05: 0.48-0.50,
            # two free holes missed); then the red share inside its convex hull decides: free holes 0.74-0.99, a hole
            # with a plug in it (red ring around the soil) 0.41-0.42
            grown = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) & m0
            cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            hm = np.zeros_like(m0)
            cv2.fillConvexPoly(hm, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
            if (m0 & hm).sum() < 0.65 * max(int(hm.sum()), 1):
                continue
        sel = (lab == k) & inarea
        if sel.sum() >= 0.2 * a or ((lab == k) & ok).sum() < 0.05 * a:
            # in the area, or no depth at all to tell (checked below by the fit)
            if sel.sum() >= 0.2 * a or not ((lab == k) & ok).any():
                blobs.append(k)
    if len(blobs) < (1 if prior else 3):
        return {"ok": False, "status": f"only {len(blobs)} red hole-like blobs in the work area"
                + (" (disc full, or not in view)" if prior else "")}
    # top plane: dense depth on the (non-red) top face in a band around the blobs
    allb = np.isin(lab, blobs).astype(np.uint8)
    band = cv2.dilate(allb, np.ones((15, 15), np.uint8)) & ~cv2.dilate(allb, np.ones((5, 5), np.uint8)).astype(bool)
    zb = XYZ[band.astype(bool) & ok & ~red, 2]
    if len(zb) >= 200:
        h, e = np.histogram(zb, bins=np.arange(np.nanmin(zb), np.nanmax(zb) + 0.004, 0.002))
        hs = np.convolve(h, [1, 2, 1], "same")
        z_pk = e[int(np.argmax(hs))] + 0.001
        z_top = float(np.median(zb[np.abs(zb - z_pk) < 0.004]))
    elif prior:
        z_top = float(prior["z_top"])
    else:
        return {"ok": False, "status": "no depth on the disc top face"}
    outlines = []
    for k in blobs:
        grown = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) & m0
        cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # convex hull: a bright (over-exposed) patch on the lit far wall can bite a notch into the blob; the rim is convex
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))[:, 0, :].astype(float)
        pts = [np.linspace(hull[i], hull[(i + 1) % len(hull)], max(2, int(np.linalg.norm(hull[(i + 1) % len(hull)] - hull[i]))),
                           endpoint=False) for i in range(len(hull))]  # evenly along the hull, not just its vertices
        outlines.append((k, np.vstack(pts)))

    def fit_all(zp):
        out = []
        for k, c in outlines:
            ctr, r, res = _fit_circle(_ray_plane(frame, T_BC, c, zp)[:, :2])
            out.append({"centre": ctr, "r": r, "rms_mm": float(np.sqrt(np.mean(res ** 2))) * 1000, "px": int(st[k, 4])})
        return out

    rh = g["hole_d"] / 2
    z_depth = z_top
    holes = fit_all(z_top)
    good = [hh for hh in holes if abs(hh["r"] - rh) < 0.005 and hh["rms_mm"] < 3.0]
    if len(good) < 3 and prior:
        return _from_outer_rim(frame, T_BC, XYZ, hsv, red, ok, float(prior["z_top"]), g, prior, len(good))
    if len(good) < 3:
        return {"ok": False, "status": f"{len(good)} hole rims fit a {g['hole_d'] * 1000:.1f} mm circle (radii mm: "
                f"{[round(hh['r'] * 1000, 1) for hh in holes]})", "z_top": z_top}
    C = np.array([hh["centre"] for hh in good])
    # disc centre and hole-circle radius: with 4+ holes both fitted (circle through the hole centres), else the
    # radius fixed to the nominal one (Gauss-Newton from the mean)
    if len(C) >= 4:
        c, r_holes, _ = _fit_circle(C)
    else:
        c, r_holes = C.mean(0), g["hole_r"]
        for _ in range(20):
            d = C - c
            L = np.hypot(*d.T)
            step, *_ = np.linalg.lstsq(-d / L[:, None], -(L - r_holes), rcond=None)
            c = c + step
            if np.linalg.norm(step) < 1e-6:
                break
    rsd = np.hypot(*(C - c).T) - r_holes
    ang = np.degrees(np.arctan2(C[:, 1] - c[1], C[:, 0] - c[0]))
    # spacing check: angles modulo 60 should agree
    phase = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ang * 6))))) / 6
    dphase = (ang - phase + 30) % 60 - 30
    status = "ok"
    okk = True
    if abs(r_holes - g["hole_r"]) > 0.008:
        okk, status = False, f"hole circle radius {r_holes * 1000:.0f} mm, not about {g['hole_r'] * 1000:.0f} mm"
    elif np.abs(rsd).max() > 0.004:
        okk, status = False, f"hole centres up to {np.abs(rsd).max() * 1000:.0f} mm off their common circle"
    elif np.abs(dphase).max() > 6:
        okk, status = False, f"hole spacing off by up to {np.abs(dphase).max():.0f} deg from 60 deg"
    # all six hole centres from the fitted centre and phase (the ones not seen included)
    all_h = [[float(c[0] + r_holes * np.cos(np.radians(phase + 60 * i))), float(c[1] + r_holes * np.sin(np.radians(phase + 60 * i)))]
             for i in range(g["n_holes"])]
    # a hole is free when its red interior was seen (a plug in it shows soil and leaves instead)
    free = [i + 1 for i, h in enumerate(all_h) if np.hypot(*(C - np.array(h)).T).min() < 0.010]
    return {"ok": okk, "status": status, "centre": [float(c[0]), float(c[1])], "z_top": round(z_top, 4), "z_top_depth": round(z_depth, 4),
            "free_holes": free,
            "phase_deg": round(float(phase), 1), "hole_circle_r_mm": round(float(r_holes) * 1000, 1), "holes_seen": [[round(float(v), 4) for v in hh["centre"]] for hh in good],
            "hole_radius_mm": [round(hh["r"] * 1000, 1) for hh in good], "rim_rms_mm": [round(hh["rms_mm"], 1) for hh in good],
            "centre_residual_mm": [round(float(v) * 1000, 1) for v in rsd], "spacing_residual_deg": [round(float(v), 1) for v in dphase],
            "holes": all_h, "hole_bottom_z": round(z_top - g["hole_depth"], 4)}


def taken_holes(disc: dict) -> list[int]:
    """the holes with a plant in them: "occupied_holes" (a plant seen above it, or put there by us; 60 / 77 keep it)
    when the file has it, else every hole not seen free. A hole that was not seen at all (out of view, hidden) is
    neither free nor taken then: no placing into it, and no plant assumed there (2026-11-05: an unseen H4 under the
    forearm's way blocked grasps for a plant that was not there)"""
    n = len(disc.get("holes", []))
    if "occupied_holes" in disc:
        return [int(h) for h in disc["occupied_holes"]]
    free = set(disc.get("free_holes", range(1, n + 1)))
    return [h for h in range(1, n + 1) if h not in free]


def occupied_from_view(disc: dict, points_B, prior: dict | None = None, r: float = 0.020, min_points: int = 30) -> list[int]:
    """holes not seen free with a plant seen above them in this picture (3D points within r of the hole centre,
    15-250 mm over the disc top), plus those occupied before when the disc has not moved (within 10 mm)"""
    import numpy as np

    P = np.asarray(points_B)
    zt = float(disc["z_top"])
    out = []
    free = set(disc.get("free_holes", []))
    for i, (hx, hy) in enumerate(disc["holes"], 1):
        if i in free:
            continue
        m = (np.hypot(P[:, 0] - hx, P[:, 1] - hy) < r) & (P[:, 2] > zt + 0.015) & (P[:, 2] < zt + 0.25)
        if int(m.sum()) >= min_points:
            out.append(i)
    if prior and "occupied_holes" in prior and prior.get("centre") is not None and \
            np.hypot(*(np.subtract(prior["centre"][:2], disc["centre"][:2]))) < 0.010:
        out += [h for h in prior["occupied_holes"] if h not in free and h not in out]
    return sorted(out)
