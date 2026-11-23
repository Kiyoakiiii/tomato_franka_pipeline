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


def _free_by_inside(XYZ, ok, H, z_top, red_frac, min_red: float = 0.9, r: float = 0.020, min_points: int = 30) -> list[bool]:
    """per hole: free by its inside, without a clean rim seen: mostly red inside (>= min_red) and no plant above it
    (fewer than min_points 3D points within r of its centre, 15-250 mm over the top). 2026-11-17, from 20 earlier
    pictures with planted holes: a plug's hole shows 0.47-0.56 red, and two planted holes 1.0 (their plants seen above)"""
    P = XYZ[ok]
    out = []
    for hc, rf in zip(H, red_frac):
        m = (np.hypot(P[:, 0] - hc[0], P[:, 1] - hc[1]) < r) & (P[:, 2] > z_top + 0.015) & (P[:, 2] < z_top + 0.25)
        out.append(rf >= min_red and int(m.sum()) < min_points)
    return out


def _red_fractions(frame, T_BC, red, H, z_top) -> list[float]:
    """share of red pixels inside an 18 mm circle around each hole centre on the top plane (a free hole shows its red
    inside; a plug in it soil and leaves)"""
    K, T_CB = frame.intr.K(), np.linalg.inv(T_BC)
    out = []
    for hc in H:
        ring = np.array([[hc[0] + 0.018 * np.cos(t), hc[1] + 0.018 * np.sin(t), z_top] for t in np.linspace(0, 2 * np.pi, 40)])
        pc = apply_T(T_CB, ring)
        px = (K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.int32)
        mm = np.zeros(red.shape, np.uint8)
        cv2.fillPoly(mm, [px], 1)
        out.append(float(red[mm > 0].mean()) if mm.any() else 0.0)
    return out


def _turn_from_shading(frame, T_BC, c, r_h, z_top, g, step_deg: float = 0.25):
    """the disc's turn from the picture's brightness: the six hole circles at the turn where the top face around them is
    brightest compared with their insides (lit from above, the holes are shaded; a plug in one is dark soil). Independent
    of the colours (2026-11-17: after a restart, at dusk, the top face came out as red as the holes, S 166 instead of
    114, and neither the hole blobs nor the top-face depth gave the turn; the shading did: top 166 vs holes 81 gray,
    against 20 at the wrong turn). Mean gray inside 24 mm of each hole centre (nearly the whole 24.75 mm hole) against
    the top face 1 mm beyond the rims: within 0.5 deg of the turn the hole rims gave on four pictures (the medians inside
    18 mm: up to 4 deg off). (turn deg, contrast, contrast 30 deg off it) or None"""
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY).astype(float)
    K, T_CB = frame.intr.K(), np.linalg.inv(T_BC)
    rim = np.array([[c[0] + (g["outer_d"] / 2) * np.cos(a), c[1] + (g["outer_d"] / 2) * np.sin(a), z_top]
                    for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
    pc = apply_T(T_CB, rim)
    if (pc[:, 2] <= 0).any():
        return None
    px = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
    h_, w_ = gray.shape
    u0, v0 = np.clip(np.floor(px.min(0)).astype(int), 0, [w_ - 1, h_ - 1])
    u1, v1 = np.clip(np.ceil(px.max(0)).astype(int), 0, [w_ - 1, h_ - 1])
    if u1 <= u0 or v1 <= v0:
        return None
    uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    uv = np.column_stack([uu.ravel(), vv.ravel()]).astype(float)
    xy = _ray_plane(frame, T_BC, uv, z_top)[:, :2]
    rr = np.hypot(*(xy - c).T)
    sel = rr < g["outer_d"] / 2 - 0.004
    if sel.sum() < 500:
        return None
    xy, gv = xy[sel], gray[uv[sel, 1].astype(int), uv[sel, 0].astype(int)]
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    rad = np.hypot(*(xy - c).T)

    def contrast(ph):
        # distance to the nearest hole centre at this turn, from the polar coordinates (law of cosines)
        da = (ang - np.radians(ph) + np.pi / 6) % (np.pi / 3) - np.pi / 6
        d = np.sqrt(np.maximum(rad ** 2 + r_h ** 2 - 2 * rad * r_h * np.cos(da), 0.0))
        ins, top = d < 0.024, d > g["hole_d"] / 2 + 0.001
        if ins.sum() < 100 or top.sum() < 100:
            return -np.inf
        return float(np.mean(gv[top]) - np.mean(gv[ins]))

    phs = np.arange(-30.0, 30.0, step_deg)
    cs = np.array([contrast(ph) for ph in phs])
    i = int(np.argmax(cs))
    off = contrast(phs[i] + 30.0)
    return float(phs[i]), float(cs[i]), float(off)


def _from_outer_rim(frame, T_BC, XYZ, hsv, red, ok, z_top, g, prior, n_good, clean=()) -> dict:
    """centre from the outer rim; turn from the clean hole rims seen (clean: their centres), else from the top face;
    free holes from the red inside (see the module notes)"""
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
    r_h = float(prior.get("hole_circle_r_mm", g["hole_r"] * 1000)) / 1000
    # turn from the clean hole rims that lie on the hole circle around this centre (2026-11-17: from the overview the
    # camera's depth fills the 55 mm deep holes up to the top face (z median 0.057-0.063 inside all six), so the top
    # face test below found no turn in 2 of 4 pictures although 2 clean rims were seen; the rims give it directly)
    on = [h for h in clean if abs(np.hypot(*(np.asarray(h) - c)) - r_h) < 0.004]
    if on:
        ah = np.array([np.arctan2(h[1] - c[1], h[0] - c[0]) for h in on])
        phi = float(np.degrees(np.angle(np.mean(np.exp(6j * ah)))) / 6)  # -30..30 deg, as from the hole rims in locate_disc
        frac, how = None, f"turn {{:.1f}} deg from {len(on)} clean hole rims"
    else:
        phi, frac, how = None, None, None
    # else: the six hole circles at the phase that leaves the fewest top-face points inside them
    P = XYZ[ok]
    P = P[(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < g["outer_d"] / 2 - 0.003) & (np.abs(P[:, 2] - z_top) < 0.003)]
    if phi is None:
        sh = _turn_from_shading(frame, T_BC, c, r_h, z_top, g)
        if sh is not None and sh[1] >= 40.0 and sh[1] - sh[2] >= 25.0:
            phi, how = sh[0], f"turn {{:.1f}} deg from the shading (top face {sh[1]:.0f} gray brighter than the holes, {sh[2]:.0f} at 30 deg off)"
    if phi is None and not prior:
        return {"ok": False, "status": f"{n_good} clean hole rims on the hole circle, no clear shading and no earlier disc: the turn is unknown"}
    if phi is None:
        scores = []
        for ph in np.arange(0.0, 60.0, 0.5):
            a = np.radians(ph + 60.0 * np.arange(6))
            H = c + r_h * np.column_stack([np.cos(a), np.sin(a)])
            d = np.min(np.hypot(P[:, 0, None] - H[None, :, 0], P[:, 1, None] - H[None, :, 1]), axis=1)
            scores.append((float(np.mean(d < g["hole_d"] / 2 - 0.004)), ph))
        frac, phi = min(scores)
        if frac > 0.3:
            return {"ok": False, "status": f"{n_good} clean hole rims; no turn of the disc leaves its holes clear of the top face"}
        how = f"top face (turn {{:.1f}} deg, {frac * 100:.0f}% top points in the holes)"
    # numbering kept: the same hole keeps its number when the disc is turned by less than 30 deg
    p0 = float(prior.get("phase_deg", phi))
    phi = p0 + ((phi - p0 + 30) % 60 - 30)
    a = np.radians(phi + 60.0 * np.arange(6))
    H = c + r_h * np.column_stack([np.cos(a), np.sin(a)])
    red_frac = _red_fractions(frame, T_BC, red, H, z_top)
    by_in = _free_by_inside(XYZ, ok, H, z_top, red_frac)
    free = [i + 1 for i, hc in enumerate(H) if by_in[i] or any(np.hypot(*(np.asarray(h) - hc)) < 0.010 for h in clean)]
    return {"ok": True, "status": f"outer rim ({inl.mean() * 100:.0f}% of the outline over {cover} deg, rms {rms * 1000:.1f} mm) and "
            + how.format(phi) + f"; {n_good} clean hole rims", "centre": [float(c[0]), float(c[1])],
            "z_top": round(z_top, 4), "z_top_depth": round(z_top, 4), "free_holes": free, "phase_deg": round(float(phi), 1),
            "hole_circle_r_mm": round(r_h * 1000, 1), "holes_seen": [], "hole_radius_mm": [], "rim_rms_mm": [],
            "red_fraction": [round(v, 2) for v in red_frac], "holes": H.tolist(), "hole_bottom_z": round(z_top - g["hole_depth"], 4),
            "from_outer_rim": True}


def _top_z_by_colour(XYZ, ok, hsv, inarea) -> float | None:
    """the disc top's height without hole blobs: the most common height of the disc-coloured (red-orange) points in the
    work area above the table (the flat top face outnumbers the side wall at any one height)"""
    col = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 50)
    with np.errstate(invalid="ignore"):
        z = XYZ[ok & inarea & col & (XYZ[..., 2] > 0.015), 2]
    if len(z) < 2000:
        return None
    h, e = np.histogram(z, bins=np.arange(float(z.min()), float(z.max()) + 0.004, 0.002))
    z_pk = e[int(np.argmax(np.convolve(h, [1, 2, 1], "same")))] + 0.001
    return float(np.median(z[np.abs(z - z_pk) < 0.004]))


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
    def rim_route(n_good, clean, z_seen=None, why=""):
        """the outer rim and the turn from the clean rims / shading (_from_outer_rim): when the holes alone do not do"""
        z_ = float(prior["z_top"]) if prior else (z_seen if z_seen is not None else _top_z_by_colour(XYZ, ok, hsv, inarea))
        if z_ is None:
            return {"ok": False, "status": why + "; no disc top face height"}
        d_ = _from_outer_rim(frame, T_BC, XYZ, hsv, red, ok, z_, g, prior or {}, n_good, clean=clean)
        if not d_.get("ok") and why:
            d_["status"] = why + "; " + d_["status"]
        return d_

    if len(blobs) < (1 if prior else 3):
        return rim_route(0, [], why=f"only {len(blobs)} red hole-like blobs in the work area")
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
        return rim_route(0, [], why="no depth on the top face around the hole blobs")
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
    if len(good) >= 2:
        # holes of one disc lie within a hole-circle diameter of each other: keep the largest such group (2026-11-17: a
        # red blob 0.30 m away fit a 22.5 mm circle, rms 2.6 mm, and spoilt the disc with two real holes seen)
        C0 = np.array([hh["centre"] for hh in good])
        near = np.hypot(*(C0[:, None, :] - C0[None, :, :]).transpose(2, 0, 1)) <= 2 * g["hole_r"] + 0.008
        best = max(range(len(good)), key=lambda i: near[i].sum())
        good = [hh for i, hh in enumerate(good) if near[best, i]]
    if len(good) < 3:
        # the outer rim, the turn from the clean rims seen or the shading (2026-11-17: from the overview 2 of 4 pictures of
        # a new, empty disc had 2 clean rims, the near holes' insides running into the red side wall; after a restart
        # none, the top face as red as the holes)
        return rim_route(len(good), [hh["centre"] for hh in good], z_seen=z_top)
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
    # a hole is free when its red interior was seen (a plug in it shows soil and leaves instead): a clean rim, or mostly
    # red inside (2026-11-17: from the overview the near holes' insides run into the red side wall in the picture, no
    # clean rim; 3 of 6 empty holes were left unknown and not placed into; see _free_by_inside)
    red_frac = _red_fractions(frame, T_BC, red, np.array(all_h), z_top)
    by_in = _free_by_inside(XYZ, ok, np.array(all_h), z_top, red_frac)
    free = [i + 1 for i, h in enumerate(all_h) if np.hypot(*(C - np.array(h)).T).min() < 0.010 or by_in[i]]
    return {"ok": okk, "status": status, "centre": [float(c[0]), float(c[1])], "z_top": round(z_top, 4), "z_top_depth": round(z_depth, 4),
            "free_holes": free,
            "phase_deg": round(float(phase), 1), "hole_circle_r_mm": round(float(r_holes) * 1000, 1), "holes_seen": [[round(float(v), 4) for v in hh["centre"]] for hh in good],
            "hole_radius_mm": [round(hh["r"] * 1000, 1) for hh in good], "rim_rms_mm": [round(hh["rms_mm"], 1) for hh in good],
            "centre_residual_mm": [round(float(v) * 1000, 1) for v in rsd], "spacing_residual_deg": [round(float(v), 1) for v in dphase],
            "holes": all_h, "hole_bottom_z": round(z_top - g["hole_depth"], 4), "red_fraction": [round(v, 2) for v in red_frac]}


def taken_holes(disc: dict) -> list[int]:
    """the holes with a plant in them: "occupied_holes" (a plant seen above it, or put there by us; locate.py / run.py keep it)
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


def disc_absent(frame: Frame, T_BC: np.ndarray, prev: dict, table_z: float, radius: float = 0.09, margin_px: int = 20,
                min_points: int = 500, max_high_share: float = 0.03, above: float = 0.02, tray_cells: dict | None = None,
                tray_pad: float = 0.06) -> tuple[bool, str]:
    """(True, why) when the disc of the last record is gone: its whole footprint (radius, at its top height and at the
    table) in the picture and the points there at the table (at most max_high_share more than `above` over it).
    Hidden (an arm, leaves) or out of the picture: (False, why), the old record kept as an obstacle. tray_cells: the
    tray located in the same picture; its footprint (cell centres + tray_pad: half a cell, the rim and a stacked tray's
    flange reach ~55 mm past the outer cell centres, 2026-11-16) is left out (a tray moved onto part of where the disc
    stood). 2026-11-16: the
    disc of 11-06 was taken away; still in disc.json it stood between the robot and the tray and bent the paths around
    plants that were not there"""
    from .transforms import inv_T

    c = np.asarray(prev.get("centre") or [np.nan, np.nan], float)[:2]
    if not np.all(np.isfinite(c)):
        return False, "no earlier disc position"
    H, W = frame.rgb.shape[:2]
    T_CB = inv_T(T_BC)
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for z in (float(prev.get("z_top", table_z + 0.06)), table_z):
        ring = np.column_stack([c[0] + radius * np.cos(t), c[1] + radius * np.sin(t), np.full_like(t, z)])
        pc = apply_T(T_CB, ring)
        if (pc[:, 2] <= 0).any():
            return False, "the earlier disc position is behind the camera"
        uv = (frame.intr.K() @ (pc / pc[:, 2:3]).T).T[:, :2]
        if (uv[:, 0] < margin_px).any() or (uv[:, 1] < margin_px).any() or (uv[:, 0] > W - margin_px).any() or (uv[:, 1] > H - margin_px).any():
            return False, "the earlier disc position is not fully in the picture"
    ok = np.isfinite(frame.xyz).all(-1)
    P = apply_T(T_BC, frame.xyz[ok])
    inside = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < radius
    if tray_cells:
        C_ = np.array([v[:2] for v in tray_cells.values()], float)
        o = C_.mean(0)
        _, _, Vt = np.linalg.svd(C_ - o)
        A_ = (C_ - o) @ Vt.T
        lo, hi = A_.min(0) - tray_pad, A_.max(0) + tray_pad
        Q_ = (P[:, :2] - o) @ Vt.T
        inside &= ~np.all((Q_ >= lo) & (Q_ <= hi), axis=1)
    n = int(inside.sum())
    if n < min_points:
        return False, f"too few depth points where the disc was ({n})"
    high = float(np.mean(P[inside, 2] > table_z + above))
    if high > max_high_share:
        return False, f"something stands where the disc was ({high * 100:.0f}% of the points above the table)"
    return True, f"gone: the table where it was ({n} points, {high * 100:.1f}% above it)"


def load_disc(path=None) -> dict | None:
    """the disc record (calibration/disc.json), or None when there is none or the last picture saw it gone"""
    import json
    from pathlib import Path

    from .config import PROJECT_DIR

    p = Path(path) if path else PROJECT_DIR / "calibration" / "disc.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if d.get("absent") else d
