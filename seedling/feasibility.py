"""Before a run: can each plant be grasped, and each free hole be filled, with a flat hand in a natural posture?

A flat hand (pitch 0, camera above) reaches a point in a natural posture (base turned towards it, elbow not swung
sideways, wrist not flipped; seedling.cartesian.posture_cost within grasp.max_posture_deg) only in a band of distance
from the robot that depends on the height: 2026-11-04, straight ahead, the grasp and 6 cm back both natural from
0.80 m at a grasp height of 0.19 m, from 0.68 m at 0.34 m; nothing closer. Joint 6 at its limit decides it, so the
posture limit hardly matters (the alternatives need 110-155 deg). This check takes a few IK solutions per point (no
paths, no obstacles): the full plans still decide; this only tells early what cannot work and what to change.
"""
from __future__ import annotations

import numpy as np

from . import cartesian, kinematics
from .transforms import inv_T, make_T


def _flat_R(xy, dyaw_deg):
    yaw = np.arctan2(xy[1], xy[0]) + np.radians(dyaw_deg)
    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    return np.column_stack([np.cross(across, a), across, a]), a


def natural_flat(xy, z, F, G, max_posture: float, back_m: float = 0.06, headings=(0, -20, 20, -40, 40),
                 margin: float = 0.05, n_random: int = 16, workspace=None) -> bool:
    """a flat hand with its contact point at (xy, z) and back_m behind it, both in a natural posture, for some heading"""
    xy = np.asarray(xy, float)[:2]
    if workspace is not None and not all(workspace[ax][0] <= v <= workspace[ax][1] for ax, v in zip("xyz", (xy[0], xy[1], z))):
        return False  # the robot refuses targets outside robot.workspace_m
    for d in headings:
        R, a = _flat_R(xy, d)
        okk = True
        for b in (0.0, back_m):
            p = np.array([xy[0], xy[1], z]) - a * b
            sols = kinematics.ik_all(make_T(R, p) @ inv_T(G), F, limit_margin_rad=margin, n_random=n_random)
            if not any(cartesian.posture_cost(q, xy) <= max_posture for q in sols):
                okk = False
                break
        if okk:
            return True
    return False


def suggestion(xy, z, F, G, max_posture: float, workspace=None) -> str:
    """the smallest change (raised by dz, moved out from the robot by dr, 5 cm steps) that makes (xy, z) work"""
    xy = np.asarray(xy, float)[:2]
    u = xy / max(float(np.hypot(*xy)), 1e-9)
    quick = dict(headings=(0, -30, 30), n_random=12)
    for total in range(1, 11):  # 5 cm steps, smallest total change first; out from the robot or towards it
        for i in range(total + 1):
            for sgn in (1, -1):
                dz, dr = 0.05 * i, sgn * 0.05 * (total - i)
                if dz > 0.25 or abs(dr) > 0.30 or (sgn < 0 and total == i):
                    continue
                if natural_flat(xy + u * dr, z + dz, F, G, max_posture, workspace=workspace, **quick):
                    parts = ([f"raise it by about {dz * 100:.0f} cm"] if dz else []) + (
                        [f"move it about {abs(dr) * 100:.0f} cm {'further from' if dr > 0 else 'closer to'} the robot"] if dr else [])
                    return " and ".join(parts)
    return "no change up to 25 cm higher / 30 cm further helps"


def check_layout(grid: dict, disc: dict | None, plants, free_holes, cfg: dict, *, below_m: float = 0.07) -> dict:
    """{'plants': {cell: '' or what to change}, 'holes': {hole: '' or what to change}}"""
    from .config import ee_to_contact, flange_to_ee

    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    mp = min(float(cfg["grasp"].get("max_posture_deg", 60.0)), 60.0)  # "natural": what is reported as needing a twisted posture
    z_grasp = float(grid["z_top"]) + 0.04  # the grasp centre ~40 mm over the tray top (soil + closing height)
    ws = cfg["robot"]["workspace_m"]
    out = {"plants": {}, "holes": {}}
    for k in plants:
        out["plants"][k] = "" if natural_flat(grid["cells"][k][:2], z_grasp, F, G, mp, workspace=ws) else "not reachable flat"
    bad = [k for k, v in out["plants"].items() if v]
    if bad:  # what to change: worked out for the plant nearest the robot (the hardest one)
        k0 = min(bad, key=lambda k: np.hypot(*grid["cells"][k][:2]))
        s0 = suggestion(grid["cells"][k0][:2], z_grasp, F, G, mp, workspace=ws)
        out["tray_change"] = f"the tray: {s0} (worked out for {k0}, the nearest of them)"
    if disc is not None and disc.get("ok", True) and "holes" in disc:
        z_rel = float(disc["z_top"]) + below_m  # the held stem when the plug bottom is at the disc top
        for h in free_holes:
            out["holes"][h] = "" if natural_flat(disc["holes"][h - 1][:2], z_rel, F, G, mp, workspace=ws) else "not reachable flat"
        if any(out["holes"].values()):
            out["disc_change"] = f"the disc: {suggestion(disc['centre'][:2], z_rel, F, G, mp, workspace=ws)}"
    return out
