"""Wrist-camera observation poses: where to put the arm so the camera looks at a tray cell, safely.

The camera is calibrated relative to the flange (T_FC, scripts 32/33). For a target point (a cell's grasp
point) the camera is placed on a ray through the target: ``distance_m`` away, looking down by an
elevation angle, from an azimuth around the base->target direction, with a roll about the optical axis.
Candidates are tried in configuration order; the first one that is reachable from the current joints
without a large joint change and whose whole joint path keeps the arm body (camera housing, flange,
gripper contact point, wrist joints) above the floor is used.

The camera must be stationary when the image is taken: T_BC = T_BF(now) @ T_FC is only valid then.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import kinematics
from .config import ee_to_contact, flange_to_ee
from .transforms import apply_T, inv_T, make_T, unit


def housing_corners_C(cfg: dict) -> np.ndarray:
    """Corners of the (padded) camera housing box in the camera frame."""
    (x0, x1), (y0, y1), (z0, z1) = cfg["camera"]["housing_box_m"]
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)


_LINK_POINTS = None


def link_surface_points(frames) -> np.ndarray:
    """Surface points of links 3-7 (Franka's own FR3 collision meshes, franka_description meshes/robots/fr3/collision,
    thinned to ~1 cm) in the base frame. Until 2026-11-04 only the wrist joint centres were checked (+ a fixed
    clearance): the forearm/wrist housing (link 5) then touched a disc it was modelled 10 cm above."""
    global _LINK_POINTS
    if _LINK_POINTS is None:
        from pathlib import Path

        d = np.load(Path(__file__).with_name("fr3_collision_points.npz"))
        _LINK_POINTS = [np.asarray(d[f"link{i}"], float) for i in range(1, 8)]
    return np.vstack([(frames[i - 1][:3, :3] @ _LINK_POINTS[i - 1].T).T + frames[i - 1][:3, 3] for i in range(3, 8)])


def body_points_B(q: np.ndarray, T_FC: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Points of the arm that must stay clear of what is around, in the base frame: the surface of links 3-7 ("arm",
    the official collision meshes), the flange, the gripper contact point, the camera housing."""
    F_T_EE = flange_to_ee(cfg)
    frames, T_BE = kinematics._frames(np.asarray(q, float), F_T_EE)
    T_BF = T_BE @ inv_T(F_T_EE)
    pts = {"arm": link_surface_points(frames), "flange": T_BF[:3, 3][None]}
    if cfg["gripper"].get("type", "none") != "none":
        pts["gripper contact point"] = (T_BE @ ee_to_contact(cfg))[:3, 3][None]
    if T_FC is not None:
        pts["camera housing"] = apply_T(T_BF @ T_FC, housing_corners_C(cfg))
    return pts


def self_clearance(q: np.ndarray, T_FC: np.ndarray, cfg: dict) -> float:
    """Smallest distance from the hand (flange, camera housing, gripper contact point) to the axes of the arm's base,
    upper arm and forearm. Franka's self-collision reflex stopped two moves ending at 69 mm (2026-11-02, D4/D3:
    the wrist folded back onto the forearm, joint 4 at -2.85); the poses that ran fine had 100 mm or more. At most
    ~139 mm (the flange's fixed distance from the wrist joint, where the forearm axis ends): only a folded wrist is less."""
    frames, _ = kinematics._frames(np.asarray(q, float), flange_to_ee(cfg))
    P = [np.zeros(3)] + [f[:3, 3] for f in frames]
    pts = body_points_B(q, T_FC, cfg)
    hand = np.vstack([pts[k] for k in ("flange", "camera housing", "gripper contact point") if k in pts])
    d = np.inf
    for a, b in ((P[0], P[2]), (P[2], P[4]), (P[4], P[5])):  # base -> shoulder, upper arm, forearm (elbow -> wrist)
        ab = b - a
        t = np.clip(((hand - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)[:, None]
        d = min(d, float(np.linalg.norm(hand - (a + t * ab), axis=1).min()))
    return d


def path_self_clearance(qa: np.ndarray, qb: np.ndarray, T_FC: np.ndarray, cfg: dict, n: int = 21) -> float:
    """self_clearance along the straight joint path qa -> qb (what move_joints executes)."""
    return min(self_clearance(qa + s * (qb - qa), T_FC, cfg) for s in np.linspace(0.0, 1.0, n))


class Surfaces:
    """What lies under a point: the tray top within the tray (the rectangle of its cell centres + half a pitch + 15 mm
    rim, square corners), the disc top within the disc (+1 cm), else the table. From the located tray grid and disc
    (calibration/*.json)."""

    def __init__(self, grid: dict, disc: dict | None, cfg: dict, disc_keepout_m: float = 0.0):
        """disc_keepout_m: over the disc (and within reach of it) nothing may come lower than this above its top
        (the grasp: the hand has no business there; 2026-11-04 the flat hand reaching over a disc in front of the
        tray touched it, the wrist centres 10 cm above its top)"""
        import cv2

        self.tray_top = float(grid["z_top"])
        self.table_z = float(grid.get("table_z", self.tray_top - 0.08))
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.array([v[:2] for v in grid["cells"].values()], np.float32))
        a = np.radians(ang)
        self.c, self.R = np.array([cx, cy]), np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        edge = float(cfg["tray"]["model"]["pitch_m"]) / 2 + 0.015
        self.half = np.array([w / 2 + edge, h / 2 + edge])
        ok = disc is not None and disc.get("ok", True) and "centre" in disc
        self.disc_top = float(disc["z_top"]) + disc_keepout_m if ok else None
        self.disc_c = np.array(disc["centre"][:2]) if ok else None
        self.disc_r = 0.085 + 0.01

    def under(self, P: np.ndarray, reach: float = 0.0, extra_off_tray: float = 0.0) -> np.ndarray:
        """the highest surface within `reach` (horizontally) of each point; `extra_off_tray` is added to the disc and
        table tops (a larger clearance there than over the tray, see grasp.side_wrist_clearance_off_tray_mm)"""
        P = np.atleast_2d(P)[:, :2]
        uv = (P - self.c) @ self.R  # tray axes
        f = np.where((np.abs(uv) <= self.half + reach).all(axis=1), self.tray_top, self.table_z + extra_off_tray)
        if self.disc_c is not None:
            f = np.where(np.hypot(*(P - self.disc_c).T) < self.disc_r + reach, np.maximum(f, self.disc_top + extra_off_tray), f)
        return f


def lowest_body_z(q: np.ndarray, T_FC: np.ndarray, cfg: dict) -> tuple[float, str]:
    best = (np.inf, "")
    for name, P in body_points_B(q, T_FC, cfg).items():
        z = float(P[:, 2].min())
        if z < best[0]:
            best = (z, name)
    return best


def path_clearance(q_from: np.ndarray, q_to: np.ndarray, T_FC: np.ndarray, cfg: dict, n: int = 31) -> tuple[float, str]:
    """Lowest body point along the straight joint-space path (what a JointMotion follows, up to timing)."""
    worst = (np.inf, "")
    for s in np.linspace(0, 1, n):
        z = lowest_body_z(q_from + s * (q_to - q_from), T_FC, cfg)
        if z[0] < worst[0]:
            worst = z
    return worst


def look_from(target_B: np.ndarray, azimuth_rad: float, elevation_deg: float, distance_m: float, roll_deg: float) -> np.ndarray:
    """T_BC of a camera ``distance_m`` from the target, looking at it from horizontal direction ``azimuth``
    (camera on the side opposite to that direction) and ``elevation`` below horizontal, image x roughly horizontal."""
    e = np.radians(elevation_deg)
    h = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
    z = unit(np.cos(e) * h + np.sin(e) * np.array([0, 0, -1.0]))  # viewing direction
    eye = np.asarray(target_B, float) - distance_m * z
    x = unit(np.cross(z, [0, 0, 1.0])) if abs(z[2]) < 0.999 else np.array([0, -1.0, 0])
    y = np.cross(z, x)
    T = make_T(np.column_stack([x, y, z]), eye)
    return T @ make_T(Rotation.from_euler("z", roll_deg, degrees=True).as_matrix())


@dataclass
class ObservationPlan:
    ok: bool
    q: np.ndarray | None = None
    T_BC: np.ndarray | None = None
    desc: str = ""
    max_joint_step_deg: float = 0.0
    lowest_z: float = 0.0
    lowest_part: str = ""
    tried: list | None = None


JOINT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 1.0])  # big joints sweep the whole arm; joint 7 counts fully (big wrist rolls strain the camera cable)


def plan_observation(target_B: np.ndarray, q_now: np.ndarray, T_FC: np.ndarray, floor_z: float, cfg: dict,
                     _global_ik: bool | None = None, valid=None) -> ObservationPlan:
    """(see below) First only the IK solutions reached from the current joints; the global IK (random starts) for the
    views that have none only when that finds no pose at all (2026-11-04: 1.7 s instead of 6.5 s per cell; the far
    cells still need the global one)"""
    if _global_ik is None:
        pl = plan_observation(target_B, q_now, T_FC, floor_z, cfg, _global_ik=False, valid=valid)
        return pl if pl.ok else plan_observation(target_B, q_now, T_FC, floor_z, cfg, _global_ik=True, valid=valid)
    return _plan_observation(target_B, q_now, T_FC, floor_z, cfg, _global_ik, valid)


def _plan_observation(target_B: np.ndarray, q_now: np.ndarray, T_FC: np.ndarray, floor_z: float, cfg: dict,
                      global_ik: bool, valid=None) -> ObservationPlan:
    """For the first elevation (config order) that has any valid candidate: the azimuth/roll candidate with the
    least weighted joint change that is reachable and keeps the arm body above ``floor_z`` (or, if it starts
    below, never lower than it starts) along the whole joint path. With observe.all_elevations, every elevation
    is tried and the least joint change wins overall (plus observe.elevation_penalty_deg per place down the
    elevation list, so the preferred view still wins a near tie)."""
    o = cfg["tray"]["observe"]
    F_T_EE = flange_to_ee(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    ws = cfg["robot"]["workspace_m"]
    radial = np.arctan2(target_B[1], target_B[0])  # camera between the robot and the target, looking outward
    start_low = lowest_body_z(q_now, T_FC, cfg)[0]
    floor = min(floor_z, start_low - 1e-3)
    tried = []
    overall = None
    for ie, elev in enumerate(o["elevation_deg"]):
        best = None
        for az in o["azimuth_offsets_deg"]:
            for roll in o["roll_deg"]:
                desc = f"elev {elev:g} deg, azimuth {az:+g} deg, roll {roll:+g} deg"
                T_BC = look_from(target_B, radial + np.radians(az), float(elev), float(o["distance_m"]), float(roll))
                T_BE = T_BC @ inv_T(T_FC) @ F_T_EE
                q = kinematics.solve_ik_near(T_BE, q_now, F_T_EE, margin)
                if q is None and global_ik:
                    q = kinematics.solve_ik(T_BE, q_now, F_T_EE, margin)
                if q is None:
                    tried.append((desc, "no IK"))
                    continue
                bad = [j for j, (lo, hi) in (o.get("joint_range_rad") or {}).items() if not lo <= q[int(j) - 1] <= hi]
                if bad:
                    tried.append((desc, f"joint {bad[0]} outside the calibrated range"))
                    continue
                if valid is not None and not valid(q):  # the caller's condition (e.g. clear of the plants in the disc)
                    tried.append((desc, "not clear (the caller's world)"))
                    continue
                p_G = apply_T(T_BE @ ee_to_contact(cfg), np.zeros(3))
                out = [ax for i, ax in enumerate("xyz") if not ws[ax][0] <= p_G[i] <= ws[ax][1]]
                if out:
                    tried.append((desc, f"gripper contact point outside workspace ({','.join(out)})"))
                    continue
                step = float(np.degrees(np.abs(q - q_now)).max())
                if step > float(o["max_joint_step_deg"]):
                    tried.append((desc, f"joint change {step:.0f} deg"))
                    continue
                cost = float(np.degrees(np.abs(q - q_now) * JOINT_WEIGHTS).max())
                if best is not None and cost >= best[0]:
                    tried.append((desc, f"valid IK, larger joint change ({step:.0f} deg)"))
                    continue
                z, part = path_clearance(q_now, q, T_FC, cfg)
                if z < floor:
                    tried.append((desc, f"{part} would go down to z={z:.3f} m"))
                    continue
                sc = path_self_clearance(q_now, q, T_FC, cfg)
                if sc < float(o.get("min_self_clearance_m", 0.09)):  # Franka's self-collision reflex (see self_clearance)
                    tried.append((desc, f"hand {sc * 1000:.0f} mm from the arm itself"))
                    continue
                if best is not None:
                    tried.append((best[1].desc, f"valid, replaced by a smaller joint change ({best[1].max_joint_step_deg:.0f} deg)"))
                best = (cost, ObservationPlan(True, q, T_BC, desc, step, z, part))
        if best is not None:
            if not o.get("all_elevations"):
                best[1].tried = tried
                return best[1]
            c = best[0] + ie * float(o.get("elevation_penalty_deg", 5.0))
            if overall is None or c < overall[0]:
                overall = (c, best[1])
    if overall is not None:
        overall[1].tried = tried
        return overall[1]
    return ObservationPlan(False, desc="no safe observation pose", tried=tried)
