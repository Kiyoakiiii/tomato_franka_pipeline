"""Grasp candidate generation and ranking in the robot base frame.

Gripper contact frame G: z = approach direction, y = closing direction,
x = y cross z (approximately along the stem for a perpendicular grasp).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kinematics
from .config import ee_to_contact, flange_to_ee
from .stem3d import InstanceGeometry
from .transforms import apply_T, inv_T, make_T, rotate, unit


@dataclass
class GraspCandidate:
    instance: int
    seg_index: int
    yaw_offset_deg: float
    pitch_deg: float
    T_BG: np.ndarray
    T_BG_pre: np.ndarray
    p_B: np.ndarray
    s_B: np.ndarray
    diameter_m: float
    grasp_width_m: float
    feasible: bool = False
    reject_reason: str = ""
    clearance_points: int = 0
    score: float = -np.inf
    q_grasp: np.ndarray | None = None
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "instance": self.instance,
            "seg_index": self.seg_index,
            "yaw_offset_deg": self.yaw_offset_deg,
            "pitch_deg": self.pitch_deg,
            "p_B": np.round(self.p_B, 5).tolist(),
            "s_B": np.round(self.s_B, 4).tolist(),
            "diameter_mm": round(self.diameter_m * 1000, 2),
            "grasp_width_mm": round(self.grasp_width_m * 1000, 2),
            "feasible": self.feasible,
            "reject_reason": self.reject_reason,
            "clearance_points": self.clearance_points,
            "score": None if not np.isfinite(self.score) else round(float(self.score), 4),
        }


def gripper_frame(p: np.ndarray, s: np.ndarray, a: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Contact frame at p with approach a and closing axis perpendicular to both a and stem tangent s."""
    a = unit(a)
    c = np.cross(a, s)
    if np.linalg.norm(c) < 0.2:  # approach nearly parallel to the stem
        raise ValueError("approach direction too close to stem direction")
    c = sign * unit(c)
    x = np.cross(c, a)
    return make_T(np.column_stack([x, c, a]), p)


def approach_direction(p_B: np.ndarray, yaw_offset_deg: float, pitch_deg: float) -> np.ndarray:
    radial = np.array([p_B[0], p_B[1], 0.0])
    radial = unit(radial) if np.linalg.norm(radial) > 1e-3 else np.array([1.0, 0, 0])
    yaw = np.radians(yaw_offset_deg)
    h = np.array([np.cos(yaw) * radial[0] - np.sin(yaw) * radial[1], np.sin(yaw) * radial[0] + np.cos(yaw) * radial[1], 0.0])
    pitch = np.radians(pitch_deg)
    return unit(np.cos(pitch) * h + np.sin(pitch) * np.array([0, 0, -1.0]))


def finger_boxes_G(gcfg: dict, open_width: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned boxes (lo, hi) in G for the two finger pads (stationary pose)."""
    hh = float(gcfg["finger_half_height_m"])
    t = float(gcfg["finger_thickness_m"])
    dz = float(gcfg["finger_depth_m"])
    w2 = open_width / 2
    return [
        (np.array([-hh, w2, -dz]), np.array([hh, w2 + t, 0.01])),
        (np.array([-hh, -w2 - t, -dz]), np.array([hh, -w2, 0.01])),
    ]


def count_swept_collisions(scene_B: np.ndarray, T_BG: np.ndarray, gcfg: dict, open_width: float, exclude_mask: np.ndarray) -> int:
    if scene_B is None or len(scene_B) == 0:
        return 0
    P = apply_T(inv_T(T_BG), scene_B[~exclude_mask])
    back = float(gcfg["pregrasp_distance_m"])
    n = 0
    for lo, hi in finger_boxes_G(gcfg, open_width):
        lo2 = lo.copy()
        lo2[2] -= back  # sweep from pre-grasp to grasp along -z
        n += int(np.all((P >= lo2) & (P <= hi), axis=1).sum())
    return n


def lowest_finger_z(T_BG: np.ndarray, gcfg: dict, open_width: float) -> float:
    zs = []
    for lo, hi in finger_boxes_G(gcfg, open_width):
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        zs.append(apply_T(T_BG, corners)[:, 2].min())
    return float(min(zs))


def generate_candidates(
    geoms: list[InstanceGeometry],
    T_BC: np.ndarray,
    cfg: dict,
    scene_C: np.ndarray | None = None,
    q_seed: np.ndarray | None = None,
    check_ik: bool = True,
    max_feasible: int = 3,
) -> list[GraspCandidate]:
    g = cfg["grasp"]
    gr = cfg["gripper"]
    ws = cfg["robot"]["workspace_m"]
    scene_B = apply_T(T_BC, scene_C) if scene_C is not None and len(scene_C) else None
    open_w = float(gr["open_width_m"])
    out: list[GraspCandidate] = []
    for geo in geoms:
        for seg in geo.segments:
            if not seg.ok:
                continue
            p_B = apply_T(T_BC, np.array(seg.p_C))
            s_B = unit(rotate(T_BC, np.array(seg.dir_C)))
            if s_B[2] < 0:
                s_B = -s_B
            tilt = float(np.degrees(np.arccos(np.clip(s_B[2], -1, 1))))
            diam = float(np.clip(seg.diameter_m or g["default_stem_diameter_m"], 0.0015, g["max_stem_diameter_m"]))
            gw = float(gr["grasp_width_m"]) or float(gr["empty_close_width_m"]) + 0.8 * diam
            excl = None
            if scene_B is not None:
                r = scene_B - p_B
                excl = np.linalg.norm(r - np.outer(r @ s_B, s_B), axis=1) < max(diam, 0.004) * 1.5 + 0.004
            for yaw in g["approach_yaw_offsets_deg"]:
                for pitch in g["approach_pitch_deg"]:
                    a = approach_direction(p_B, float(yaw), float(pitch))
                    try:
                        sign = 1.0
                        T_BG = gripper_frame(p_B, s_B, a, sign)
                    except ValueError as exc:
                        out.append(GraspCandidate(geo.instance, seg.seg_index, yaw, pitch, make_T(), make_T(), p_B, s_B, diam, gw, reject_reason=str(exc)))
                        continue
                    # prefer the wrist roll with hand x pointing up; the flipped roll is tried if IK fails
                    if T_BG[2, 0] < 0:
                        sign = -1.0
                        T_BG = gripper_frame(p_B, s_B, a, sign)
                    T_pre = _pre(T_BG, a, g)
                    cand = GraspCandidate(geo.instance, seg.seg_index, float(yaw), float(pitch), T_BG, T_pre, p_B, s_B, diam, gw)
                    cand.details = {"tilt_deg": tilt, "seg_score": seg.score}
                    out.append(cand)
                    if tilt > float(g["max_stem_tilt_deg"]):
                        cand.reject_reason = f"stem tilt {tilt:.0f} deg"
                        continue
                    bad_ws = [ax for i, ax in enumerate("xyz") for pt in (p_B, T_pre[:3, 3]) if not ws[ax][0] <= pt[i] <= ws[ax][1]]
                    if bad_ws:
                        cand.reject_reason = f"outside workspace ({','.join(sorted(set(bad_ws)))})"
                        continue
                    if p_B[2] < float(g["table_z_m"]) + float(g["min_contact_height_m"]):
                        cand.reject_reason = "contact too close to tray surface"
                        continue
                    if lowest_finger_z(T_BG, g, open_w) < float(g["table_z_m"]):
                        cand.reject_reason = "finger would hit tray surface"
                        continue
                    if scene_B is not None:
                        cand.clearance_points = count_swept_collisions(scene_B, T_BG, g, open_w, excl)
                        if cand.clearance_points > int(g["clearance_points_max"]):
                            cand.reject_reason = f"approach blocked ({cand.clearance_points} pts)"
                            continue
                    cand.details.update({"a": a, "sign": sign})
                    cand.feasible = True  # geometric checks passed; IK is checked lazily below
                    height = p_B[2] - float(g["table_z_m"])
                    cand.score = (
                        seg.score
                        - 0.3 * abs(float(yaw)) / 30.0
                        - 0.2 * float(pitch) / 30.0
                        - 0.5 * cand.clearance_points / max(int(g["clearance_points_max"]), 1)
                        - 1.0 * height / 0.10  # lower on the stem is better for extraction
                        - 0.5 * tilt / 30.0
                    )
    out.sort(key=lambda c: c.score, reverse=True)
    if check_ik:
        _check_ik_lazily(out, cfg, q_seed, max_feasible)
    return out


def plan_poses(T_BG: np.ndarray, cfg: dict) -> list[tuple[str, np.ndarray]]:
    """Contact-frame poses of the whole pick-and-place sequence, keeping the grasp orientation."""
    g, t = cfg["grasp"], cfg["task"]
    a = T_BG[:3, 2]
    seq = [("pregrasp", _pre(T_BG, a, g)), ("grasp", T_BG)]

    def at(p):
        T = T_BG.copy()
        T[:3, 3] = p
        return T

    top = T_BG[:3, 3] + [0, 0, float(t["extraction_height_m"])]
    seq.append(("extract_top", at(top)))
    seq.append(("transport_lift", at([top[0], top[1], max(top[2], float(t["transport_height_m"]))])))
    for i, wp in enumerate(t.get("transport_waypoints_m") or []):
        seq.append((f"waypoint_{i+1}", at(wp)))
    place = np.array(t["place_position_m"], dtype=float)
    seq.append(("above_place", at(place + [0, 0, float(t["place_approach_height_m"])])))
    seq.append(("place", at(place)))
    back = place - a * float(g["pregrasp_distance_m"])
    seq.append(("place_retreat", at(back)))
    seq.append(("place_up", at([back[0], back[1], max(back[2], float(t["transport_height_m"]))])))
    return seq


def check_plan_ik(T_BG: np.ndarray, cfg: dict, q_seed=None) -> tuple[np.ndarray | None, str]:
    """IK for every pose of the sequence (seeded along the chain). Returns (q_grasp, failing_pose_name)."""
    F_T_EE = flange_to_ee(cfg)
    T_GE = inv_T(ee_to_contact(cfg))
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    ws = cfg["robot"]["workspace_m"]
    q = q_seed
    q_grasp = None
    for k, (name, T) in enumerate(plan_poses(T_BG, cfg)):
        p = T[:3, 3]
        if not all(ws[ax][0] <= p[i] <= ws[ax][1] for i, ax in enumerate("xyz")):
            return None, f"{name} outside workspace"
        # first pose: search more widely; later poses are small moves seeded from the previous solution
        q = kinematics.solve_ik(T @ T_GE, q, F_T_EE, margin, n_random_seeds=6 if k == 0 else 2)
        if q is None:
            return None, name
        if name == "grasp":
            q_grasp = q
    return q_grasp, ""


def _check_ik_lazily(cands: list[GraspCandidate], cfg: dict, q_seed, max_feasible: int) -> None:
    """IK-verify the full motion plan of candidates in rank order until ``max_feasible`` are found."""
    n_ok = 0
    for cand in cands:
        if not cand.feasible:
            continue
        if n_ok >= max_feasible:
            cand.feasible = False
            cand.reject_reason = "lower rank (IK not evaluated)"
            continue
        a, sign = cand.details["a"], cand.details["sign"]
        why = ""
        for T_try in (cand.T_BG, gripper_frame(cand.p_B, cand.s_B, a, -sign)):
            q, why = check_plan_ik(T_try, cfg, q_seed)
            if q is not None:
                cand.T_BG, cand.T_BG_pre, cand.q_grasp = T_try, _pre(T_try, a, cfg["grasp"]), q
                break
        if cand.q_grasp is None:
            cand.feasible = False
            cand.reject_reason = f"no IK solution ({why})"
            cand.score = -np.inf
        else:
            n_ok += 1
    cands.sort(key=lambda c: (c.feasible, c.score), reverse=True)


def _pre(T_BG: np.ndarray, a: np.ndarray, g: dict) -> np.ndarray:
    T = T_BG.copy()
    T[:3, 3] = T_BG[:3, 3] - a * float(g["pregrasp_distance_m"])
    return T


def scene_points(frame, stride: int = 6, max_range_m: float = 1.5) -> np.ndarray:
    P = frame.xyz[::stride, ::stride].reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    return P[(P[:, 2] > 0.05) & (P[:, 2] < max_range_m)]


def housing_lowest_z(T_BG: np.ndarray, T_FC: np.ndarray, cfg: dict) -> float:
    """Lowest point of the wrist camera housing when the contact frame is at T_BG."""
    from .wrist import housing_corners_C

    T_BC = T_BG @ inv_T(ee_to_contact(cfg)) @ inv_T(flange_to_ee(cfg)) @ T_FC
    return float(apply_T(T_BC, housing_corners_C(cfg))[:, 2].min())


def candidate_at(p_B: np.ndarray, s_B: np.ndarray, cfg: dict, q_seed=None, approach_options: list[tuple[float, float]] | None = None,
                 diameter_m: float | None = None, label: tuple[int, int] = (-1, -1), T_FC: np.ndarray | None = None) -> GraspCandidate:
    """Grasp at a known stem point (e.g. a tray cell): try approach (yaw offset, pitch) options in order and
    return the first whose whole pick-and-place sequence is reachable; otherwise a rejected candidate.
    With a wrist camera (T_FC), the camera housing must stay above the table at the grasp and pre-grasp poses."""
    g, gr = cfg["grasp"], cfg["gripper"]
    p_B = np.asarray(p_B, dtype=float)
    s_B = unit(s_B)
    if s_B[2] < 0:
        s_B = -s_B
    diam = float(diameter_m or g["default_stem_diameter_m"])
    gw = float(gr["grasp_width_m"]) or float(gr["empty_close_width_m"]) + 0.8 * diam
    opts = approach_options or [(float(y), float(p)) for p in g["approach_pitch_deg"] for y in g["approach_yaw_offsets_deg"]]
    last = None
    for yaw, pitch in opts:
        a = approach_direction(p_B, yaw, pitch)
        try:
            T1 = gripper_frame(p_B, s_B, a, 1.0)
        except ValueError as exc:
            last = GraspCandidate(label[0], label[1], yaw, pitch, make_T(), make_T(), p_B, s_B, diam, gw, reject_reason=str(exc))
            continue
        sign = 1.0 if T1[2, 0] >= 0 else -1.0
        cand = GraspCandidate(label[0], label[1], yaw, pitch, gripper_frame(p_B, s_B, a, sign), make_T(), p_B, s_B, diam, gw)
        cand.T_BG_pre = _pre(cand.T_BG, a, g)
        cand.details = {"a": a, "sign": sign}
        last = cand
        if lowest_finger_z(cand.T_BG, g, float(gr["open_width_m"])) < float(g["table_z_m"]):
            cand.reject_reason = "finger would hit tray surface"
            continue
        why = ""
        for T_try in (cand.T_BG, gripper_frame(p_B, s_B, a, -sign)):
            if T_FC is not None:
                hz = min(housing_lowest_z(T_try, T_FC, cfg), housing_lowest_z(_pre(T_try, a, g), T_FC, cfg))
                if hz < float(g["table_z_m"]) + 0.01:
                    why = f"camera housing would hit the table (z={hz:.3f})"
                    continue
            q, why = check_plan_ik(T_try, cfg, q_seed)
            if q is not None:
                cand.T_BG, cand.T_BG_pre, cand.q_grasp = T_try, _pre(T_try, a, g), q
                cand.feasible, cand.reject_reason, cand.score = True, "", 0.0
                return cand
        cand.reject_reason = f"no IK solution ({why})"
    return last
