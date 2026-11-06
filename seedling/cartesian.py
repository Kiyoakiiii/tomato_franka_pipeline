"""Predictable motions: the hand moves along straight lines between given poses (as a person would teach it), the
joints follow by continuation from where the arm is (no jump between joint-solution families), and each straight
piece is executed as a few joint waypoints whose joint-space interpolation stays within ``max_dev_m`` of the line.

2026-11-04: the search over all joint solutions and via poses (seedling.motion) found collision-free but strange
motions (the base turned 70 deg away from the plant, the hand swung low beside the robot) and, with a too coarse
arm model, put the forearm onto the disc. Here the shape of the motion is fixed; the checks only accept or refuse it.
"""
from __future__ import annotations

import numpy as np

from . import kinematics
from .transforms import inv_T, make_T


def interpolate(Ta: np.ndarray, Tb: np.ndarray, step_m: float = 0.02, step_rad: float = np.radians(3.0)) -> list[np.ndarray]:
    """Poses from Ta to Tb (Tb included, Ta not): position on the straight line, attitude by slerp"""
    d = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
    rv = kinematics._rotvec(Tb[:3, :3] @ Ta[:3, :3].T)  # slerp: the relative rotation's vector scaled (no scipy: fast)
    ang = float(np.linalg.norm(rv))
    n = max(1, int(np.ceil(max(d / step_m, ang / step_rad))))
    return [make_T(kinematics._exp(rv * t) @ Ta[:3, :3], Ta[:3, 3] + t * (Tb[:3, 3] - Ta[:3, 3])) for t in np.linspace(0, 1, n + 1)[1:]]


def follow(q0: np.ndarray, poses_G: list[np.ndarray], F: np.ndarray, G: np.ndarray, margin: float,
           max_step_rad: float = np.radians(12.0)):
    """Joints for each contact-point pose in turn, each solved from the previous joints (least change). None if one
    has no solution or the joints jump (more than max_step_rad between neighbours: a family switch)."""
    out, q = [], np.asarray(q0, float)
    for T in poses_G:
        qn = kinematics.solve_ik_near(T @ inv_T(G), q, F, margin, steps=2)
        if qn is None or np.abs(qn - q).max() > max_step_rad:
            return None
        out.append(qn)
        q = qn
    return out


def tilt_deg(q: np.ndarray, F: np.ndarray, G: np.ndarray) -> float:
    """how far the hand is from flat: the angle between the hand's x axis (vertical when the fingers are level: the
    approach and the closing axis both horizontal) and the vertical, either way up"""
    R = (kinematics.fk(np.asarray(q, float), F) @ G)[:3, :3]
    return float(np.degrees(np.arccos(min(1.0, abs(float(R[2, 0]))))))


def max_tilt_between(qa, qb, F, G, step_deg: float = 1.5) -> float:
    """the largest tilt on the joint-space straight move qa -> qb, sampled every step_deg of the largest joint change"""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    n = max(3, int(np.ceil(np.degrees(np.abs(qb - qa).max()) / step_deg)) + 1)
    return max(tilt_deg(qa + t * (qb - qa), F, G) for t in np.linspace(0, 1, n))


def sparsify(q0: np.ndarray, qs: list[np.ndarray], poses_G: list[np.ndarray], F: np.ndarray, G: np.ndarray,
             max_dev_m: float = 0.005, max_tilt: float | None = None) -> list[np.ndarray]:
    """Fewest joint waypoints (from qs) such that the contact point of every joint-space straight piece stays within
    max_dev_m of the straight line between its ends (one straight Cartesian piece per call), and, with max_tilt (deg),
    the hand within that of flat all along (carrying a plant: the joint interpolation between two flat poses can tip
    the hand far over, 2026-11-05: up to 90 deg)"""
    pts = [kinematics.fk(q0, F) @ G] + list(poses_G)
    allq = [np.asarray(q0, float)] + list(qs)
    p_a, p_b = pts[0][:3, 3], pts[-1][:3, 3]
    ab = p_b - p_a
    L = float(np.linalg.norm(ab))

    def dev(qa, qb):
        worst = 0.0
        for t in np.linspace(0, 1, 11)[1:-1]:
            p = (kinematics.fk(qa + t * (qb - qa), F) @ G)[:3, 3]
            if L < 1e-6:
                d = float(np.linalg.norm(p - p_a))
            else:
                d = float(np.linalg.norm(np.cross(p - p_a, ab)) / L)
            worst = max(worst, d)
        return worst

    out, i = [], 0
    while i < len(allq) - 1:
        j = len(allq) - 1
        while j > i + 1 and (dev(allq[i], allq[j]) > max_dev_m
                             or (max_tilt is not None and max_tilt_between(allq[i], allq[j], F, G) > max_tilt)):
            j -= 1
        out.append(allq[j])
        i = j
    return out


def chain(q0: np.ndarray, targets_G: list[tuple[str, np.ndarray]], F: np.ndarray, G: np.ndarray, margin: float,
          max_dev_m: float = 0.005, max_tilt: float | None = None, step_m: float = 0.02, step_rad: float = np.radians(3.0),
          sparse: bool = True, max_jump_rad: float = np.radians(12.0)):
    """Straight pieces from the current contact pose through the named target poses. Returns
    [(name, dense joints, waypoints), ...] (each piece's dense joints for checking, its waypoints for executing), or
    (None, reason)."""
    T = kinematics.fk(q0, F) @ G
    q = np.asarray(q0, float)
    pieces = []
    for name, Tt in targets_G:
        poses = interpolate(T, Tt, step_m, step_rad)
        qs = follow(q, poses, F, G, margin, max_jump_rad)
        if qs is None:
            return None, f"{name}: no continuous joint solution along the straight line"
        pieces.append((name, qs, sparsify(q, qs, poses, F, G, max_dev_m, max_tilt) if sparse else qs[-1:]))
        q, T = qs[-1], Tt
    return pieces, ""


def natural_posture(q: np.ndarray, target_xy: np.ndarray, max_base_deg: float = 35.0, max_twist_deg: float = 50.0) -> str:
    """'' if q is a posture a person would use for a target at target_xy: the base turned towards it (joint 1 within
    max_base_deg of its direction), the elbow not swung sideways and the wrist not flipped (joints 3 and 5 within
    max_twist_deg of 0); else what is wrong"""
    d1 = (q[0] - np.arctan2(target_xy[1], target_xy[0]) + np.pi) % (2 * np.pi) - np.pi
    bad = []
    if abs(d1) > np.radians(max_base_deg):
        bad.append(f"base turned {np.degrees(d1):+.0f} deg from the plant")
    if abs(q[2]) > np.radians(max_twist_deg):
        bad.append(f"elbow swung sideways (joint 3 {np.degrees(q[2]):+.0f} deg)")
    if abs(q[4]) > np.radians(max_twist_deg):
        bad.append(f"wrist flipped (joint 5 {np.degrees(q[4]):+.0f} deg)")
    return ", ".join(bad)


def posture_cost(q: np.ndarray, target_xy: np.ndarray) -> float:
    """How far q is from a posture a person would use for a target at target_xy, in deg: the largest of the base's
    turn away from the target direction, the elbow's sideways swing (joint 3) and the wrist's twist (joint 5)"""
    d1 = (q[0] - np.arctan2(target_xy[1], target_xy[0]) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(max(abs(d1), abs(q[2]), abs(q[4]))))
