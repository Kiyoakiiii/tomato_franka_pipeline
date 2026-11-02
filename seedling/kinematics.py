"""FR3 forward kinematics and a numerical IK used only as a reachability pre-check.

The robot itself executes Cartesian motions with its own controller; this module
only rejects grasp candidates that are clearly outside the joint limits before
any motion is sent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import make_T

# Craig (modified) DH parameters for Panda/FR3: (a_{i-1}, d_i, alpha_{i-1})
_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
FLANGE_D = 0.107

# FR3 joint limits (rad) from the Franka FR3 datasheet.
Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

# Correction added to the REPORTED joint angles to get the ones this model needs (true geometry). Found by the
# wrist-camera calibration (scripts/35_wrist_calib_eval.py --refit): 2026-10-13 joint 5 is off by ~1.1 deg, which no
# fixed camera mount can absorb. Set with set_joint_offsets (calibration.load_T_FC does it); zero = nominal model.
JOINT_OFFSETS = np.zeros(7)


def set_joint_offsets(dq) -> None:
    JOINT_OFFSETS[:] = np.asarray(dq if dq is not None else np.zeros(7), dtype=float).reshape(7)


# Default flange -> EE for the Franka Hand as configured in Desk.
DEFAULT_F_T_EE = make_T(Rotation.from_euler("z", -np.pi / 4).as_matrix(), [0.0, 0.0, 0.1034])


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_flange(q: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """Flange pose for reported joint angles q (plus JOINT_OFFSETS, or the given offsets)."""
    q = np.asarray(q, dtype=float) + (JOINT_OFFSETS if offsets is None else np.asarray(offsets, dtype=float))
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, q):
        T = T @ _dh(a, d, alpha, theta)
    return T @ _dh(0.0, FLANGE_D, 0.0, 0.0)


def fk(q: np.ndarray, F_T_EE: np.ndarray = DEFAULT_F_T_EE) -> np.ndarray:
    return fk_flange(np.asarray(q, dtype=float)) @ F_T_EE


def _frames(q: np.ndarray, F_T_EE: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Joint frames (after each joint rotation) and the EE pose."""
    T = np.eye(4)
    frames = []
    for (a, d, alpha), theta in zip(_DH, np.asarray(q, dtype=float) + JOINT_OFFSETS):
        T = T @ _dh(a, d, alpha, theta)
        frames.append(T)
    return frames, T @ _dh(0.0, FLANGE_D, 0.0, 0.0) @ F_T_EE


def _jacobian(q: np.ndarray, target: np.ndarray, F_T_EE: np.ndarray, rot_weight: float) -> np.ndarray:
    """Geometric Jacobian of ``_residual`` (rotation part linearised around zero error)."""
    frames, T = _frames(q, F_T_EE)
    pe = T[:3, 3]
    J = np.zeros((6, 7))
    Rt = target[:3, :3].T
    for i, Fi in enumerate(frames):
        z = Fi[:3, 2]
        J[:3, i] = np.cross(z, pe - Fi[:3, 3])
        J[3:, i] = rot_weight * (Rt @ z)
    return J


def _residual(q: np.ndarray, target: np.ndarray, F_T_EE: np.ndarray, rot_weight: float) -> np.ndarray:
    T = fk(q, F_T_EE)
    dp = T[:3, 3] - target[:3, 3]
    dr = Rotation.from_matrix(target[:3, :3].T @ T[:3, :3]).as_rotvec()
    return np.concatenate([dp, rot_weight * dr])


def _dls(seed, target, F_T_EE, lo, hi, rot_weight, iters=150, lam=0.02, k_null=0.0):
    """Damped least squares, joints clipped to [lo, hi]. A null-space pull (k_null>0) slowed convergence
    together with clipping in tests, so it is off by default."""
    q = seed.copy()
    mid = (lo + hi) / 2
    for _ in range(iters):
        r = _residual(q, target, F_T_EE, rot_weight)
        if np.linalg.norm(r[:3]) < 1e-5 and np.linalg.norm(r[3:]) < 1e-5:
            break
        J = _jacobian(q, target, F_T_EE, rot_weight)
        JJt = J @ J.T + lam**2 * np.eye(6)
        Jp = J.T @ np.linalg.solve(JJt, np.eye(6))
        dq = -Jp @ r + (np.eye(7) - Jp @ J) @ (k_null * (mid - q))
        step = np.max(np.abs(dq))
        if step > 0.3:
            dq *= 0.3 / step
        q = np.clip(q + dq, lo, hi)
    return q


def solve_ik(
    target_T_BE: np.ndarray,
    q_seed: np.ndarray | None = None,
    F_T_EE: np.ndarray = DEFAULT_F_T_EE,
    limit_margin_rad: float = 0.05,
    pos_tol_m: float = 1e-3,
    rot_tol_deg: float = 0.5,
    n_random_seeds: int = 4,
    rng_seed: int = 0,
) -> np.ndarray | None:
    """Return joint angles reaching ``target_T_BE`` within limits (with margin), or None."""
    lo = Q_MIN + limit_margin_rad
    hi = Q_MAX - limit_margin_rad
    seeds = []
    if q_seed is not None:
        seeds.append(np.clip(np.asarray(q_seed, dtype=float), lo, hi))
    # structured seeds: ready pose and typical forward-reaching postures, turned toward the target azimuth
    az = float(np.arctan2(target_T_BE[1, 3], target_T_BE[0, 3]))
    for base in (Q_READY, [0.0, 0.2, 0.0, -2.2, 0.0, 2.6, 0.785], [0.0, 0.5, 0.0, -1.6, 0.0, 2.4, 0.785], [0.0, -0.3, 0.0, -2.6, 0.0, 3.2, 0.785]):
        q0 = np.array(base, dtype=float)
        q0[0] = az if base is not Q_READY else q0[0]
        seeds.append(np.clip(q0, lo, hi))
    rng = np.random.default_rng(rng_seed)
    seeds += [rng.uniform(lo, hi) for _ in range(n_random_seeds)]
    rot_weight = 0.2  # 1 rad ~ 0.2 m
    for seed in seeds:
        q = _dls(seed, target_T_BE, F_T_EE, lo, hi, rot_weight)
        res = _residual(q, target_T_BE, F_T_EE, rot_weight)
        if np.linalg.norm(res[:3]) <= pos_tol_m and np.degrees(np.linalg.norm(res[3:]) / rot_weight) <= rot_tol_deg:
            return q
    return None


def solve_ik_near(
    target_T_BE: np.ndarray,
    q_start: np.ndarray,
    F_T_EE: np.ndarray = DEFAULT_F_T_EE,
    limit_margin_rad: float = 0.05,
    steps: int = 10,
    pull: float = 0.03,
    pos_tol_m: float = 1e-3,
    rot_tol_deg: float = 0.5,
) -> np.ndarray | None:
    """IK that stays on the current branch: follow the straight Cartesian path from fk(q_start) to the target in
    small steps, each solved from the previous joints with a weak pull toward them (least joint change).
    Use this for the joint moves the robot actually executes; ``solve_ik`` may return a far-away solution."""
    from scipy.optimize import least_squares

    lo, hi = Q_MIN + limit_margin_rad, Q_MAX - limit_margin_rad
    q = np.clip(np.asarray(q_start, dtype=float), lo, hi)
    T0 = fk(q, F_T_EE)
    dR = Rotation.from_matrix(target_T_BE[:3, :3] @ T0[:3, :3].T).as_rotvec()
    rot_weight = 0.2
    for k in range(1, steps + 1):
        Tk = T0.copy()
        Tk[:3, :3] = Rotation.from_rotvec(dR * k / steps).as_matrix() @ T0[:3, :3]
        Tk[:3, 3] = T0[:3, 3] + (target_T_BE[:3, 3] - T0[:3, 3]) * k / steps
        q_prev = q.copy()
        q = least_squares(lambda x: np.concatenate([_residual(x, Tk, F_T_EE, rot_weight), pull * (x - q_prev)]), q, bounds=(lo, hi)).x
    r = _residual(q, target_T_BE, F_T_EE, rot_weight)
    if np.linalg.norm(r[:3]) <= pos_tol_m and np.degrees(np.linalg.norm(r[3:]) / rot_weight) <= rot_tol_deg:
        return q
    return None
