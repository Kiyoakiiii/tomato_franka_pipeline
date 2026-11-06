"""FR3 forward kinematics and a numerical IK used only as a reachability pre-check.

The robot itself executes Cartesian motions with its own controller; this module
only rejects grasp candidates that are clearly outside the joint limits before
any motion is sent.
"""
from __future__ import annotations

import math

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
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    T = np.empty((4, 4))
    T[0] = ct, -st, 0.0, a
    T[1] = st * ca, ct * ca, -sa, -d * sa
    T[2] = st * sa, ct * sa, ca, d * ca
    T[3] = 0.0, 0.0, 0.0, 1.0
    return T


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """3-vector cross product (np.cross costs ~15 us per call: most of the IK time)"""
    return np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (scipy's Rotation for angles near 180 deg, where this formula is unstable)."""
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if c < -0.9:
        return Rotation.from_matrix(R).as_rotvec()
    sn = 0.5 * math.sqrt(float(w @ w))  # sin(angle)
    return 0.5 * w if sn < 1e-12 else w * (math.atan2(sn, c) / (2.0 * sn))


_PIN = None


def _pin():
    """pinocchio's model of the same arm (Franka's FR3 URDF, the same DH frames: checked to 1e-15), ~20x faster than
    the Python DH products below; None if pinocchio or the URDF is missing (then the Python version is used)"""
    global _PIN
    if _PIN is None:
        try:
            import pinocchio as pin

            from .config import PROJECT_DIR

            urdf = PROJECT_DIR / "data" / "third_party" / "fr3_build" / "fr3.urdf"
            model = pin.buildModelFromUrdf(str(urdf))
            data = model.createData()
            _PIN = (pin, model, data, model.getFrameId("fr3_link8"), [model.getJointId(f"fr3_joint{i}") for i in range(1, 8)])
        except Exception:  # noqa: BLE001
            _PIN = False
    return _PIN or None


def fk_flange(q: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """Flange pose for reported joint angles q (plus JOINT_OFFSETS, or the given offsets)."""
    q = np.asarray(q, dtype=float) + (JOINT_OFFSETS if offsets is None else np.asarray(offsets, dtype=float))
    P = _pin()
    if P is not None:
        pin, model, data, fid, _ = P
        pin.framesForwardKinematics(model, data, q)
        return data.oMf[fid].homogeneous.copy()
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, q):
        T = T @ _dh(a, d, alpha, theta)
    return T @ _dh(0.0, FLANGE_D, 0.0, 0.0)


def fk(q: np.ndarray, F_T_EE: np.ndarray = DEFAULT_F_T_EE) -> np.ndarray:
    return fk_flange(np.asarray(q, dtype=float)) @ F_T_EE


def _frames(q: np.ndarray, F_T_EE: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Joint frames (after each joint rotation) and the EE pose."""
    P = _pin()
    if P is not None:
        pin, model, data, fid, jids = P
        pin.framesForwardKinematics(model, data, np.asarray(q, dtype=float) + JOINT_OFFSETS)
        return [data.oMi[j].homogeneous for j in jids], data.oMf[fid].homogeneous @ F_T_EE
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
        J[:3, i] = _cross(z, pe - Fi[:3, 3])
        J[3:, i] = rot_weight * (Rt @ z)
    return J


def _residual(q: np.ndarray, target: np.ndarray, F_T_EE: np.ndarray, rot_weight: float) -> np.ndarray:
    T = fk(q, F_T_EE)
    dp = T[:3, 3] - target[:3, 3]
    dr = _rotvec(target[:3, :3].T @ T[:3, :3])
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


def _gn_pull(q, target, F_T_EE, rot_weight, q_prev, pull, lo, hi, iters: int = 60):
    """argmin |residual(x)|^2 + pull^2 |x - q_prev|^2 for lo <= x <= hi, from q: Gauss-Newton with the joints that sit
    on a limit and are pushed beyond it held there for the step (active set), steps <= 0.3 rad"""
    x = np.clip(np.asarray(q, float), lo, hi)
    P2 = pull * pull
    eps = 1e-9
    for _ in range(iters):
        r = _residual(x, target, F_T_EE, rot_weight)
        J = _jacobian(x, target, F_T_EE, rot_weight)
        g = J.T @ r + P2 * (x - q_prev)
        H = J.T @ J + (P2 + 1e-9) * np.eye(7)
        free = ~(((x <= lo + eps) & (g > 0)) | ((x >= hi - eps) & (g < 0)))  # descent would leave the limit
        dx = np.zeros(7)
        if free.any():
            dx[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        s = float(np.abs(dx).max())
        if s > 0.3:
            dx *= 0.3 / s
        xn = np.clip(x + dx, lo, hi)
        if float(np.abs(xn - x).max()) < 1e-8:
            return xn
        x = xn
    return x


def solve_ik_near(
    target_T_BE: np.ndarray,
    q_start: np.ndarray,
    F_T_EE: np.ndarray = DEFAULT_F_T_EE,
    limit_margin_rad: float = 0.05,
    steps: int | None = None,
    pull: float = 0.03,
    pos_tol_m: float = 1e-3,
    rot_tol_deg: float = 0.5,
) -> np.ndarray | None:
    """IK that stays on the current branch: follow the straight Cartesian path from fk(q_start) to the target in
    small steps, each solved from the previous joints with a weak pull toward them (least joint change).
    Use this for the joint moves the robot actually executes; ``solve_ik`` may return a far-away solution."""
    lo, hi = Q_MIN + limit_margin_rad, Q_MAX - limit_margin_rad
    q = np.clip(np.asarray(q_start, dtype=float), lo, hi)
    T0 = fk(q, F_T_EE)
    dR = Rotation.from_matrix(target_T_BE[:3, :3] @ T0[:3, :3].T).as_rotvec()
    rot_weight = 0.2
    if steps is None:  # a step per 3 cm or 6 deg (10 steps for every move until 2026-11-04, most of the planning time)
        steps = int(np.clip(np.ceil(max(np.linalg.norm(target_T_BE[:3, 3] - T0[:3, 3]) / 0.03,
                                        np.linalg.norm(dR) / np.radians(6.0))), 1, 10))
    for k in range(1, steps + 1):
        Tk = T0.copy()
        Tk[:3, :3] = Rotation.from_rotvec(dR * k / steps).as_matrix() @ T0[:3, :3]
        Tk[:3, 3] = T0[:3, 3] + (target_T_BE[:3, 3] - T0[:3, 3]) * k / steps
        q_prev = q.copy()
        # residual + weak pull towards the previous joints, within the limits: Gauss-Newton on the stacked residual
        # (the pull keeps it well conditioned), steps clipped to the limits (scipy's least_squares did the same with
        # ~1 ms of overhead per call: a quarter of the planning time, 2026-11-05)
        q = _gn_pull(q, Tk, F_T_EE, rot_weight, q_prev, pull, lo, hi)
    r = _residual(q, target_T_BE, F_T_EE, rot_weight)
    if np.linalg.norm(r[:3]) <= pos_tol_m and np.degrees(np.linalg.norm(r[3:]) / rot_weight) <= rot_tol_deg:
        return q
    return None


def ik_fixed_q7(target_T_BE: np.ndarray, q_seed: np.ndarray, q7: float, F_T_EE: np.ndarray = DEFAULT_F_T_EE,
                limit_margin_rad: float = 0.05, iters: int = 80, pos_tol_m: float = 1e-3, rot_tol_deg: float = 0.5):
    """IK with joint 7 held at q7 (the arm's one redundant degree of freedom): joints 1-6 by damped least squares
    from q_seed. Joint angles or None."""
    lo, hi = Q_MIN + limit_margin_rad, Q_MAX - limit_margin_rad
    if not lo[6] <= q7 <= hi[6]:
        return None
    q = np.clip(np.asarray(q_seed, dtype=float), lo, hi)
    q[6] = q7
    rw = 0.2
    for _ in range(iters):
        r = _residual(q, target_T_BE, F_T_EE, rw)
        if np.linalg.norm(r[:3]) < 2e-5 and np.linalg.norm(r[3:]) < 2e-5:
            break
        J = _jacobian(q, target_T_BE, F_T_EE, rw)[:, :6]
        dq = -J.T @ np.linalg.solve(J @ J.T + 0.02 ** 2 * np.eye(6), r)
        s = np.max(np.abs(dq))
        if s > 0.3:
            dq *= 0.3 / s
        q[:6] = np.clip(q[:6] + dq, lo[:6], hi[:6])
    r = _residual(q, target_T_BE, F_T_EE, rw)
    if np.linalg.norm(r[:3]) <= pos_tol_m and np.degrees(np.linalg.norm(r[3:]) / rw) <= rot_tol_deg:
        return q
    return None


def ik_all(target_T_BE: np.ndarray, F_T_EE: np.ndarray = DEFAULT_F_T_EE, seeds=(), limit_margin_rad: float = 0.05,
           q7_step_deg: float = 10.0, dedupe_rad: float = 0.05, n_random: int = 48) -> list[np.ndarray]:
    """Joint solutions covering the whole self-motion of the arm at this pose: joint 7 swept over its range in
    q7_step_deg steps, each branch followed by continuation (the previous solution seeds the next q7) from every seed
    (the given ones plus structured ones: base turned to the target, elbow and wrist variants). A seed that runs onto
    a branch already found stops there."""
    lo, hi = Q_MIN + limit_margin_rad, Q_MAX - limit_margin_rad
    az = float(np.arctan2(target_T_BE[1, 3], target_T_BE[0, 3]))
    base = [np.asarray(s, dtype=float) for s in seeds]
    # every branch must get a seed: random ones over the whole joint box (a structured set missed the branch with
    # joint 5 near -150 deg, 2026-11-04), deterministic
    rng = np.random.default_rng(12345)
    for _ in range(n_random):
        q = rng.uniform(lo, hi)
        q[0] = az + rng.uniform(-1.7, 1.7)
        base.append(q)
    grid = np.arange(lo[6], hi[6] + 1e-9, np.radians(q7_step_deg))
    found: dict[int, list[np.ndarray]] = {}
    out = []
    rw = 0.2
    for s in base:
        # onto a branch first (all 7 joints free), then along it
        q0 = _dls(np.clip(s, lo, hi), target_T_BE, F_T_EE, lo, hi, rw, iters=60)
        r = _residual(q0, target_T_BE, F_T_EE, rw)
        if np.linalg.norm(r[:3]) > 1e-3 or np.degrees(np.linalg.norm(r[3:]) / rw) > 0.5:
            continue
        i0 = int(np.argmin(np.abs(grid - q0[6])))
        for direction in (1, -1):
            prev = q0
            for i in (range(i0, len(grid)) if direction > 0 else range(i0 - 1, -1, -1)):
                q = ik_fixed_q7(target_T_BE, prev, float(grid[i]), F_T_EE, limit_margin_rad, iters=30)
                if q is None:
                    break  # the branch ends (a joint limit, or it folds away)
                if any(np.abs(q - o).max() < dedupe_rad for o in found.get(i, [])):
                    break  # on a branch already followed
                found.setdefault(i, []).append(q)
                out.append(q)
                prev = q
    return out
