"""Rigid-transform helpers. All transforms are 4x4 homogeneous numpy arrays."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray | None = None, t: Sequence[float] | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an (..., 3) array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rotate(T: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) direction vectors (no translation)."""
    return np.asarray(vecs, dtype=float) @ T[:3, :3].T


def T_from_pos_quat(pos: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    return make_T(Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix(), pos)


def pos_quat_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def T_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(T)]


def T_from_list(rows: Iterable[Iterable[float]]) -> np.ndarray:
    T = np.asarray([[float(v) for v in row] for row in rows], dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {T.shape}")
    check_rigid(T)
    return T


def check_rigid(T: np.ndarray, tol: float = 1e-4) -> None:
    if not np.all(np.isfinite(T)):
        raise ValueError("transform contains non-finite values")
    R = T[:3, :3]
    if np.abs(R @ R.T - np.eye(3)).max() > tol or abs(np.linalg.det(R) - 1.0) > tol:
        raise ValueError("transform rotation is not orthonormal")
    if np.abs(T[3] - np.array([0, 0, 0, 1])).max() > 1e-9:
        raise ValueError("transform last row must be [0, 0, 0, 1]")


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform T such that dst ~= T * src (no scale)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need >= 3 matching points")
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, cd - R @ cs)
