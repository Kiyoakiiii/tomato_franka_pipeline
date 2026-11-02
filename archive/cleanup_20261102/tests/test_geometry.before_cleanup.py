import numpy as np
from scipy.spatial.transform import Rotation

from seedling import kinematics as K
from seedling.grasp import approach_direction, gripper_frame
from seedling.stem3d import fit_line_ransac
from seedling.transforms import apply_T, inv_T, kabsch, make_T


def test_kabsch_recovers_transform():
    rng = np.random.default_rng(0)
    T = make_T(Rotation.random(random_state=1).as_matrix(), [0.1, -0.2, 0.3])
    src = rng.normal(size=(20, 3))
    assert np.allclose(kabsch(src, apply_T(T, src)), T, atol=1e-9)
    assert np.allclose(inv_T(T) @ T, np.eye(4))


def test_fk_ready_pose_matches_known_values():
    T = K.fk(K.Q_READY)
    assert np.allclose(T[:3, 3], [0.307, 0.0, 0.487], atol=2e-3)
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-6)  # hand points down at the ready pose


def test_ik_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(K.Q_MIN + 0.3, K.Q_MAX - 0.3)
        T = K.fk(q)
        sol = K.solve_ik(T, q + rng.normal(0, 0.2, 7))
        assert sol is not None
        assert np.linalg.norm(K.fk(sol)[:3, 3] - T[:3, 3]) < 1e-3


def test_ik_rejects_unreachable():
    assert K.solve_ik(make_T(None, [2.0, 0, 0.3]), None, n_random_seeds=1) is None


def test_gripper_frame_axes_perpendicular_to_stem():
    p = np.array([0.5, 0.05, 0.1])
    s = np.array([0.1, 0.0, 1.0]) / np.linalg.norm([0.1, 0, 1])
    a = approach_direction(p, 0, 30)
    T = gripper_frame(p, s, a)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert abs(R[:, 1] @ s) < 1e-9  # closing axis perpendicular to the stem
    assert np.allclose(R[:, 2], a)
    assert a[2] < 0  # tilted downward


def test_ransac_line_with_outliers():
    rng = np.random.default_rng(0)
    d = np.array([0.1, -0.9, 0.2]); d /= np.linalg.norm(d)
    t = rng.uniform(-0.03, 0.03, 200)
    pts = np.array([0.0, 0.0, 0.5]) + np.outer(t, d) + rng.normal(0, 0.0005, (200, 3))
    pts = np.vstack([pts, rng.uniform(-0.05, 0.05, (60, 3)) + [0, 0, 0.55]])
    c, dd, inl = fit_line_ransac(pts, 0.002, 300)
    assert abs(abs(dd @ d) - 1) < 1e-3
    assert inl[:200].mean() > 0.95 and inl[200:].mean() < 0.2
