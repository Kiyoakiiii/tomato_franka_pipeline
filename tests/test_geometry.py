import numpy as np
from scipy.spatial.transform import Rotation

from seedling import kinematics as K
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
