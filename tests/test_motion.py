import numpy as np

from seedling import motion


def _blocked_middle(q):
    # a wall in joint space: joint 1 between -0.2 and 0.2 is forbidden unless joint 2 is above 0.8
    return not (abs(q[0]) < 0.2 and q[1] < 0.8)


def test_straight_when_free():
    qa, qb = np.zeros(7) - 0.5, np.zeros(7) + 0.5
    qa[3] = qb[3] = -1.5
    assert len(motion.find_path(qa, qb, lambda q: True)) == 1


def test_via_or_tree_around_a_wall():
    qa = np.array([-0.6, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    qb = np.array([0.6, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    p = motion.find_path(qa, qb, _blocked_middle, time_s=5.0)
    assert p is not None and np.allclose(p[-1], qb)
    prev = qa
    for q in p:
        assert motion.segment_ok(prev, q, _blocked_middle)
        prev = q


def test_tree_search_alone():
    qa = np.array([-0.6, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    qb = np.array([0.6, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    p = motion.rrt_connect(qa, qb, _blocked_middle, time_s=5.0)
    assert p is not None and np.allclose(p[-1], qb)
    prev = qa
    for q in p:
        assert motion.segment_ok(prev, q, _blocked_middle)
        prev = q


def test_none_when_target_bad():
    qa, qb = np.zeros(7), np.ones(7) * 0.1
    assert motion.find_path(qa, qb, lambda q: q[0] < 0.05) is None
