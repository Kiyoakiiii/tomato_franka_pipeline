"""scripts/tool_pivot.py (2026-11-16): the side tool's contact point from poses that all put it on one fixed point."""
import importlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from seedling.transforms import make_T


def _poses(p_F, c_B, yaws, tilts, rng, noise=0.0):
    out = []
    for yaw, tilt in zip(yaws, tilts):
        R = Rotation.from_euler("zx", [yaw, 180 + tilt], degrees=True).as_matrix()  # flange down, turned, tilted
        t = c_B - R @ p_F + rng.normal(0, noise, 3)
        out.append(make_T(R, t))
    return out


def test_pivot_finds_the_tool_point():
    from seedling.config import PROJECT_DIR as project_dir
    sys.path.insert(0, str(project_dir / "scripts"))
    tp = importlib.import_module("tool_pivot")
    rng = np.random.default_rng(0)
    p_true = np.array([0.259 * np.cos(np.radians(-45)), 0.259 * np.sin(np.radians(-45)), 0.048])
    c_true = np.array([0.55, 0.05, 0.10])
    # flange kept vertical: the direction and reach found, the height along the axis taken from the config
    Ts = _poses(p_true, c_true, [0, 60, 120, 200, 280], [0] * 5, rng, noise=0.0005)
    p, c, res, z_seen = tp.solve_pivot(Ts, below_m=0.050)
    assert not z_seen and abs(p[2] - 0.050) < 1e-9
    assert np.degrees(abs(np.arctan2(p[1], p[0]) - np.radians(-45))) < 0.3 and abs(np.hypot(p[0], p[1]) - 0.259) < 0.0015
    # tilted now and then: the height is seen too
    Ts = _poses(p_true, c_true, [0, 60, 120, 200, 280], [0, 15, -10, 0, 20], rng, noise=0.0005)
    p, c, res, z_seen = tp.solve_pivot(Ts, below_m=0.050)
    assert z_seen and np.linalg.norm(p - p_true) < 0.003 and np.linalg.norm(c - c_true) < 0.003


def test_weight_from_two_headings_cancels_the_offset():
    from seedling.config import PROJECT_DIR

    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    tp = importlib.import_module("tool_pivot")
    g, m0, c0 = 9.81, 0.73, np.array([-0.01, 0.0, 0.03])
    m, th = 0.92, 135.0
    u_F = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    c_true = 0.14 * u_F
    off = np.array([0.22, -0.05])  # the estimate's offset, (mass x lever) units, base frame
    reads = []
    for heading in (0.0, 138.0):  # the arm's direction in the base frame
        ub = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])
        # base frame = flange frame turned so that the arm (u_F) points along ub
        rot = np.arctan2(ub[1], ub[0]) - np.radians(th)
        Rz = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        mc = Rz @ (m * c_true - m0 * c0[:2]) + off
        reads.append({"arm_dir_B": ub.tolist(), "F": [0, 0, (m - m0) * g], "tau": [mc[1] * g, -mc[0] * g, 0]})
    w = tp.weight_from_readings(reads, m0, c0, th)
    assert abs(w["m_total"] - m) < 1e-6 and np.allclose(w["c_total"][:2], c_true, atol=1e-6)


def test_joint7_range_of_the_side_tool_narrows_every_check(cfg):
    """2026-11-16: with joint 7 above -14 deg the camera on the side tool runs into the wrist; the config's range is
    applied to the joint limits every IK and joint move reads (the old overview pose, +41 deg, is refused; the
    configured overview and home poses are inside)"""
    from seedling import kinematics
    from seedling.robot import SimRobot, UnsafeTarget
    import pytest

    lo, hi = cfg["robot"]["side_tool"]["joint7_range_deg"]
    assert np.isclose(np.degrees(kinematics.Q_MAX[6]), hi) and np.isclose(np.degrees(kinematics.Q_MIN[6]), max(lo, -172.8), atol=0.05)
    r = SimRobot(cfg, step_mode=False)
    with pytest.raises(UnsafeTarget):
        r.move_joints(np.array([0.086, -1.095, -0.163, -2.261, -0.167, 1.714, 0.710]))  # the old overview pose
    assert r.move_joints(np.array(cfg["robot"]["home_joints_rad"], float)).ok
    assert r.move_joints(np.array(cfg["tray"]["overview_q"], float)).ok


def test_joint6_lower_limit_from_the_config(cfg):
    """2026-11-17: the arm stood at joint 6 = 0.424 without an error (datasheet 0.5445); robot.joint6_min_rad sets the
    lower limit every check reads: the hand-placed pose at 0.516 is a valid target, one below 0.424 is refused"""
    from seedling import kinematics
    from seedling.robot import SimRobot, UnsafeTarget
    import pytest

    assert np.isclose(kinematics.Q_MIN[5], cfg["robot"]["joint6_min_rad"]) and cfg["robot"]["joint6_min_rad"] < 0.5445
    r = SimRobot(cfg, step_mode=False)
    assert r.move_joints(np.array([0.042, -0.679, -0.08, -1.905, -0.222, 0.516, -2.521])).ok
    with pytest.raises(UnsafeTarget):
        r.move_joints(np.array([0.042, -0.679, -0.08, -1.905, -0.222, 0.40, -2.521]))
