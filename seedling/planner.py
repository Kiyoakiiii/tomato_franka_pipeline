"""Joint-space paths with OMPL (RRT-Connect, then OMPL's path simplification: shortcuts and smoothing), the validity
of each pose given by the caller (seedling.world.World.ok and any extra condition). The straight joint move is
tried first. Replaces the hand-made search of seedling.motion (2026-11-04).
"""
from __future__ import annotations

import numpy as np

from . import kinematics

_LOG_SET = False


def _quiet():
    global _LOG_SET
    if not _LOG_SET:
        try:
            from ompl import util as ou

            ou.setLogLevel(ou.LogLevel.LOG_WARN)
        except Exception:  # noqa: BLE001  (older bindings: keep OMPL's own log level)
            pass
        _LOG_SET = True


def segment_ok(qa, qb, valid, step_rad: float = np.radians(1.5)) -> bool:
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    n = max(1, int(np.ceil(np.abs(qb - qa).max() / step_rad)))
    ts = sorted(np.linspace(0.0, 1.0, n + 1)[1:], key=lambda t: abs(t - 0.5))  # the middle first
    return all(valid(qa + t * (qb - qa)) for t in ts)


def plan_path(q_start, q_goal, valid, *, time_s: float = 3.0, step_rad: float = np.radians(1.5),
              margin_rad: float = 0.05, seed: int = 1):
    """waypoints after q_start ending with q_goal ([q_goal] when the straight joint move is valid), or None.
    The start pose itself is not checked (the caller decides whether it may start where it is)."""
    qa, qb = np.asarray(q_start, float), np.asarray(q_goal, float)
    if not valid(qb):
        return None
    if segment_ok(qa, qb, valid, step_rad):
        return [qb]
    _quiet()
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    try:
        ou.RNG.setSeed(seed)
    except Exception:  # noqa: BLE001
        pass
    lo, hi = kinematics.Q_MIN + margin_rad, kinematics.Q_MAX - margin_rad
    lo, hi = np.minimum(lo, np.minimum(qa, qb)), np.maximum(hi, np.maximum(qa, qb))
    space = ob.RealVectorStateSpace(7)
    b = ob.RealVectorBounds(7)
    for i in range(7):
        b.setLow(i, float(lo[i]))
        b.setHigh(i, float(hi[i]))
    space.setBounds(b)
    ss = og.SimpleSetup(space)
    first = [True]

    def check(s):
        q = np.array([s[i] for i in range(7)])
        if first[0] and np.abs(q - qa).max() < 1e-9:
            return True  # the start (OMPL checks it; the caller already decided)
        return bool(valid(q))

    ss.setStateValidityChecker(check)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(float(step_rad / max(space.getMaximumExtent(), 1e-9)))
    st, gl = space.allocState(), space.allocState()
    for i in range(7):
        st[i], gl[i] = float(qa[i]), float(qb[i])
    ss.setStartAndGoalStates(st, gl)
    ss.setPlanner(og.RRTConnect(si))
    if not ss.solve(float(time_s)) or not ss.haveExactSolutionPath():
        return None
    ss.simplifySolution(min(1.0, float(time_s)))
    path = ss.getSolutionPath()
    pts = [np.array([path.getState(k)[i] for i in range(7)]) for k in range(path.getStateCount())]
    # OMPL checks its own motions at the resolution set above; checked once more here at ours (start excluded)
    out, prev = [], qa
    for q in pts[1:]:
        if not segment_ok(prev, q, valid, step_rad):
            return None
        out.append(q)
        prev = q
    out[-1] = qb
    return out


def lift_then_plan(q0, q_goal, world, cfg, *, lifts_m=(0.0, 0.05, 0.10, 0.15, 0.20), time_s: float = 6.0, valid=None):
    """From q0 (e.g. just after letting go, the hand still near the plants) to q_goal: straight up first (the least
    of lifts_m that works; straight in Cartesian space, see seedling.cartesian), then a path (plan_path) on which every
    pose is valid (default world.ok). Going up, nothing may come closer than it was at q0.
    Returns (steps [(name, q_from, q_to)], message) or (None, why)."""
    from . import cartesian
    from .config import ee_to_contact, flange_to_ee

    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    valid = valid or world.ok
    T0 = kinematics.fk(np.asarray(q0, float), F) @ G
    zmax = float(cfg["robot"]["workspace_m"]["z"][1]) - 0.02
    m0 = min(v[0] for v in world.clearances(q0).values())
    tried = []
    for d in lifts_m:
        if d == 0.0:
            q1, pieces = np.asarray(q0, float), []
        else:
            T1 = T0.copy()
            T1[2, 3] = min(T0[2, 3] + d, zmax)
            pcs, why = cartesian.chain(q0, [("lift", T1)], F, G, margin)
            if pcs is None:
                tried.append(f"lift {d * 100:.0f} cm: {why}")
                continue
            pieces = pcs[0][2]
            q1, qa, okk = pieces[-1], np.asarray(q0, float), True
            for qb in pieces:
                if world.worst(qa, qb)[0] < min(0.0, m0) - 0.001:
                    okk = False
                    break
                qa = qb
            if not okk:
                tried.append(f"lift {d * 100:.0f} cm: comes closer to something")
                continue
        if not valid(q1):
            tried.append(f"{'no lift' if d == 0 else f'lift {d * 100:.0f} cm'}: still too close to something there")
            continue
        path = plan_path(q1, q_goal, valid, time_s=time_s / len(lifts_m))
        if path is None:
            tried.append(f"{'no lift' if d == 0 else f'lift {d * 100:.0f} cm'}: no path on from there")
            continue
        steps, qa = [], np.asarray(q0, float)
        for qb in pieces:
            steps.append(("lift in place", qa, qb))
            qa = qb
        for i, qb in enumerate(path):
            steps.append(("to the goal" if len(path) == 1 else (f"via {i + 1}/{len(path) - 1}" if i < len(path) - 1 else "to the goal"), qa, qb))
            qa = qb
        return steps, ("no lift needed" if d == 0 else f"lifted {d * 100:.0f} cm")
    return None, "; ".join(tried[-4:]) or "no way found"
