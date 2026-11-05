"""Collision-free joint paths for the long moves (turning to a flat hand, carrying to the disc, back to the overview).

A move is executed as a straight line in joint space (move_joints). Between two safe poses that straight line can
sweep the hand into the tray: the poses differ by ~100 deg in joint 6 (hand pointing down -> flat). Fixed via poses
only cover the layouts they were tuned on (2026-11-04: C7, the tray moved, both fixed vias failed), so here:

1. the straight line;
2. one via pose: some joints go to their target angles first, the others after (every subset of 1-5 joints);
3. RRT-Connect in joint space (random samples inside a box around both ends, widened if needed), then shortcut.

``state_ok(q) -> bool`` is the caller's check of one pose (clearances, self clearance, ...). Segments are checked
every ``step_deg`` of the largest joint change, the start pose excluded (the caller has its own rule there).
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from . import kinematics

_SUBSETS = [(4, 5, 6), (0, 1, 2, 3)] + [c for n in (1, 2, 3, 4, 5) for c in itertools.combinations(range(7), n)
                                        if c not in ((4, 5, 6), (0, 1, 2, 3))]


def segment_ok(qa, qb, state_ok, step_deg: float = 2.0) -> bool:
    n = max(1, int(np.ceil(np.degrees(np.abs(qb - qa).max()) / step_deg)))
    # the middle first: a bad segment is found sooner
    ts = sorted(np.linspace(0.0, 1.0, n + 1)[1:], key=lambda t: abs(t - 0.5))
    return all(state_ok(qa + t * (qb - qa)) for t in ts)


def via_path(qa, qb, state_ok, step_deg: float = 2.0):
    """[qb] if the straight line is safe, else [qv, qb] through the first safe one-via pose, else None"""
    if segment_ok(qa, qb, state_ok, step_deg):
        return [qb]
    for sub in _SUBSETS:
        qv = qa.copy()
        qv[list(sub)] = qb[list(sub)]
        if state_ok(qv) and segment_ok(qa, qv, state_ok, step_deg) and segment_ok(qv, qb, state_ok, step_deg):
            return [qv, qb]
    return None


def shortcut(path, state_ok, step_deg: float = 2.0):
    """greedy: from each waypoint straight to the farthest later one that can be reached safely"""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_ok(path[i], path[j], state_ok, step_deg):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _chain(nodes, parents, k):
    out = []
    while k != -1:
        out.append(nodes[k])
        k = parents[k]
    return out


def rrt_connect(qa, qb, state_ok, *, time_s: float = 4.0, step_deg: float = 2.0, eps: float = 0.3,
                margin_rad=(0.5, 1.0, None), seed: int = 0):
    """RRT-Connect between qa and qb; samples within the joint box spanned by both ends plus margin (each margin in
    turn; None: the whole joint range), so the arm stays near both poses. Waypoints after qa, ending with qb, or None."""
    rng = np.random.default_rng(seed)
    lo_lim, hi_lim = kinematics.Q_MIN + 0.05, kinematics.Q_MAX - 0.05
    t_end = time.time() + time_s
    per = time_s / len(margin_rad)
    for m in margin_rad:
        if m is None:
            lo, hi = lo_lim, hi_lim
        else:
            lo = np.maximum(np.minimum(qa, qb) - m, lo_lim)
            hi = np.minimum(np.maximum(qa, qb) + m, hi_lim)
        ta, tb = [qa.copy()], [qb.copy()]  # nodes
        pa, pb = [-1], [-1]                 # parents
        t_stage = min(t_end, time.time() + per)
        swapped = False
        while time.time() < t_stage:
            s = rng.uniform(lo, hi)
            # extend tree A one step towards the sample
            i = int(np.argmin(np.abs(np.array(ta) - s).max(axis=1)))
            d = s - ta[i]
            q = ta[i] + d * min(1.0, eps / max(np.abs(d).max(), 1e-9))
            if state_ok(q) and segment_ok(ta[i], q, state_ok, step_deg):
                ta.append(q)
                pa.append(i)
                # connect tree B towards it, step by step
                j = int(np.argmin(np.abs(np.array(tb) - q).max(axis=1)))
                reached = False
                while True:
                    d = q - tb[j]
                    if np.abs(d).max() < 1e-9:
                        reached = True
                        break
                    q2 = tb[j] + d * min(1.0, eps / max(np.abs(d).max(), 1e-9))
                    if not (state_ok(q2) and segment_ok(tb[j], q2, state_ok, step_deg)):
                        break
                    tb.append(q2)
                    pb.append(j)
                    j = len(tb) - 1
                if reached:
                    a_side = _chain(ta, pa, len(ta) - 1)[::-1]  # root ... q
                    b_side = _chain(tb, pb, j)                  # q ... root
                    path = a_side + b_side[1:]
                    if swapped:
                        path = path[::-1]
                    return shortcut(path, state_ok, step_deg)[1:]
            ta, tb, pa, pb = tb, ta, pb, pa
            swapped = not swapped
    return None


def find_path(qa, qb, state_ok, *, time_s: float = 4.0, step_deg: float = 2.0, seed: int = 0):
    """waypoints after qa ending with qb (straight: [qb]; one via: [qv, qb]; else an RRT path), or None"""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    if not state_ok(qb):
        return None
    p = via_path(qa, qb, state_ok, step_deg)
    if p is not None:
        return p
    if time_s <= 0:
        return None
    return rrt_connect(qa, qb, state_ok, time_s=time_s, step_deg=step_deg, seed=seed)


def lift_then_path(q0, q_goal, scene, cfg, T_FC, *, up_to: float = 0.32, time_s: float = 10.0):
    """From q0 (e.g. just after letting go, the hand still among the plants) to q_goal: straight up first, then a path
    (find_path) on which every pose is clear in the scene. The lift height is the highest one (lowest part of the hand
    at most --up-to, in 2 cm steps down to no lift at all) from which a path is found: a lift that is too high can fold
    the hand onto the arm (2026-11-04 simulation: 0 mm self clearance after the lift). Moving up, nothing may come
    closer than it was at q0.

    Returns (steps, message): steps = [(name, q_from, q_to), ...] or None."""
    from .config import ee_to_contact, flange_to_ee
    from .transforms import inv_T
    from .wrist import lowest_body_z

    F, G = flange_to_ee(cfg), ee_to_contact(cfg)
    margin = float(cfg["robot"]["ik_joint_margin_rad"])
    T = kinematics.fk(q0, F) @ G
    z0 = float(T[2, 3])
    top = up_to + max(0.0, z0 - lowest_body_z(q0, T_FC, cfg)[0])  # the grasp centre height that lifts the lowest part to up_to
    m0 = min(scene.margins(q0).values())
    tried = []
    for zt in list(np.arange(top, z0 + 0.005, -0.02)) + [None]:
        if zt is None:
            q1 = q0
        else:
            T1 = T.copy()
            T1[2, 3] = zt
            q1 = kinematics.solve_ik_near(T1 @ inv_T(G), q0, F, margin)
            if q1 is None:
                continue
            if scene.worst(q0, q1)[0] < min(0.0, m0) - 0.001:
                tried.append(f"lift to z={zt:.2f}: comes closer to something")
                continue
        if not scene.free_ok(q1):
            bad = {k: round(v * 1000) for k, v in scene.margins(q1).items() if v < 0}
            tried.append(f"{'no lift' if zt is None else f'lift to z={zt:.2f}'}: too close there (margins mm {bad})")
            continue
        path = find_path(q1, q_goal, scene.free_ok, time_s=time_s / 3)
        if path is None:
            tried.append(f"{'no lift' if zt is None else f'lift to z={zt:.2f}'}: no path on from there")
            continue
        steps = [] if zt is None else [("lift in place", q0, q1)]
        qa = q1
        for i, qb in enumerate(path):
            steps.append(("to the goal" if len(path) == 1 else (f"to via pose {i + 1}/{len(path) - 1}" if i < len(path) - 1 else "via pose to the goal"), qa, qb))
            qa = qb
        return steps, ("lifted to z=%.2f" % zt) if zt is not None else "no lift needed"
    return None, "; ".join(tried[-4:]) or "cannot lift straight up from here"
