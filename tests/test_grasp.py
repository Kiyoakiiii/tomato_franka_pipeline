"""seedling/grasp.py: the side grasp's shared rules, against the copies pick.py and plan.py had of them until 2026-11-11
(frozen below as they were), on many random cases: the merge changed no number."""
from __future__ import annotations

import numpy as np
import pytest

from seedling import grasp as G

RNG = np.random.default_rng(7)


# ------------------------------------------------- frozen: scripts/plan.py soil_info, 2026-11-11 -------------
def soil_info_76(grid, cell, obs, tray_top, disagree=0.008, plug_dz=0.013, leafy_cover=0.35):
    HEIGHT_EST, HEIGHT_SURE, HEIGHT_LEAFY = 0.035, 0.017, 0.0145
    z_ov = (grid.get("soil_z_overview") or {}).get(cell)
    z_near = obs["measure"]["soil_z"] if obs and obs["measure"].get("ok") else None
    off_c = obs and np.hypot(*obs["measure"].get("shift_mm", [0, 0])) > 12.0
    fb = tray_top + plug_dz
    if z_near is not None and z_ov is not None and off_c and z_near > z_ov + 0.003:
        soil, src, sure = z_ov, "overview (the close patch was off the cell centre)", True
    elif z_near is not None and z_ov is not None and abs(z_near - z_ov) > disagree:
        soil, src = max(z_near, z_ov), "overview and close disagree: the higher"
        sure = soil == z_near
    elif z_near is not None:
        soil, src, sure = z_near, "close measurement", True
    elif z_ov is not None:
        soil, src, sure = (z_ov, "overview", False) if z_ov >= fb else (fb, "overview below the usual plug height: tray top + 13 mm", False)
    else:
        soil, src, sure = fb, "NOT MEASURED: tray top + 13 mm", False
    h = HEIGHT_SURE if sure else HEIGHT_EST
    cov = (obs or {}).get("leaf_cover")
    if sure and cov is not None and cov > leafy_cover:
        h = min(h, HEIGHT_LEAFY)
    low = min(v for v in (z_near, z_ov, soil) if v is not None)
    return {"soil": float(soil), "soil_low": float(low), "src": src, "sure": sure, "height": h}


# ------------------------------------------------- frozen: scripts/pick.py plan (inline), 2026-11-11 -------
def soil_65(grid, cell, obs, tray_top, disagree_mm=8.0, plug_dz_mm=13.0, height_close_mm=17.0, height_mm=35.0,
            leafy_cover=0.35, height_leafy_mm=14.5):
    z_ov = (grid.get("soil_z_overview") or {}).get(cell)
    z_near = obs["measure"]["soil_z"] if obs and obs["measure"].get("ok") else None
    off_c = obs and np.hypot(*obs["measure"].get("shift_mm", [0, 0])) > 12.0
    if z_near is not None and z_ov is not None and off_c and z_near > z_ov + 0.003:
        soil_z, src = z_ov, f"overview {z_ov:.3f}: the close {z_near:.3f} came from soil {np.hypot(*obs['measure']['shift_mm']):.0f} mm off the cell centre"
    elif z_near is not None and z_ov is not None and abs(z_near - z_ov) > disagree_mm / 1000:
        soil_z, src = max(z_near, z_ov), f"overview {z_ov:.3f} and close {z_near:.3f} disagree: the higher"
    elif z_near is not None:
        soil_z, src = z_near, "close measurement" + (f" (overview {z_ov:.3f})" if z_ov is not None else "")
    elif z_ov is not None:
        fb = tray_top + plug_dz_mm / 1000
        soil_z, src = (z_ov, "overview measurement (no close one)") if z_ov >= fb else \
            (fb, f"overview {z_ov:.3f} (no close one) below the usual plug height: tray top + plug_dz")
    else:
        soil_z, src = tray_top + plug_dz_mm / 1000.0, "NOT MEASURED: tray top + plug_dz"
    sure = z_near is not None and (soil_z == z_near or src.startswith(f"overview {z_ov:.3f}: the close"))
    height = height_close_mm if sure else height_mm
    cov = (obs or {}).get("leaf_cover")
    if sure and cov is not None and cov > leafy_cover:
        height = min(height, height_leafy_mm)
    return soil_z, sure, height / 1000.0


# ------------------------------------------------- frozen: the finger floor (pick.py inline = plan.py finger_floor_fn) --------
def floor_old(P, c, yaw, clearance_mm=3.0):
    ah = np.array([np.cos(yaw), np.sin(yaw)])
    d = P[:, :2] - c[:2]
    s_ = d @ ah
    lat = np.abs(d @ np.array([-ah[1], ah[0]]))
    m = (lat > 0.012) & (lat < 0.034) & (s_ > -0.045) & (s_ < 0.015)
    if m.sum() < 30:
        return None
    return float(np.percentile(P[m, 2] + clearance_mm / 1000 + 0.0145, 98))


# ------------------------------------------------- frozen: hand clearance (pick.py inline = Program.hand_clearance) -------
def clearance_old(c, occupied, yaw, standoff_mm=60.0):
    back = -np.array([np.cos(yaw), np.sin(yaw)])
    lat_ax = np.array([-back[1], back[0]])
    so = standoff_mm / 1000.0
    boxes = [(-0.015, 0.075, 0.031), (0.075, 0.18 + so, 0.045)]
    best = np.inf
    for o in occupied:
        d = o - c[:2]
        s_, l_ = float(d @ back), abs(float(d @ lat_ax))
        for s0, s1, hw in boxes:
            best = min(best, float(np.hypot(max(0.0, s0 - s_, s_ - s1), max(0.0, l_ - hw))))
    return best


def _case(i):
    tray_top = 0.15 + RNG.uniform(-0.05, 0.05)
    grid = {"soil_z_overview": {}}
    if RNG.random() < 0.8:
        grid["soil_z_overview"]["C5"] = tray_top + RNG.uniform(-0.03, 0.03)
    obs = None
    if RNG.random() < 0.8:
        ok = RNG.random() < 0.85
        obs = {"measure": {"ok": ok, "soil_z": tray_top + RNG.uniform(-0.03, 0.035), "shift_mm": list(RNG.uniform(-16, 16, 2))}}
        if RNG.random() < 0.6:
            obs["leaf_cover"] = float(RNG.uniform(0, 0.6))
    return grid, obs, tray_top


@pytest.mark.parametrize("i", range(400))
def test_soil_info_as_plan_and_pick(i):
    grid, obs, tray_top = _case(i)
    new = G.soil_info(grid, "C5", obs, tray_top)
    old = soil_info_76(grid, "C5", obs, tray_top)
    for k in ("soil", "soil_low", "sure", "height"):
        assert new[k] == pytest.approx(old[k]), k
    s65, sure65, h65 = soil_65(grid, "C5", obs, tray_top)
    assert (new["soil"], new["sure"]) == (pytest.approx(s65), sure65)
    assert new["height"] == pytest.approx(h65)


@pytest.mark.parametrize("i", range(60))
def test_finger_floor_as_before(i):
    c = np.array([0.7, -0.1, 0.18])
    P = np.column_stack([c[0] + RNG.uniform(-0.06, 0.06, 3000), c[1] + RNG.uniform(-0.06, 0.06, 3000), 0.19 + RNG.normal(0, 0.004, 3000)])
    if i % 10 == 0:
        P = P[:20]  # too few points: None
    yaw = RNG.uniform(-np.pi, np.pi)
    assert G.finger_floor(P, c, yaw) == floor_old(P, c, yaw)
    assert G.finger_floor(None, c, yaw) is None


@pytest.mark.parametrize("i", range(60))
def test_hand_clearance_as_before(i):
    c = np.array([0.7, -0.1])
    stems = [c + RNG.uniform(-0.2, 0.2, 2) for _ in range(RNG.integers(0, 8))]
    yaw = RNG.uniform(-np.pi, np.pi)
    assert G.hand_clearance(c, stems, yaw) == pytest.approx(clearance_old(c, stems, yaw))


def test_grasp_cell_follows_closeup_setting():
    grid = {"cells": {"C5": [0.70, 0.10, 0.18]}}
    obs = {"overview_cell": [0.70, 0.10, 0.18], "closeup_xy": {"ok": True, "status": "ok", "shift_mm": [3.0, -4.0], "xy": [0.703, 0.096]}}
    quiet = lambda *a: None  # noqa: E731
    assert np.allclose(G.grasp_cell(grid, "C5", obs, {"grasp": {"closeup_xy": "log"}}, quiet), [0.70, 0.10, 0.18])
    assert np.allclose(G.grasp_cell(grid, "C5", obs, {"grasp": {"closeup_xy": "apply"}}, quiet), [0.703, 0.096, 0.18])
    assert np.allclose(G.grasp_cell(grid, "C5", None, {"grasp": {"closeup_xy": "apply"}}, quiet), [0.70, 0.10, 0.18])
