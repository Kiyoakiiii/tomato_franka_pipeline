"""The side grasp's shared rules, used by the one-plant planner (scripts/pick.py) and the program
(scripts/plan.py, seedling/program.py): where the fingers close (the cell centre, or the close-up's), how high
(the plug's soil surface and the closing height above it), what the open hand passes on its way in (the other plants'
stems, the soil under the finger tracks) and where the camera looks from before. Until 2026-11-11 each script had its
own copy of these; they had drifted only in their messages.

No robot, no camera here: numbers in, numbers out (tests/test_grasp.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# closing point above the soil surface (m): measured close up / only estimated / leaves lying on the plug
HEIGHT_SURE = 0.017     # finger bottom ~2.5 mm above the soil: the stem grasped where it leaves it (25 until 2026-10-28: D4 missed)
HEIGHT_EST = 0.035      # 25 until 2026-10-26: C4 grasped too low
HEIGHT_LEAFY = 0.0145   # C4 2026-10-28: 43% leaves over the plug, grasped 18 mm up the fingers held only leaves
RIM = 0.020             # closing point at least this far above the tray top (the fingers pass over the cell walls)
PLUG_DZ = 0.013         # plug top above the tray top when nothing was measured
DISAGREE = 0.008        # overview and close-up soil further apart than this: the higher is taken
OFF_CENTRE = 0.012      # close-up soil patch further than this off the cell centre: the plug's rim or a neighbour seen
LEAFY_COVER = 0.35      # share of green over the plug top above which it counts as leafy
FINGER_BELOW = 0.0145   # finger bottom below the closing point
FINGER_CLEARANCE = 0.003  # finger bottom above the soil under its track
STANDOFF = 0.060        # pre-grasp behind the grasp
OBSERVE_DISTANCE = 0.34  # camera to the cell, looking down (ZED: 0.3 m at the least)


def grasp_cell(grid: dict, cell: str, obs: dict | None, cfg: dict, log=print) -> np.ndarray:
    """the cell centre to grasp at (x, y, tray-top z): the close-up's when grasp.closeup_xy is "apply" and it is sound
    (seedling/closeup.py), else the overview's"""
    c = np.array(grid["cells"][cell], float)
    if obs is not None:
        from .closeup import grasp_xy

        xy, why = grasp_xy(obs, cfg, grid["cells"][cell])
        log(why)
        if xy is not None:
            c = np.array([xy[0], xy[1], c[2]])
    return c


def soil_info(grid: dict, cell: str, obs: dict | None, tray_top: float, disagree: float = DISAGREE, plug_dz: float = PLUG_DZ,
              leafy_cover: float = LEAFY_COVER, heights: tuple = (HEIGHT_SURE, HEIGHT_EST, HEIGHT_LEAFY)) -> dict:
    """the soil surface of this plug and the closing height above it:
    - the close-up's, unless its soil patch lay off the cell centre and higher than the overview's (D6 2026-10-28:
      +27 mm, 15 mm off centre, grasped above the stem): the overview's then, still counted as measured;
    - both, further apart than `disagree`: the higher (measured only if that is the close-up's: C5 2026-10-28);
    - the overview's alone: never below the usual plug height (it reads low: from far, at an angle, under leaves);
    - nothing: tray top + plug_dz.
    heights: closing height when measured / estimated / measured with leaves over more than leafy_cover of the plug.
    soil_low: the lowest reading, for the plug's length below the held stem (2026-11-06: the close-ups read 13-16 mm
    above the overview, the higher taken, the plug 20 mm longer than planned, pushed onto the hole edge)."""
    h_sure, h_est, h_leafy = heights
    z_ov = (grid.get("soil_z_overview") or {}).get(cell)
    m = (obs or {}).get("measure") or {}
    z_near = m.get("soil_z") if m.get("ok") else None
    off = float(np.hypot(*m.get("shift_mm", [0.0, 0.0]))) / 1000.0 if obs else 0.0
    fb = tray_top + plug_dz
    if z_near is not None and z_ov is not None and off > OFF_CENTRE and z_near > z_ov + 0.003:
        soil, sure = z_ov, True
        src = f"overview {z_ov:.3f}: the close {z_near:.3f} came from soil {off * 1000:.0f} mm off the cell centre"
    elif z_near is not None and z_ov is not None and abs(z_near - z_ov) > disagree:
        soil = max(z_near, z_ov)
        sure = soil == z_near
        src = f"overview {z_ov:.3f} and close {z_near:.3f} disagree: the higher"
    elif z_near is not None:
        soil, sure = z_near, True
        src = "close measurement" + (f" (overview {z_ov:.3f})" if z_ov is not None else "")
    elif z_ov is not None:
        sure = False
        soil, src = (z_ov, "overview measurement (no close one)") if z_ov >= fb else \
            (fb, f"overview {z_ov:.3f} (no close one) below the usual plug height: tray top + {plug_dz * 1000:.0f} mm")
    else:
        soil, sure, src = fb, False, f"NOT MEASURED: tray top + {plug_dz * 1000:.0f} mm"
    height = h_sure if sure else h_est
    cov = (obs or {}).get("leaf_cover")
    leafy = bool(sure and cov is not None and cov > leafy_cover)
    if leafy:
        height = min(height, h_leafy)
    low = min(v for v in (z_near, z_ov, soil) if v is not None)
    return {"soil": float(soil), "soil_low": float(low), "src": src, "sure": bool(sure), "height": float(height),
            "leafy": leafy, "leaf_cover": cov}


def soil_points(npz: Path, q_obs, T_FC: np.ndarray, cfg: dict) -> np.ndarray | None:
    """the close-up's soil points (peat, no plant) in the base frame, or None without the picture"""
    npz = Path(npz)
    if not npz.exists():
        return None
    from . import kinematics
    from .camera import Frame
    from .plugsense import substrate_mask
    from .stemline import _mask
    from .transforms import apply_T

    fr = Frame.load(npz)
    ok = np.isfinite(fr.xyz).all(-1) & substrate_mask(fr.rgb, cfg["stemline"]) & ~_mask(fr.rgb, cfg["stemline"])
    return apply_T(kinematics.fk_flange(np.asarray(q_obs, float)) @ T_FC, fr.xyz[ok])


def finger_floor(P: np.ndarray | None, c, yaw: float, clearance: float = FINGER_CLEARANCE) -> float | None:
    """the lowest closing-point height whose open finger bottoms clear the soil under both finger tracks, approaching
    along yaw (None: no soil data there). The open fingers (+-22 mm across, a finger wide) pass over the neighbouring
    plugs, which can stand higher than this one (D5 2026-10-28: grasped 17 mm above its soil, the fingers hit a plug
    6-9 mm higher and moved the tray); the tracks from 45 mm behind the closing point to 15 mm past it, taken as flat"""
    if P is None:
        return None
    ah = np.array([np.cos(yaw), np.sin(yaw)])
    d = P[:, :2] - np.asarray(c, float)[:2]
    s_ = d @ ah  # ahead of the closing point (+) / behind it, towards the robot (-)
    lat = np.abs(d @ np.array([-ah[1], ah[0]]))
    m = (lat > 0.012) & (lat < 0.034) & (s_ > -0.045) & (s_ < 0.015)
    if m.sum() < 30:
        return None
    return float(np.percentile(P[m, 2] + clearance + FINGER_BELOW, 98))


def hand_clearance(c, stems, yaw: float, standoff: float = STANDOFF) -> float:
    """how far the other plants' stems (xy) stay from what the open hand sweeps on its way in, in the plane of the
    grasp: the open fingers (a finger outside each side of the 44 mm gap: 31 mm either side) from 15 mm past the stem
    to 75 mm behind it, the gripper body (45 mm either side) behind that, the pre-grasp standoff added (inf: no stems).
    2026-11-05: C9 (the right end of its row) was approached from the left past its neighbour C8"""
    c = np.asarray(c, float)[:2]
    back = -np.array([np.cos(yaw), np.sin(yaw)])
    lat_ax = np.array([-back[1], back[0]])
    boxes = [(-0.015, 0.075, 0.031), (0.075, 0.18 + standoff, 0.045)]  # (from, to behind the stem, half width)
    best = np.inf
    for o in stems:
        d = np.asarray(o, float)[:2] - c
        s_, l_ = float(d @ back), abs(float(d @ lat_ax))
        for s0, s1, hw in boxes:
            best = min(best, float(np.hypot(max(0.0, s0 - s_, s_ - s1), max(0.0, l_ - hw))))
    return best


def observe_pose(cfg: dict, grid: dict, cell: str, T_FC: np.ndarray, tray_top: float, q_from=None, elevations=(65, 55, 75),
                 distance: float = OBSERVE_DISTANCE, plug_dz: float = PLUG_DZ, valid=None, near_only: bool = False):
    """the observation pose for this cell (camera `distance` from the plug top, looking down at one of `elevations`,
    the least joint change from q_from winning, default the overview pose): seedling.wrist.plan_observation.
    near_only: only the joint solutions reached from q_from (the global search with random starts took seconds and
    found poses far from the hand's configuration, from which the turn failed anyway)"""
    import copy

    from .wrist import plan_observation

    c = np.array(grid["cells"][cell], float)
    guess = (grid.get("soil_z_overview") or {}).get(cell) or tray_top + plug_dz
    c2 = copy.deepcopy(cfg)
    # all elevations compared: the least joint change wins (a 65 deg view can need a 70 deg wrist turn where an
    # 80 deg view needs 45); 2 deg of joint change per place down the list keeps the first on a near tie
    c2["tray"]["observe"].update(distance_m=distance, elevation_deg=list(elevations), azimuth_offsets_deg=[0, -20, 20, -40, 40],
                                 min_clearance_m=0.15, all_elevations=True, elevation_penalty_deg=2.0)
    q0 = np.array(cfg["tray"]["overview_q"], float) if q_from is None else np.asarray(q_from, float)
    return plan_observation(np.array([c[0], c[1], guess]), q0, T_FC, guess + 0.15, c2, valid=valid,
                            _global_ik=False if near_only else None)
