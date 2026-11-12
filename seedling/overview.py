"""Everything one overview picture tells (scripts/locate.py): the tray (grid fit, cross-checked against its
outline, the outline as the fallback), the cells with a plug and their soil height, the cells with a plant over them,
and the transplant disc with its free holes. No robot, no camera, no files: the live script and the offline replay
(tools/replay_overview.py, tests/test_overview_replay.py) run the same code on a frame and its camera pose."""
from __future__ import annotations

import numpy as np

from .camera import Frame
from .traygrid import TrayGrid, locate_tray, locate_tray_outline, occupied_cells


def _cross_check(frame: Frame, T_BC: np.ndarray, cfg: dict, g: TrayGrid, log) -> TrayGrid:
    """the grid against the outline at its own top height; g corrected by whole cells if needed (or rejected)"""
    m = cfg["tray"]["model"]
    # cross-check: the outline (+ known pitch) is good to a few mm and cannot slip by whole cells; a grid
    # fit that disagrees by more than a third of a cell has locked onto the wrong cells (2026-10-22: one column)
    o = locate_tray_outline(frame, T_BC, int(m["rows"]), int(m["cols"]), cfg, pitch=float(m["pitch_m"]), top_z=g.z_top)
    if not o["ok"]:
        log("cross-check not possible:", o["status"])
        return g
    d = np.array([np.linalg.norm(np.array(g.cells[k][:2]) - np.array(o["cells"][k][:2])) for k in g.cells])
    log(f"cross-check grid vs outline: median {np.median(d) * 1000:.1f} mm, max {d.max() * 1000:.1f} mm")
    pm = float(m["pitch_m"])
    nr = nc = 0
    if np.median(d) > pm / 3:
        # the grid is good to a mm but can slip by whole cells (it is periodic); the outline is good to ~1 cm and
        # cannot: the grid moved by the whole number of cells that brings it onto the outline
        vr = (np.array(g.cells["B1"][:2]) - np.array(g.cells["A1"][:2]))  # one row
        vc = (np.array(g.cells["A2"][:2]) - np.array(g.cells["A1"][:2]))  # one column
        off = np.mean([np.array(o["cells"][k][:2]) - np.array(g.cells[k][:2]) for k in g.cells], axis=0)
        nr, nc = np.round(np.linalg.solve(np.column_stack([vr, vc]), off)).astype(int)
        moved = {k: [round(v[0] + nr * vr[0] + nc * vc[0], 4), round(v[1] + nr * vr[1] + nc * vc[1], 4), v[2]] for k, v in g.cells.items()}
        d2 = np.array([np.linalg.norm(np.array(moved[k][:2]) - np.array(o["cells"][k][:2])) for k in moved])
        # the outline decides only the whole number of cells (its rims are uneven: 2026-11-06, a tray moved
        # back 6 cm on its stand: the grid 2 rows off onto the stand, and after the 2-row move still 15 mm
        # from the outline, just over a third of a cell: the run was stopped). Rounding is unambiguous while
        # the rest is under half a cell. The grid keeps its own mm (a re-fit near the moved grid pulled it 15 mm toward the
        # robot that day and the grasp missed: grasping goes by these coordinates)
        if (nr or nc) and np.median(d2) < 0.45 * pm:
            g.cells = moved
            g.status = f"ok (grid moved by {nr:+d} rows, {nc:+d} columns onto the tray outline)"
            log(f"CORRECTED: the grid fit was {abs(nr)} row(s), {abs(nc)} column(s) off; moved onto the outline"
                f" (now median {np.median(d2) * 1000:.1f} mm from it)")
            d = d2
    if np.median(d) > (0.45 if (nr or nc) else 1 / 3) * pm:
        g.ok, g.status = False, (f"grid fit and tray outline disagree by {np.median(d) * 1000:.0f} mm (about "
                                 f"{np.median(d) / pm:.1f} cells): grid locked onto the wrong cells")
        log("REJECTED:", g.status)
    return g


def locate_grid(frame: Frame, T_BC: np.ndarray, cfg: dict, z_hint: float | None = None, method: str = "auto", log=print) -> TrayGrid:
    """the tray's cells: grid fit (+ cross-check with the outline), the outline alone as the fallback"""
    m = cfg["tray"]["model"]
    g = locate_tray(frame, T_BC, cfg, z_hint=z_hint) if method in ("auto", "grid") else None
    if g is not None:
        import json

        log("grid:", json.dumps(g.summary()))
    if g is not None and g.ok and method == "auto":
        g = _cross_check(frame, T_BC, cfg, g, log)
    if g is None or (not g.ok and method == "auto"):  # outline as the fallback (printed as such)
        import json

        o = locate_tray_outline(frame, T_BC, int(m["rows"]), int(m["cols"]), cfg, pitch=float(m["pitch_m"]))
        log("outline:", json.dumps({k: v for k, v in o.items() if k != "cells"}, default=float))
        g = TrayGrid(o["ok"], "outline: " + o["status"], o.get("cells"), o.get("rim_z"), o.get("yaw_deg"), 0.0, 0.0,
                     {"method": "outline", "corners": o.get("corners")})
    return g


def tray_contents(frame: Frame, T_BC: np.ndarray, cfg: dict, g: TrayGrid, log=print) -> dict:
    """for a located tray: the cells with a plug, each plug's soil height, the cells with a plant over them"""
    if not g.ok:
        return {"occupied": [], "planted": [], "green": [], "soil_z_overview": {}}
    occ = occupied_cells(frame, T_BC, g.cells, g.z_top, cfg)
    log(f"{len(occ)} cells with a plug: {sorted(occ)}")
    from .plugsense import measure_plug

    soil = {}
    for k in sorted(occ):  # each plug's own soil surface (it may stand above, level with or below the tray top)
        r = measure_plug(frame, T_BC, g.cells[k], cfg)
        soil[k] = r["soil_z"] if r["ok"] else None
    log("soil surface (overview), mm vs the tray top:",
        {k: (None if v is None else round((v - g.z_top) * 1000)) for k, v in soil.items()})
    # cells with a plant over them (green within 15 mm of the centre, seen from the overview): the peat test misses
    # plugs hidden under leaves; the grasp planner keeps its path clear of these too
    from .stemline import _mask
    from .transforms import apply_T, inv_T

    planted, green_cells = set(occ), set()
    green = _mask(frame.rgb, cfg["stemline"])
    Kc = frame.intr.K()
    for k, (x, y, zc) in g.cells.items():
        ring = np.array([[x + rr * np.cos(t), y + rr * np.sin(t), zc + 0.02] for rr in (0.0, 0.007, 0.014) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
        pc = apply_T(inv_T(T_BC), ring)
        uv = np.round((Kc @ (pc / pc[:, 2:3]).T).T[:, :2]).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < green.shape[1]) & (uv[:, 1] < green.shape[0])
        if inside.sum() > 10 and green[uv[inside, 1], uv[inside, 0]].mean() > 0.3:
            planted.add(k)
            green_cells.add(k)
    log("cells with a plant over them (path obstacles):", sorted(planted))
    return {"occupied": sorted(occ), "planted": sorted(planted), "green": sorted(green_cells), "soil_z_overview": soil}


def locate_disc_view(frame: Frame, T_BC: np.ndarray, prior: dict | None = None, log=print) -> dict:
    """the transplant disc, if one is in view (seedling/disc.py), with the holes that hold a plant. prior: the last saved
    disc (its position used only if it was found then; its occupied holes as they were)"""
    from .disc import locate_disc, occupied_from_view
    from .transforms import apply_T

    d = locate_disc(frame, T_BC, prior=prior if prior and prior.get("ok") else None)
    if d.get("centre") is None:
        log("disc:", d["status"])
        return d
    log(f"disc: {d['status']}; centre {np.round(d['centre'], 4).tolist()}, top z {d['z_top']:.3f}, "
        f"{len(d.get('holes_seen', []))} holes seen (rim radius mm {d.get('hole_radius_mm')}, hole circle {d.get('hole_circle_r_mm')} mm)")
    if d["ok"]:
        # the holes with a plant in them: seen above the hole, or occupied before (the disc not moved); a hole seen
        # neither free nor with a plant is unknown (not placed into, no plant assumed there)
        okp = np.isfinite(frame.xyz).all(-1)
        d["occupied_holes"] = occupied_from_view(d, apply_T(T_BC, frame.xyz[okp]), prior)
        unknown = [i for i in range(1, len(d["holes"]) + 1) if i not in d["free_holes"] and i not in d["occupied_holes"]]
        log(f"disc holes: free {d['free_holes']}, with a plant {d['occupied_holes']}"
            + (f", not seen (unknown: no plant assumed, not used) {unknown}" if unknown else ""))
    return d


def locate_overview(frame: Frame, T_BC: np.ndarray, cfg: dict, z_hint: float | None = None, prev_disc: dict | None = None,
                    method: str = "auto", log=print) -> dict:
    """grid (TrayGrid), rec (the tray record as written to tray.grid_file, without q / T_BC / K / calibration), disc (dict)"""
    g = locate_grid(frame, T_BC, cfg, z_hint=z_hint, method=method, log=log)
    rec = {**g.summary(), **tray_contents(frame, T_BC, cfg, g, log=log), "cells": g.cells}
    return {"grid": g, "rec": rec, "disc": locate_disc_view(frame, T_BC, prior=prev_disc, log=log)}
