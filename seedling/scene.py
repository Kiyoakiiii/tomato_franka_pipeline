"""One model of what is around the arm, for every free move (turning to the grasp, carrying, going back).

Before 2026-11-04 each script had its own rule: 65 kept the grasp centre 12 cm over the tray top, 71 the plug bottom
5 cm over it, 68 every part above z = 0.30 m wherever it was. The last one could not leave a pose just above a disc
close to the robot (the wrist at 0.17 m over a disc top at 0.07). Here:

- what lies under a point: the table; the tray top within the tray; the disc top within the disc;
- the plants still in the tray: upright cylinders (``tray_plant_radius_mm``, ``tray_plant_height_mm`` above the tray
  top) at their cells; the plants already in the disc: cylinders at their holes (``placed_plant_*``);
- every part of the hand stays ``clearance_mm`` above the highest of these within its reach (the wrist housings:
  ``wrist_reach_mm`` sideways, and ``wrist_clearance_mm`` over the tray, ``wrist_clearance_off_tray_mm`` elsewhere,
  the user's tested 20 mm and the 70 mm of 2026-11-02);
- a held plug's bottom also stays ``clearance_mm`` above them;
- the hand stays ``self_clearance_mm`` from the arm's own links (Franka's self-collision reflex, see wrist.self_clearance);
- the gripper contact point stays inside robot.workspace_m (the robot refuses targets outside it).

The checks of the short moves at the plant and at the hole (approach, lift, insert) stay in 65 / 71: there the hand
is meant to be among the plants.
"""
from __future__ import annotations

import numpy as np

from .config import ee_to_contact, flange_to_ee
from .wrist import Surfaces, body_points_B, self_clearance
from . import kinematics

DEFAULTS = {
    "tray_plant_height_mm": 120.0,
    "tray_plant_radius_mm": 35.0,
    "placed_plant_height_mm": 150.0,
    "placed_plant_radius_mm": 40.0,
    "clearance_mm": 20.0,
    "camera_clearance_mm": 20.0,
    "wrist_reach_mm": 0.0,
    "self_clearance_mm": 90.0,
}


class Scene:
    def __init__(self, grid: dict, disc: dict | None, cfg: dict, T_FC: np.ndarray, *, tray_plants=None,
                 taken_holes=None, held_below_m: float | None = None, hold_forward_m: float = 0.0):
        """tray_plants: the cells with a plant now (default: the grid's planted / occupied cells); taken_holes: the
        disc holes (1..6) with a plant in them now (default: those not in the disc's free_holes); held_below_m: a plug
        held this far below the gripper's closing point (None: nothing held)"""
        p = dict(DEFAULTS)
        p.update(cfg.get("scene") or {})
        g = cfg.get("grasp", {})
        p.setdefault("wrist_clearance_mm", float(g.get("side_wrist_clearance_mm", 20.0)))
        p.setdefault("wrist_clearance_off_tray_mm", float(g.get("side_wrist_clearance_off_tray_mm", 20.0)))
        self.p = {k: float(v) for k, v in p.items()}
        self.cfg, self.T_FC = cfg, T_FC
        self.F, self.G = flange_to_ee(cfg), ee_to_contact(cfg)
        self.surf = Surfaces(grid, disc, cfg)
        cells = (grid.get("planted") or grid.get("occupied", [])) if tray_plants is None else tray_plants
        self.tray_plants = np.array([grid["cells"][k][:2] for k in cells], float).reshape(-1, 2)
        ok = disc is not None and disc.get("ok", True) and "holes" in disc
        holes = np.array(disc["holes"], float).reshape(-1, 2) if ok else np.zeros((0, 2))
        if taken_holes is None:
            free = set(disc.get("free_holes", range(1, len(holes) + 1))) if ok else set()
            taken_holes = [h for h in range(1, len(holes) + 1) if h not in free]
        self.placed = np.array([holes[h - 1] for h in taken_holes], float).reshape(-1, 2)
        self.held_below, self.fwd = held_below_m, hold_forward_m
        self.tray_plant_top = self.surf.tray_top + self.p["tray_plant_height_mm"] / 1000
        self.placed_top = (self.surf.disc_top or 0.0) + self.p["placed_plant_height_mm"] / 1000

    def height(self, P: np.ndarray, reach: float = 0.0) -> np.ndarray:
        """the highest thing within `reach` (horizontally) of each point: surfaces and plants"""
        P = np.atleast_2d(P)[:, :2]
        h = self.surf.under(P, reach)
        for C, r, top in ((self.tray_plants, self.p["tray_plant_radius_mm"] / 1000, self.tray_plant_top),
                          (self.placed, self.p["placed_plant_radius_mm"] / 1000, self.placed_top)):
            if len(C):
                d = np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]).min(axis=1)
                h = np.where(d < r + reach, np.maximum(h, top), h)
        return h

    def margins(self, q: np.ndarray) -> dict[str, float]:
        """clearance of each part over what lies under it, minus what it needs (>= 0: fine), and the self clearance
        minus its minimum"""
        p = self.p
        pts = body_points_B(q, self.T_FC, self.cfg)
        wr = np.vstack([pts["arm"], pts["flange"]])
        reach = p["wrist_reach_mm"] / 1000
        over_tray = self.surf.under(wr, reach) >= self.surf.tray_top - 1e-6
        need_w = np.where(over_tray, p["wrist_clearance_mm"], p["wrist_clearance_off_tray_mm"]) / 1000
        m = {"wrist/flange": float((wr[:, 2] - self.height(wr, reach) - need_w).min())}
        cm = pts["camera housing"]
        m["camera"] = float((cm[:, 2] - self.height(cm) - p["camera_clearance_mm"] / 1000).min())
        T_BE = kinematics.fk(q, self.F)
        Gp = (T_BE @ self.G)[:3, 3]
        fl = pts["flange"][0]
        hand = np.array([fl + s * (Gp - fl) for s in (0.33, 0.67, 1.0)])  # the gripper body down to the fingers
        m["gripper"] = float((hand[:, 2] - self.height(hand) - p["clearance_mm"] / 1000).min())
        # the robot refuses any target whose contact point lies outside robot.workspace_m (seedling.robot._validate):
        # every pose of a path stays inside it (2026-11-04: a via pose 0.777 m high was refused, limit 0.60)
        ws = self.cfg["robot"]["workspace_m"]
        m["workspace"] = float(min(min(Gp[i] - ws[ax][0], ws[ax][1] - Gp[i]) for i, ax in enumerate("xyz")))
        if self.held_below is not None:
            stem = Gp + (T_BE @ self.G)[:3, 2] * self.fwd
            plug = np.array([stem[0], stem[1], stem[2] - self.held_below])
            m["held plug"] = float(plug[2] - self.height(plug)[0] - p["clearance_mm"] / 1000)
        m["self"] = float(self_clearance(q, self.T_FC, self.cfg) - p["self_clearance_mm"] / 1000)
        return m

    def free_ok(self, q: np.ndarray) -> bool:
        return min(self.margins(q).values()) >= 0.0

    def worst(self, qa: np.ndarray, qb: np.ndarray, step_deg: float = 2.0) -> tuple[float, str]:
        """the smallest margin along the straight joint path (start included) and which part"""
        n = max(1, int(np.ceil(np.degrees(np.abs(qb - qa).max()) / step_deg)))
        best = (np.inf, "")
        for t in np.linspace(0.0, 1.0, n + 1):
            for k, v in self.margins(qa + t * (qb - qa)).items():
                if v < best[0]:
                    best = (v, k)
        return best
