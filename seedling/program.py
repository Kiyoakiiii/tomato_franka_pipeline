"""The transplant job as an industrial pick-and-place program (2026-11-05, replacing the per-plant search of 65/71).

Taught once per layout (after the tray and the disc are located):
- one arm configuration for the whole job: the joint solution of a grasp in the middle of the plants that can grasp,
  be carried flat to the disc and let go there, with the most natural posture. Every cell's grasp is reached from it
  by a straight move of the same hand (continuation), so all grasps share it (as a robot program fixes its
  configuration flags);
- one approach direction for every cell: straight out at the tray's centre, turned --approach-deg towards the
  robot's right (the plants are taken right to left, so the hand comes in on the emptied side);
- one hand heading per disc hole, chosen in the filling order (farthest hole first) with the plants of the holes
  filled before it in place.

Each plant is then the same template moved to its cell: stage above the pre-grasp, straight down, straight in,
close, straight up; straight up to the carry height, a straight carry of the flat hand (turning only about the
vertical) to above its hole, straight down into it, open, back and up. The only per-plant input is the soil height
(overview, then close up), which moves the template up or down. Every piece is checked against the world model.
"""
from __future__ import annotations

import numpy as np

from . import cartesian, kinematics
from .transforms import inv_T, make_T

# the geometry of the template (the values 65 / 71 used)
STANDOFF = 0.060      # pre-grasp behind the grasp
STAGE_UP = 0.120      # stage above the pre-grasp
LIFT = 0.100          # straight up after closing
HEIGHT_EST = 0.035    # closing point above an estimated soil surface
HEIGHT_SURE = 0.017   # ... above a soil surface measured close up
HEIGHT_LEAFY = 0.0145
RIM = 0.020           # closing point at least this far above the tray top
PLUG = 0.040          # plug height
ABOVE = 0.030         # plug bottom above the disc top before going in
TRANSIT = 0.150       # plug bottom above the tray top while carrying
PLACED_PLANT = 0.150  # plants in the disc: this tall
BACK = 0.060          # retreat along the approach after letting go
UP_AFTER = 0.100      # then up
MAX_TILT = 5.0        # deg, the hand from flat while the plant is held
# hand headings at a hole, from straight out (0) all the way round: letting go only needs the hand flat, and the nearest
# holes are reached more easily with the hand coming from beyond the disc (180) than from the robot's side
HEADINGS = [0.0] + [s_ * d_ for d_ in range(10, 180, 10) for s_ in (-1, 1)] + [180.0]


def flat_R(yaw: float):
    """flat hand (fingers level, camera above) approaching along yaw: (R, approach vector)"""
    a = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    across = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    return np.column_stack([np.cross(across, a), across, a]), a


class Program:
    def __init__(self, cfg: dict, T_FC: np.ndarray, grid: dict, disc: dict, approach_deg: float = 15.0):
        from .config import ee_to_contact, flange_to_ee

        self.cfg, self.T_FC, self.grid, self.disc = cfg, T_FC, grid, disc
        self.F, self.G = flange_to_ee(cfg), ee_to_contact(cfg)
        self.margin = float(cfg["robot"]["ik_joint_margin_rad"])
        self.fwd = float(cfg["gripper"]["pin_brake"].get("close_forward_m", 0.0))
        self.tray_top, self.disc_top = float(grid["z_top"]), float(disc["z_top"])
        cells = np.array([v[:2] for v in grid["cells"].values()], float)
        ctr = cells.mean(0)
        self.yaw = float(np.arctan2(ctr[1], ctr[0]) + np.radians(approach_deg))
        self.R, self.a = flat_R(self.yaw)
        self.holes = np.array(disc["holes"], float)
        # filling order: farthest hole from the robot first (the hand trails back over the nearer, still empty ones)
        self.hole_order = [int(i) + 1 for i in np.argsort(-np.hypot(self.holes[:, 0], self.holes[:, 1]))]
        # the taught configurations: q_ref (a grasp's joints), ref_T, ref (cell), posture, cells (reached),
        # hole_heading (hole -> hand heading, deg from straight out at the hole)
        self.configs = []

    # ---- the pieces ------------------------------------------------------------------------------------------------
    def G_of(self, q):
        return kinematics.fk(np.asarray(q, float), self.F) @ self.G

    def chain(self, q0, targets, **kw):
        return cartesian.chain(q0, targets, self.F, self.G, self.margin, **kw)

    def cell_poses(self, cell: str, z_close: float) -> dict:
        c = self.grid["cells"][cell]
        close = np.array([c[0], c[1], z_close])
        grasp = close - self.a * self.fwd  # the open gap centre: the fingers swing forward onto the closing point
        pre = grasp - self.a * STANDOFF
        return {"close": close, "grasp": make_T(self.R, grasp), "pre": make_T(self.R, pre),
                "stage": make_T(self.R, pre + [0, 0, STAGE_UP]), "lift": make_T(self.R, grasp + [0, 0, LIFT])}

    def grasp_joints(self, T_grasp, ci: int = 0):
        """the grasp in configuration ci of the job: a straight move of the hand from its taught grasp"""
        pcs, why = self.chain(self.configs[ci]["q_ref"], [("to the cell", T_grasp)])
        return (pcs[-1][1][-1], "") if pcs is not None else (None, f"not reachable in the arm configuration {ci + 1} ({why})")

    def config_for(self, cell: str, hole: int):
        """the first taught configuration that reaches this cell and lets go into this hole, or None"""
        for i, c in enumerate(self.configs):
            if hole in c["hole_heading"] and cell in c["cells"]:
                return i
        return None

    def template(self, q_grasp, P):
        """stage -> pre-grasp -> grasp -> lift as straight pieces, worked out backwards from the grasp (as 65)"""
        back, why = self.chain(q_grasp, [("pre-grasp", P["pre"]), ("stage", P["stage"])])
        if back is None:
            return None, why
        q_st = back[1][1][-1]
        fwd, why = self.chain(q_st, [("down", P["pre"]), ("approach", P["grasp"])])
        if fwd is None:
            return None, why
        q_gr = fwd[1][1][-1]
        up, why = self.chain(q_gr, [("lift", P["lift"])])
        if up is None:
            return None, why
        return {"stage": q_st, "pregrasp": fwd[0][1][-1], "grasp": q_gr, "lift": up[0][1][-1],
                "pieces": [("down", q_st, fwd[0][2]), ("approach", fwd[0][1][-1], fwd[1][2]), ("lift", q_gr, up[0][2])]}, ""

    def z_carry(self, z_lift, below):
        return max(self.tray_top + TRANSIT + below, z_lift, self.disc_top + PLACED_PLANT + 0.020 + below)

    def hole_poses(self, hole: int, dyaw: float, below: float, raise_m: float = 0.0):
        hxy = self.holes[hole - 1][:2]
        R_h, a_h = flat_R(float(np.arctan2(hxy[1], hxy[0]) + np.radians(dyaw)))
        z_ins = self.disc_top + below + raise_m

        def at(z):
            return make_T(R_h, np.array([hxy[0], hxy[1], z]) - a_h * self.fwd)

        T_ins = at(z_ins)
        T_back = make_T(R_h, T_ins[:3, 3] - a_h * BACK)
        return {"R": R_h, "at": at, "insert": T_ins, "above": at(self.disc_top + ABOVE + below + raise_m),
                "back": T_back, "up after": make_T(R_h, T_back[:3, 3] + [0, 0, UP_AFTER])}

    def carry(self, q_up, T_up, T_ac):
        """straight moves of the flat hand from above the tray to above the hole (moving and turning together; turning
        first; moving first), waypoints keeping the hand within MAX_TILT of flat: (joints above the hole, waypoints)"""
        p_ac = T_ac[:3, 3]
        routes = [[("carry", T_ac)],
                  [("turn", make_T(T_ac[:3, :3], T_up[:3, 3])), ("carry", T_ac)],
                  [("carry", make_T(T_up[:3, :3], p_ac)), ("turn", T_ac)]]
        for r_ in routes:
            pcs, _w = self.chain(q_up, r_, max_dev_m=0.03, max_tilt=MAX_TILT)
            if pcs is not None:
                return pcs[-1][1][-1], [w for p_ in pcs for w in p_[2]]
        return None, None

    # ---- teaching ----------------------------------------------------------------------------------------------------
    def teach(self, plants: list[str], world, world_disc, log=print, max_refs: int = 3, per_ref: int = 4) -> bool:
        """the job's arm configuration(s) and, per configuration, the hand heading of each hole it lets go into.
        Candidates: every clear joint solution of the grasp of a few plants near the middle (the most natural first);
        each is a configuration the hand reaches other cells in by straight moves. Chosen: the one reaching the most
        new holes (then the most plants, then the most natural), again until no other hole is reached: usually one,
        more when one configuration cannot reach every hole with a flat hand (2026-11-05: a disc 0.45 m out and low:
        H6 from one, H3 / H5 only from another). world: the arm at the tray (plants not checked at the grasp);
        world_disc(taken holes) -> a world with plants in those holes (for letting go)"""
        cells = np.array([self.grid["cells"][k][:2] for k in plants], float)
        mid = cells.mean(0)
        refs = sorted(plants, key=lambda k: np.hypot(*(np.array(self.grid["cells"][k][:2]) - mid)))[:max_refs]
        z_est = self.tray_top + 0.013 + HEIGHT_EST
        below = HEIGHT_EST + PLUG
        poses = {k: self.cell_poses(k, z_est) for k in plants}
        found = []
        self.configs = [None]
        for ref in refs:
            P = poses[ref]
            sols = kinematics.ik_all(P["grasp"] @ inv_T(self.G), self.F, limit_margin_rad=self.margin)
            cands = sorted(((cartesian.posture_cost(q, P["close"][:2]), q) for q in sols
                            if np.linalg.norm(self.G_of(q)[:3, 3] - P["grasp"][:3, 3]) <= 0.0015 and world.ok(q, fingers=False, plants=False)),
                           key=lambda c_: c_[0])
            for pc, q in cands[:per_ref]:
                if any(np.abs(q - f_["q_ref"]).max() < np.radians(5) and f_["ref"] == ref for f_ in found):
                    continue
                tpl, _w = self.template(q, P)
                if tpl is None:
                    continue
                self.configs[0] = {"q_ref": q}
                reach = [k for k in plants if (lambda qg: qg is not None and world.ok(qg, fingers=False, plants=False))(self.grasp_joints(poses[k]["grasp"])[0])]
                headings = self._teach_holes(tpl["lift"], below, world_disc)
                found.append({"q_ref": q, "ref_T": P["grasp"], "ref": ref, "posture": pc, "cells": reach, "hole_heading": headings})
        chosen, used, covered = [], set(), set()
        while True:
            rest = [i for i, f_ in enumerate(found) if i not in used and f_["cells"] and set(f_["hole_heading"]) - covered]
            if not rest:
                break
            b = max(rest, key=lambda i: (len(set(found[i]["hole_heading"]) - covered), len(found[i]["cells"]), -found[i]["posture"]))
            used.add(b)
            chosen.append(found[b])
            covered |= set(found[b]["hole_heading"])
        self.configs = chosen
        if not chosen:
            log("teaching: no arm configuration grasps a plant near the middle and reaches the disc with a flat hand")
            return False
        for i, c in enumerate(chosen):
            log(f"configuration {i + 1} (taught on {c['ref']}, posture {c['posture']:.0f} deg): reaches {len(c['cells'])}/{len(plants)} "
                f"plants; holes and hand headings: " + ", ".join(f"H{h} {c['hole_heading'][h]:+.0f}" for h in self.hole_order if h in c["hole_heading"]))
        log(f"approach heading {np.degrees(self.yaw):+.0f} deg (absolute) for every cell")
        return True

    def _teach_holes(self, q_lift, below, world_disc) -> dict:
        """per hole (in the filling order, the earlier ones filled), the first heading from which a flat hand carried
        from this lift pose lets go into it: joint continuity of the whole way and the release clear (no waypoints
        worked out: the plans do that)"""
        T_l = self.G_of(q_lift)
        T_up = T_l.copy()
        T_up[2, 3] = self.z_carry(T_l[2, 3], below)
        pcs, _w = self.chain(q_lift, [("up", T_up)], sparse=False)
        if pcs is None:
            return {}
        q_up = pcs[-1][1][-1]
        out, filled = {}, []
        for h in self.hole_order:
            w = world_disc(filled)
            for dy in HEADINGS:
                ok = False
                for raise_m in (0.0, 0.01, 0.02, 0.03):  # letting go higher when the arm comes too close to something there
                    H = self.hole_poses(h, dy, below, raise_m)
                    T_ac = H["at"](T_up[2, 3])
                    reached = False
                    for r_ in ([("carry", T_ac)], [("turn", make_T(T_ac[:3, :3], T_up[:3, 3])), ("carry", T_ac)]):
                        full, _w = self.chain(q_up, r_ + [("above", H["above"]), ("insert", H["insert"]), ("back", H["back"]),
                                                          ("up after", H["up after"])], sparse=False)
                        if full is None:
                            continue
                        reached = True
                        if w.ok(full[-3][1][-1], plug=False) and w.ok(full[-1][1][-1], plug=False):
                            ok = True
                            break
                    if ok or not reached:
                        break  # found, or this heading cannot be followed at all (higher does not help)
                if ok:
                    out[h] = dy
                    break
            filled.append(h)
        return out

    def layout(self) -> dict:
        g, d = self.grid, self.disc
        return {"A1": list(map(float, g["cells"]["A1"][:3])), "tray_yaw_deg": float(g.get("yaw_deg", 0.0)), "tray_top": self.tray_top,
                "disc_centre": list(map(float, d["centre"][:2])), "disc_top": self.disc_top, "disc_phase_deg": float(d.get("phase_deg", 0.0))}

    def same_layout(self, other: dict, tol_m: float = 0.005, tol_deg: float = 1.0) -> bool:
        a = self.layout()
        try:
            return (np.linalg.norm(np.subtract(a["A1"], other["A1"])) < tol_m and abs(a["tray_yaw_deg"] - other["tray_yaw_deg"]) < tol_deg
                    and abs(a["tray_top"] - other["tray_top"]) < tol_m and np.linalg.norm(np.subtract(a["disc_centre"], other["disc_centre"])) < tol_m
                    and abs(a["disc_top"] - other["disc_top"]) < tol_m and abs(a["disc_phase_deg"] - other["disc_phase_deg"]) < tol_deg)
        except (KeyError, TypeError):
            return False

    def to_dict(self) -> dict:
        return {"layout": self.layout(), "yaw": self.yaw, "hole_order": self.hole_order,
                "configs": [{"q_ref": c["q_ref"].tolist(), "ref_T": np.asarray(c["ref_T"]).tolist(), "ref": c["ref"],
                             "posture": float(c["posture"]), "cells": c["cells"],
                             "hole_heading": {str(k): v for k, v in c["hole_heading"].items()}} for c in self.configs]}

    def load(self, d: dict):
        self.yaw = float(d["yaw"])
        self.R, self.a = flat_R(self.yaw)
        self.hole_order = [int(h) for h in d.get("hole_order", self.hole_order)]
        self.configs = [{"q_ref": np.array(c["q_ref"]), "ref_T": np.array(c["ref_T"]), "ref": c["ref"], "posture": c["posture"],
                         "cells": list(c["cells"]), "hole_heading": {int(k): float(v) for k, v in c["hole_heading"].items()}}
                        for c in d.get("configs", [])]
