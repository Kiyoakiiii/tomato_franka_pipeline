"""One collision model of the arm and what is around it, for every check of every motion.

- The arm: Franka's own FR3 description (data/third_party/fr3_build/fr3.urdf, from franka_description), the
  collision mesh of each link as its convex hull (coal), placed by pinocchio (the same kinematics as
  seedling.kinematics, joint offsets included: checked to 1e-15).
- Attached to the flange: the camera housing (camera.housing_box_m in the camera frame, T_FC), the gripper body and
  fingers (gripper.body_box_m / finger_box_m in the EE frame), a held plug (a cylinder under the held stem).
- Rigid obstacles: the table; the tray as a solid block from the table up to its top (a stand under it included);
  the disc as a solid cylinder from the table up to its top.
- Soft obstacles: the plants in the tray (upright cylinders on their cells) and in the disc (on their holes).

Clearances are true distances between these shapes (coal GJK). Until 2026-11-04 the checks used a few points of
the arm with fixed clearances; the forearm then touched a disc it was modelled 10 cm above.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import kinematics
from .config import PROJECT_DIR, ee_to_contact, flange_to_ee

URDF = PROJECT_DIR / "data" / "third_party" / "fr3_build" / "fr3.urdf"

DEFAULTS = {
    "tray_plant_height_mm": 120.0, "tray_plant_radius_mm": 35.0,
    "placed_plant_height_mm": 150.0, "placed_plant_radius_mm": 20.0,  # stem and plug (holes are 57.5 mm apart; leaves are soft)
    "disc_radius_mm": 90.0,
    "env_clearance_mm": 15.0,     # arm / camera / gripper body to table, tray, disc
    "plant_clearance_mm": 10.0,   # to the plant cylinders (leaves are soft)
    "self_clearance_mm": 25.0,    # between the hand (gripper, camera, links 6-7) and links 0-4, link 5 and links 0-2
}


@lru_cache(maxsize=1)
def _robot():
    """(pinocchio model, [(parent joint, placement 4x4)] per collision mesh, convex hulls, coal, pinocchio, radii).
    Cached in .tmp/ (keyed by the URDF's modification time): reading the URDF and its meshes took ~0.5 s in every
    planning process (2026-11-04)"""
    import pickle

    import coal
    import pinocchio as pin

    if not URDF.exists():
        raise FileNotFoundError(f"{URDF}: generate it (tools/build_fr3_urdf.sh)")
    cache = PROJECT_DIR / ".tmp" / "fr3_world_cache.pkl"
    key = (str(URDF), URDF.stat().st_mtime_ns)
    try:
        d = pickle.loads(cache.read_bytes())
        if d["key"] == key:
            return d["model"], d["geoms"], d["hulls"], coal, pin, d["radii"]
    except Exception:  # noqa: BLE001  (no cache yet, or an old one)
        pass
    model, cmodel, _ = pin.buildModelsFromUrdf(str(URDF), package_dirs=[str(URDF.parent.parent)])
    hulls, radii, geoms = [], [], []
    for g in cmodel.geometryObjects:
        g.geometry.buildConvexHull(True, "Qt")
        hulls.append(g.geometry.convex)
        radii.append(float(np.linalg.norm(np.asarray(g.geometry.vertices()), axis=1).max()))
        geoms.append((int(g.parentJoint), np.array(g.placement.homogeneous)))
    try:
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps({"key": key, "model": model, "geoms": geoms, "hulls": hulls, "radii": radii}))
    except Exception:  # noqa: BLE001  (no cache: still works)
        pass
    return model, geoms, hulls, coal, pin, radii


def _box(lo_hi):
    (x0, x1), (y0, y1), (z0, z1) = lo_hi
    return np.array([x1 - x0, y1 - y0, z1 - z0]), np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])


class World:
    def __init__(self, cfg: dict, T_FC: np.ndarray, grid: dict | None = None, disc: dict | None = None, *,
                 tray_plants=None, taken_holes=None, held_below_m: float | None = None, hold_forward_m: float = 0.0):
        model, geoms, hulls, coal, pin, radii = _robot()
        self.link_r = radii
        self.model, self.data, self.geoms, self.hulls, self.coal, self.pin = model, model.createData(), geoms, hulls, coal, pin
        p = dict(DEFAULTS)
        p.update(cfg.get("world") or {})
        self.p = {k: float(v) for k, v in p.items()}
        self.cfg, self.T_FC = cfg, T_FC
        self.F, self.G = flange_to_ee(cfg), ee_to_contact(cfg)
        C = coal
        # attached shapes: (name, shape, T_flange_shape)
        att = []
        size, ctr = _box(cfg["camera"]["housing_box_m"])
        att.append(("camera", C.Box(*size), T_FC @ _T(np.eye(3), ctr)))
        gcfg = cfg["gripper"]
        size, ctr = _box(gcfg.get("body_box_m", [[-0.02, 0.02], [-0.045, 0.045], [-0.1034, 0.0]]))
        att.append(("gripper", C.Box(*size), self.F @ _T(np.eye(3), ctr)))
        size, ctr = _box(gcfg.get("finger_box_m", [[-0.009, 0.009], [-0.032, 0.032], [0.0, 0.0626]]))
        att.append(("fingers", C.Box(*size), self.F @ _T(np.eye(3), ctr)))
        if held_below_m is not None:
            # the plug hangs under the held stem (fwd ahead of the contact point), ~30 mm wide, held_below_m deep
            T_stem = self.F @ self.G @ _T(np.eye(3), [0, 0, hold_forward_m])
            att.append(("held plug", ("plug", held_below_m), T_stem))
        self.attached = att
        # obstacles: (name, kind rigid/soft, shape, T_world)
        obs = []
        tz = float((grid or {}).get("table_z", 0.0))
        obs.append(("table", "rigid", C.Box(3.0, 3.0, 0.02), _T(np.eye(3), [0.5, 0, tz - 0.01])))
        if grid is not None:
            import cv2

            cells = np.array([v[:2] for v in grid["cells"].values()], np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(cells)
            pitch = float(cfg["tray"]["model"]["pitch_m"])
            top = float(grid["z_top"])
            Rz = _Rz(np.radians(ang))
            obs.append(("tray", "rigid", C.Box(w + pitch + 0.03, h + pitch + 0.03, top - tz), _T(Rz, [cx, cy, (top + tz) / 2])))
            names = (grid.get("planted") or grid.get("occupied", [])) if tray_plants is None else tray_plants
            hgt, r = self.p["tray_plant_height_mm"] / 1000, self.p["tray_plant_radius_mm"] / 1000
            for k in names:
                x, y = grid["cells"][k][:2]
                obs.append((f"plant {k}", "soft", C.Cylinder(r, hgt), _T(np.eye(3), [x, y, top + hgt / 2])))
        if disc is not None and disc.get("ok", True) and "centre" in disc:
            dt = float(disc["z_top"])
            c = disc["centre"]
            obs.append(("disc", "rigid", C.Cylinder(self.p["disc_radius_mm"] / 1000, dt - tz), _T(np.eye(3), [c[0], c[1], (dt + tz) / 2])))
            holes = disc.get("holes", [])
            if taken_holes is None:
                free = set(disc.get("free_holes", range(1, len(holes) + 1)))
                taken_holes = [i for i in range(1, len(holes) + 1) if i not in free]
            hgt, r = self.p["placed_plant_height_mm"] / 1000, self.p["placed_plant_radius_mm"] / 1000
            for hno in taken_holes:
                x, y = holes[hno - 1][:2]
                obs.append((f"plant in H{hno}", "soft", C.Cylinder(r, hgt), _T(np.eye(3), [x, y, dt + hgt / 2])))
        self.obstacles = obs
        self._req = C.DistanceRequest()
        # per obstacle, once: coal transform, centre, bounding radius, needed clearance, kind key
        self._obs = [(name, kind, shp, C.Transform3s(T[:3, :3], T[:3, 3]), T[:3, 3].copy(), _radius(shp),
                      (self.p["env_clearance_mm"] if kind == "rigid" else self.p["plant_clearance_mm"]) / 1000,
                      "env" if kind == "rigid" else "plants") for name, kind, shp, T in obs]
        self._oc = np.array([o[4] for o in self._obs]).reshape(-1, 3)
        self._orad = np.array([o[5] + o[6] for o in self._obs])  # radius + needed clearance
        self._att_r = [None if isinstance(shp, tuple) else _radius(shp) for _, shp, _ in self.attached]

    # ---- placement of the arm's shapes -------------------------------------------------------------------------
    def _place(self, q):
        pin = self.pin
        qq = np.asarray(q, float) + kinematics.JOINT_OFFSETS
        pin.forwardKinematics(self.model, self.data, qq)
        shapes = []  # (name, shape, R, t, bounding radius)
        for i, (j, P) in enumerate(self.geoms):
            M = self.data.oMi[j].homogeneous @ P
            shapes.append((f"link{i}", self.hulls[i], M[:3, :3], M[:3, 3], self.link_r[i]))
        T_BF = kinematics.fk_flange(q)
        for (name, shp, T), r_ in zip(self.attached, self._att_r):
            Tw = T_BF @ T
            if isinstance(shp, tuple):  # the held plug: a cylinder from the stem down
                depth = shp[1]
                c = Tw[:3, 3] - np.array([0, 0, depth / 2])
                shapes.append((name, self.coal.Cylinder(0.016, depth), np.eye(3), c, depth / 2 + 0.016))
            else:
                shapes.append((name, shp, Tw[:3, :3], Tw[:3, 3], r_))
        return shapes

    def _dist(self, a, b):
        C = self.coal
        res = C.DistanceResult()
        return C.distance(a[1], C.Transform3s(a[2], a[3]), b[0], C.Transform3s(b[1], b[2]), self._req, res)

    def clearances(self, q, *, fingers: bool = True, plug: bool = True, plants: bool = True, ignore=()) -> dict:
        """smallest distance minus what it needs, per kind: 'env' (rigid obstacles), 'plants', 'self' (m; >= 0 fine),
        with the closest pair. fingers/plug False: the fingers / held plug are not checked against the obstacles
        (when they are meant to be at the plant or in the hole); ignore: obstacle names left out (the plant grasped)"""
        return self._check(q, fingers, plug, plants, ignore, None)

    def _check(self, q, fingers, plug, plants, ignore, stop_below):
        """clearances (see clearances); with stop_below set, only whether every clearance is >= stop_below: pairs
        whose bounding spheres already keep that are skipped, and it returns None at the first one below.
        Exact minimum per kind by bounding-sphere lower bounds: the pairs in increasing order of their bound, until
        the bound exceeds the smallest distance found (2026-11-05: the old loop computed ~60 exact distances per
        pose, half the planning time)"""
        C = self.coal
        S = self._place(q)
        out = {"env": (np.inf, ""), "plants": (np.inf, ""), "self": (np.inf, "")}
        movers = [s for s in S if s[0] not in ("link0", "link1") and (fingers or s[0] != "fingers") and (plug or s[0] != "held plug")]
        tfs = {s[0]: C.Transform3s(s[2], s[3]) for s in S}
        if len(self._obs):
            mc = np.array([s[3] for s in movers])
            mr = np.array([s[4] for s in movers])
            diff = mc[:, None, :] - self._oc[None, :, :]
            lb = np.sqrt((diff * diff).sum(-1)) - mr[:, None] - self._orad[None, :]  # lower bound of distance - need
            for j, (name, kind, shp, tf, _c, _r, need, key) in enumerate(self._obs):
                if name in ignore or (kind == "soft" and not plants):
                    lb[:, j] = np.inf
                elif name == "table":
                    for i, s in enumerate(movers):
                        if s[0] == "link2":
                            lb[i, j] = np.inf
            order = np.argsort(lb, axis=None)
            res = C.DistanceResult()
            for flat in order:
                i, j = divmod(int(flat), lb.shape[1])
                b = float(lb[i, j])
                if not np.isfinite(b):
                    break
                name, kind, shp, tf, _c, _r, need, key = self._obs[j]
                if stop_below is not None:
                    if b >= stop_below:
                        break
                elif b >= max(out["env"][0], out["plants"][0]):
                    break  # no pair left can come closer than the closest of either kind found so far
                if stop_below is None and b >= out[key][0]:
                    continue
                s = movers[i]
                res.clear()  # a DistanceResult keeps the smallest distance of all queries made with it
                d = C.distance(s[1], tfs[s[0]], shp, tf, self._req, res) - need
                if d < out[key][0]:
                    out[key] = (d, f"{s[0]} - {name}")
                    if stop_below is not None and d < stop_below:
                        return None
        # self: the link pairs Franka's own description does not exclude (franka_arm.srdf: adjacent or 'Never' pairs are
        # left out), and the attached camera / gripper / plug against links 0-5 (the camera came within 69 mm of the
        # forearm's axis and Franka's reflex stopped the arm, 2026-11-02)
        p = self.p
        by = {s_[0]: s_ for s_ in S}
        pairs = [(f"link{i}", f"link{j}") for i, j in ((0, 5), (0, 6), (0, 7), (1, 5), (1, 6), (1, 7), (2, 5), (2, 7))]
        pairs += [(a_, f"link{j}") for a_ in ("camera", "gripper", "fingers", "held plug") if a_ in by for j in range(6)]
        need = p["self_clearance_mm"] / 1000
        lbs = sorted((float(np.linalg.norm(by[a_][3] - by[b_][3])) - by[a_][4] - by[b_][4] - need, a_, b_) for a_, b_ in pairs)
        res = C.DistanceResult()
        for b, a_, b_ in lbs:
            if (stop_below is not None and b >= stop_below) or (stop_below is None and b >= out["self"][0]):
                break
            A, B = by[a_], by[b_]
            res.clear()
            d = C.distance(A[1], tfs[a_], B[1], tfs[b_], self._req, res) - need
            if d < out["self"][0]:
                out["self"] = (d, f"{a_} - {b_}")
                if stop_below is not None and d < stop_below:
                    return None
        # the robot refuses any target whose contact point lies outside robot.workspace_m (seedling.robot._validate):
        # every pose of every path stays inside it (2026-11-04: a carry waypoint at y = -0.509 was refused)
        ws = self.cfg["robot"]["workspace_m"]
        Tg = kinematics.fk(np.asarray(q, float), self.F) @ self.G
        m_ws = min(min(Tg[i, 3] - ws[ax][0], ws[ax][1] - Tg[i, 3]) for i, ax in enumerate("xyz"))
        out["workspace"] = (float(m_ws), "contact point near the workspace limit")
        return out

    def margins(self, q, **kw) -> dict:
        return {k: v[0] for k, v in self.clearances(q, **kw).items()}

    def free_ok(self, q) -> bool:
        return self.ok(q)

    def ok(self, q, *, fingers: bool = True, plug: bool = True, plants: bool = True, ignore=()) -> bool:
        c = self._check(q, fingers, plug, plants, ignore, 0.0)
        return c is not None and c["workspace"][0] >= 0.0

    def worst(self, qa, qb, step_deg: float = 2.0, **kw):
        """the smallest clearance (minus need) along the straight joint path, and where"""
        n = max(1, int(np.ceil(np.degrees(np.abs(np.asarray(qb) - np.asarray(qa)).max()) / step_deg)))
        best = (np.inf, "")
        for t in np.linspace(0.0, 1.0, n + 1):
            c = self.clearances(np.asarray(qa) + t * (np.asarray(qb) - np.asarray(qa)), **kw)
            for k, (v, w) in c.items():
                if v < best[0]:
                    best = (v, f"{k}: {w}")
        return best


def _radius(shp) -> float:
    """bounding-sphere radius of a coal primitive about its own origin"""
    import coal

    if isinstance(shp, coal.Box):
        return float(np.linalg.norm(np.asarray(shp.halfSide)))
    if isinstance(shp, coal.Cylinder):
        return float(np.hypot(shp.radius, shp.halfLength))
    return 1.0


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
