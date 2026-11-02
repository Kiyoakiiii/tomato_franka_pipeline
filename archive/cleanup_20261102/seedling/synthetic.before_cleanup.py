"""Synthetic RGB-D scene and a simple physical world for offline end-to-end runs.

Not a physics simulator: stems are rigid cylinders, grasp success is decided
geometrically from the finger pad volume, and extraction resistance is a
per-stem constant. It exists so the full software chain (perception -> grasp
-> state machine -> logging) can be exercised with known ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Frame, Intrinsics
from .segmentation import StemInstance
from .transforms import apply_T, inv_T, make_T, unit


@dataclass
class SimStem:
    base_B: np.ndarray  # point where the stem leaves the soil
    dir_B: np.ndarray  # unit, upward
    length: float = 0.12
    radius: float = 0.002
    extraction_force_n: float = 8.0
    attached: bool = False
    extracted: bool = False
    T_G_stem: np.ndarray | None = None  # stem pose relative to gripper while attached

    def top(self) -> np.ndarray:
        return self.base_B + self.dir_B * self.length


@dataclass
class SimWorld:
    stems: list[SimStem]
    table_z: float = 0.0
    tray_top_z: float = 0.05
    tray_xy: tuple[float, float, float, float] = (0.38, 0.62, -0.15, 0.15)  # x0, x1, y0, y1
    finger_depth: float = 0.02
    open_width: float = 0.035
    empty_width: float = 0.0
    T_BG: np.ndarray = field(default_factory=make_T)
    events: list[str] = field(default_factory=list)

    # ---- hooks used by SimRobot / SimGripper ---------------------------------------------------
    def on_moved(self, T_BG: np.ndarray) -> None:
        self.T_BG = np.asarray(T_BG).copy()
        for s in self.stems:
            if s.attached and s.T_G_stem is not None:
                T_B_stem = self.T_BG @ s.T_G_stem
                s.base_B = T_B_stem[:3, 3].copy()
                s.dir_B = T_B_stem[:3, 2].copy()
                if s.base_B[2] > self.tray_top_z + 0.01:
                    s.extracted = True

    def try_grasp(self) -> float | None:
        """Grasp the nearest stem whose axis passes between the pads (geometric rule, not physics)."""
        G_inv = inv_T(self.T_BG)
        o = self.T_BG[:3, 3]
        best, best_d = None, np.inf
        for s in self.stems:
            if s.extracted and not s.attached:
                continue
            t = np.clip((o - s.base_B) @ s.dir_B, 0.0, s.length)
            d = apply_T(G_inv, s.base_B + t * s.dir_B)
            inside = abs(d[1]) < self.open_width / 2 - s.radius and -self.finger_depth <= d[2] <= 0.01 and abs(d[0]) < 0.02
            if inside and np.linalg.norm(d) < best_d:
                best, best_d = s, float(np.linalg.norm(d))
        if best is None:
            self.events.append("grasp_missed")
            return None
        best.attached = True
        best.T_G_stem = G_inv @ make_T(_frame_from_z(best.dir_B), best.base_B)
        self.events.append("grasped")
        return self.empty_width + 2 * best.radius * 0.8

    def on_release(self) -> None:
        for s in self.stems:
            if s.attached:
                s.attached = False
                s.T_G_stem = None
                self.events.append("released")

    def motion_force(self, G_from: np.ndarray, G_to: np.ndarray) -> float:
        for s in self.stems:
            if s.attached and not s.extracted and G_to[2, 3] - G_from[2, 3] > 0.003:
                return s.extraction_force_n
        return 0.0


def _frame_from_z(z: np.ndarray) -> np.ndarray:
    z = unit(z)
    x = unit(np.cross([0, 1.0, 0], z)) if abs(z[1]) < 0.9 else unit(np.cross([1.0, 0, 0], z))
    return np.column_stack([x, np.cross(z, x), z])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """T_BC for a camera (IMAGE convention) at ``eye`` looking at ``target`` with base +z up."""
    z = unit(np.asarray(target, float) - np.asarray(eye, float))
    x = unit(np.cross(z, [0, 0, 1.0]))  # image right
    y = np.cross(z, x)  # image down
    return make_T(np.column_stack([x, y, z]), eye)


def default_sim_camera() -> tuple[np.ndarray, Intrinsics]:
    T_BC = look_at(np.array([0.50, -0.50, 0.32]), np.array([0.50, 0.0, 0.10]))
    intr = Intrinsics(708.0, 708.0, 640.0, 360.0, 1280, 720)
    return T_BC, intr


def default_world(seed: int = 0, n_stems: int = 1) -> SimWorld:
    rng = np.random.default_rng(seed)
    stems = []
    for i in range(n_stems):
        base = np.array([0.50 + rng.uniform(-0.02, 0.02), -0.06 + 0.06 * i + rng.uniform(-0.01, 0.01), 0.045])
        tilt = rng.uniform(0, 8)
        az = rng.uniform(0, 2 * np.pi)
        d = unit([np.sin(np.radians(tilt)) * np.cos(az), np.sin(np.radians(tilt)) * np.sin(az), np.cos(np.radians(tilt))])
        stems.append(SimStem(base, d, length=rng.uniform(0.10, 0.14), radius=rng.uniform(0.0016, 0.0024)))
    return SimWorld(stems)


def render(world: SimWorld, T_BC: np.ndarray, intr: Intrinsics, noise_m: float = 0.0008, seed: int = 0) -> Frame:
    rng = np.random.default_rng(seed)
    W, H = intr.width, intr.height
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dC = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], -1).reshape(-1, 3)
    dC /= np.linalg.norm(dC, axis=1, keepdims=True)
    o = T_BC[:3, 3]
    dB = dC @ T_BC[:3, :3].T
    n = len(dB)
    t_best = np.full(n, np.inf)
    label = np.zeros(n, np.int32)  # 0 none, 1 table, 2 soil, 3 leaf, 10+i stem i

    def take(t, lab):
        better = (t > 1e-3) & (t < t_best)
        t_best[better] = t[better]
        label[better] = lab

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (world.table_z - o[2]) / dB[:, 2]
        take(np.where(np.isfinite(t), t, np.inf), 1)
        t = (world.tray_top_z - 0.005 - o[2]) / dB[:, 2]
        p = o + t[:, None] * dB
        x0, x1, y0, y1 = world.tray_xy
        inside = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
        take(np.where(inside, t, np.inf), 2)
        for i, s in enumerate(world.stems):
            w = o - s.base_B
            dp = dB - np.outer(dB @ s.dir_B, s.dir_B)
            wp = w - (w @ s.dir_B) * s.dir_B
            a = np.einsum("ij,ij->i", dp, dp)
            b = 2 * dp @ wp
            c = wp @ wp - s.radius**2
            disc = b * b - 4 * a * c
            tt = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            ax = (w @ s.dir_B) + tt * (dB @ s.dir_B)
            ok = (disc > 0) & (ax >= 0) & (ax <= s.length)
            take(np.where(ok, tt, np.inf), 10 + i)
            # a leaf near the top: disc tilted 30 deg
            centre = s.base_B + s.dir_B * s.length * 0.92 + np.array([0.0, 0.0, 0.0])
            nrm = unit([0.3, 0.2, 1.0])
            tl = ((centre - o) @ nrm) / (dB @ nrm)
            pl = o + tl[:, None] * dB
            okl = np.linalg.norm(pl - centre, axis=1) < 0.03
            take(np.where(okl, tl, np.inf), 3)

    hit = np.isfinite(t_best)
    tz = np.where(hit, t_best, np.nan)
    xyz = (dC * tz[:, None]).astype(np.float32)
    xyz += (rng.normal(0, noise_m, n)[:, None] * dC).astype(np.float32)
    # textures
    rgb = np.zeros((n, 3), np.float32)
    # texture fixed to the surface (a function of the 3-D hit point), so both cameras of a stereo pair see it
    pw = o + np.where(hit, t_best, 0.0)[:, None] * dB
    tex = 8.0 * (np.sin(pw @ np.array([523.0, 311.0, 97.0])) + np.sin(pw @ np.array([-187.0, 701.0, 409.0])) * np.cos(pw @ np.array([61.0, -433.0, 877.0])))
    tex += rng.normal(0, 2, n)
    rgb[label == 1] = [150, 150, 145]
    rgb[label == 2] = [70, 50, 35]
    rgb[label == 3] = [90, 170, 70]
    rgb[label >= 10] = [60, 140, 50]
    rgb += tex[:, None]
    label_img = label.reshape(H, W)
    # stereo-like dropouts: random holes, plus thin structures lose their 1-px silhouette
    drop = rng.random(n) < 0.08
    stem_any = (label_img >= 10).astype(np.uint8)
    import cv2

    edge = (stem_any - cv2.erode(stem_any, np.ones((3, 3), np.uint8))).reshape(-1) > 0
    drop |= edge & (rng.random(n) < 0.7)
    xyz[drop | ~hit] = np.nan
    conf = np.where(np.isfinite(xyz).all(1), 15.0, 100.0).astype(np.float32)
    masks = [label_img == 10 + i for i in range(len(world.stems))]
    meta = {"source": "synthetic", "T_BC": T_BC.tolist(), "gt_stems": [{"base_B": s.base_B.tolist(), "dir_B": s.dir_B.tolist(), "radius": s.radius} for s in world.stems]}
    return Frame(np.clip(rgb, 0, 255).astype(np.uint8).reshape(H, W, 3), xyz.reshape(H, W, 3), conf.reshape(H, W), intr, meta, masks)


class SyntheticCamera:
    def __init__(self, world: SimWorld, T_BC: np.ndarray, intr: Intrinsics):
        self.world, self.T_BC, self.intr = world, T_BC, intr
        self.k = 0

    def grab(self, n_average: int = 1) -> Frame:
        self.k += 1
        return render(self.world, self.T_BC, self.intr, seed=self.k)

    def close(self) -> None:
        pass


class GTSegmenter:
    """Returns ground-truth stem masks of synthetic frames (stand-in for YOLO in simulation)."""

    def predict_frame(self, frame: Frame) -> list[StemInstance]:
        out = []
        for i, m in enumerate(frame.gt_masks or []):
            if m.sum() > 50:
                ys, xs = np.nonzero(m)
                out.append(StemInstance(i, m, 0.99, (xs.min(), ys.min(), xs.max(), ys.max())))
        return out


def tray_world(rows: int = 3, cols: int = 4, pitch: float = 0.05, origin_xy=(0.42, -0.075), soil_z: float = 0.045,
               tilt_deg: float | list[float] = 0.0, seed: int = 0, grasp_height: float = 0.035):
    """Seedlings on a cell grid (row along +x, column along +y). Returns (world, tray) where the tray's
    taught points are the exact cell centres at grasp height, i.e. ideal teaching of upright plants."""
    from .tray import Tray

    rng = np.random.default_rng(seed)
    tilts = tilt_deg if isinstance(tilt_deg, (list, tuple, np.ndarray)) else [float(tilt_deg)] * (rows * cols)
    stems = []
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        base = np.array([origin_xy[0] + r * pitch, origin_xy[1] + c * pitch, soil_z]) + np.r_[rng.uniform(-0.002, 0.002, 2), 0.0]
        tl, az = np.radians(tilts[k]), rng.uniform(0, 2 * np.pi)
        d = np.array([np.sin(tl) * np.cos(az), np.sin(tl) * np.sin(az), np.cos(tl)])
        stems.append(SimStem(base, d, length=rng.uniform(0.10, 0.13), radius=rng.uniform(0.0016, 0.0024)))
    x0, y0 = origin_xy
    world = SimWorld(stems, tray_top_z=soil_z + 0.005,
                     tray_xy=(x0 - pitch / 2, x0 + (rows - 0.5) * pitch, y0 - pitch / 2, y0 + (cols - 0.5) * pitch))
    cz = soil_z + grasp_height
    tray = Tray("synthetic", rows, cols, np.array([x0, y0, cz]), np.array([x0, y0 + (cols - 1) * pitch, cz]),
                np.array([x0 + (rows - 1) * pitch, y0, cz]), grasp_height, {"source": "synthetic"})
    return world, tray


def tray_sim_camera() -> tuple[np.ndarray, Intrinsics]:
    """Fixed camera on the robot side, looking roughly along the approach direction (+x) and down."""
    T_BC = look_at(np.array([0.17, -0.22, 0.36]), np.array([0.47, 0.0, 0.06]))
    return T_BC, Intrinsics(708.0, 708.0, 640.0, 360.0, 1280, 720)


class WristSyntheticCamera:
    """Synthetic camera on the flange: renders from wherever the (simulated) robot is at grab time."""

    def __init__(self, world: SimWorld, robot, T_FC: np.ndarray, intr: Intrinsics, cfg: dict):
        self.world, self.robot, self.T_FC, self.intr, self.cfg = world, robot, T_FC, intr, cfg
        self.k = 0

    def grab(self, n_average: int = 1) -> Frame:
        from .calibration import T_base_flange

        self.k += 1
        return render(self.world, T_base_flange(self.robot.T_BE(), self.cfg) @ self.T_FC, self.intr, seed=self.k)

    def close(self) -> None:
        pass


def sim_wrist_T_FC() -> np.ndarray:
    """Camera on the flange as measured in the lab on 2026-10-12 (rounded): beside the flange, looking along it."""
    return make_T(Rotation.from_euler("xyz", [-1.4, -1.8, 43.6], degrees=True).as_matrix(), [0.045, -0.118, 0.024])


def render_stereo(world: SimWorld, T_BC: np.ndarray, intr: Intrinsics, baseline_m: float = 0.12, seed: int = 0) -> Frame:
    """Left frame with the rectified right image (right camera at +baseline along the left camera's x)."""
    f = render(world, T_BC, intr, seed=seed)
    r = render(world, T_BC @ make_T(np.eye(3), [baseline_m, 0.0, 0.0]), intr, seed=seed + 1000)
    f.rgb_right = r.rgb
    f.meta["baseline_m"] = baseline_m
    return f
