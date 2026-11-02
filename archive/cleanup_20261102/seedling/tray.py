"""Tray-based picking: cell grid from three taught corners, optional per-cell visual correction.

Grid model: the operator teaches the gripper contact point at the intended grasp height on an
upright seedling in three corner cells: A = first row/first column, B = first row/last column,
C = last row/first column. Every other cell is interpolated (rows A->C, columns A->B), so the
tray may be rotated arbitrarily on the table.

Cell vision needs no trained model: inside a small cylinder above the known cell, take the
stereo points in a height band just above the substrate, keep green ones (excess-green index),
and fit a 3D line (RANSAC + PCA). It returns the stem position at grasp height and the tilt, or
a reason why it could not; the caller decides whether to fall back to the nominal position.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .camera import Frame
from .config import project_path
from .grasp import GraspCandidate, approach_direction, candidate_at
from .stem3d import fit_line_ransac
from .transforms import apply_T, inv_T, unit


# ------------------------------------------------------------------ grid ---------------------------
@dataclass
class Tray:
    name: str
    rows: int
    cols: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    grasp_height_m: float  # taught grasp height above the substrate surface
    meta: dict = field(default_factory=dict)

    @staticmethod
    def cell_id(r: int, c: int) -> str:
        return f"{string.ascii_uppercase[r]}{c + 1}"

    @staticmethod
    def parse_id(cid: str) -> tuple[int, int]:
        cid = cid.strip().upper()
        return string.ascii_uppercase.index(cid[0]), int(cid[1:]) - 1

    def row_step(self) -> np.ndarray:
        return (self.C - self.A) / max(self.rows - 1, 1)

    def col_step(self) -> np.ndarray:
        return (self.B - self.A) / max(self.cols - 1, 1)

    def grasp_point(self, cid: str) -> np.ndarray:
        r, c = self.parse_id(cid)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise KeyError(f"cell {cid} outside {self.rows}x{self.cols} tray")
        return self.A + r * self.row_step() + c * self.col_step()

    def soil_z(self, cid: str) -> float:
        return float(self.grasp_point(cid)[2] - self.grasp_height_m)

    def cells(self) -> list[str]:
        return [self.cell_id(r, c) for r in range(self.rows) for c in range(self.cols)]

    def pitch_mm(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.row_step()) * 1000), float(np.linalg.norm(self.col_step()) * 1000)

    def squareness_deg(self) -> float:
        if self.rows < 2 or self.cols < 2:
            return 90.0
        return float(np.degrees(np.arccos(np.clip(unit(self.row_step()[:2]) @ unit(self.col_step()[:2]), -1, 1))))

    def order_for_approach(self, cells: list[str], cfg: dict) -> list[str]:
        """Cells nearest to the approach side first, so the fingers never sweep through unpicked neighbours."""
        def key(cid):
            p = self.grasp_point(cid)
            a = approach_direction(p, float(cfg["tray"]["approach_yaw_offsets_deg"][0]), 0.0)
            return float(p @ a)
        return sorted(cells, key=key)

    def save(self, path: str | Path) -> Path:
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "rows": self.rows, "cols": self.cols, "grasp_height_m": self.grasp_height_m,
                "A_first_row_first_col_m": self.A.round(5).tolist(), "B_first_row_last_col_m": self.B.round(5).tolist(),
                "C_last_row_first_col_m": self.C.round(5).tolist(),
                "pitch_rows_cols_mm": [round(v, 1) for v in self.pitch_mm()], "squareness_deg": round(self.squareness_deg(), 2), **self.meta}
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "Tray":
        d = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
        return Tray(d["name"], int(d["rows"]), int(d["cols"]), np.array(d["A_first_row_first_col_m"], float),
                    np.array(d["B_first_row_last_col_m"], float), np.array(d["C_last_row_first_col_m"], float), float(d["grasp_height_m"]),
                    {k: v for k, v in d.items() if k not in {"name", "rows", "cols", "grasp_height_m"}})


# ------------------------------------------------------------------ cell vision --------------------
@dataclass
class CellEstimate:
    ok: bool
    status: str  # "line" | "position_only" | reason for failure
    p_B: np.ndarray | None = None  # stem axis point at grasp height
    s_B: np.ndarray | None = None  # unit stem direction (upward)
    tilt_deg: float | None = None
    offset_mm: tuple[float, float] | None = None  # (dx, dy) relative to the nominal cell point
    n_roi: int = 0
    n_green: int = 0
    n_inliers: int = 0
    rms_mm: float | None = None
    soil_z: float | None = None
    pixel_box: tuple[int, int, int, int] | None = None

    def summary(self) -> dict:
        return {"ok": self.ok, "status": self.status,
                "p_B": None if self.p_B is None else np.round(self.p_B, 4).tolist(),
                "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 1),
                "offset_mm": None if self.offset_mm is None else [round(v, 1) for v in self.offset_mm],
                "n_roi": self.n_roi, "n_green": self.n_green, "n_inliers": self.n_inliers,
                "rms_mm": None if self.rms_mm is None else round(self.rms_mm, 2),
                "soil_z": None if self.soil_z is None else round(self.soil_z, 4)}


def excess_green(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    s = f.sum(-1) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    return 2 * g - r - b


def roi_pixel_box(frame: Frame, T_BC: np.ndarray, centre: np.ndarray, radius: float, z_lo: float, z_hi: float, pad: int = 10):
    corners = np.array([[centre[0] + dx, centre[1] + dy, z] for dx in (-radius, radius) for dy in (-radius, radius) for z in (z_lo, z_hi)])
    pc = apply_T(inv_T(T_BC), corners)
    if np.any(pc[:, 2] <= 0.05):
        return None
    uv = frame.intr.project(pc)
    u0, v0 = np.floor(uv.min(0)).astype(int) - pad
    u1, v1 = np.ceil(uv.max(0)).astype(int) + pad
    W, H = frame.intr.width, frame.intr.height
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
    if u1 <= u0 or v1 <= v0:
        return None
    return int(u0), int(v0), int(u1), int(v1)


def estimate_cell_stem(frame: Frame, T_BC: np.ndarray, nominal: np.ndarray, soil_z: float, grasp_height_m: float, cfg: dict) -> CellEstimate:
    t = cfg["tray"]
    v = t["vision"]
    # pixel box of the widest part of the cone (top of the band)
    radius = float(v["base_radius_m"]) + float(v["band_high_m"]) * np.tan(np.radians(float(v["max_tilt_deg"])))
    box = roi_pixel_box(frame, T_BC, nominal, radius, soil_z + float(v["band_low_m"]) - 0.01, soil_z + float(v["band_high_m"]))
    if box is None:
        return CellEstimate(False, "cell not in camera view")
    u0, v0, u1, v1 = box
    xyz = frame.xyz[v0:v1, u0:u1].reshape(-1, 3)
    rgb = frame.rgb[v0:v1, u0:u1].reshape(-1, 3)
    conf = frame.conf[v0:v1, u0:u1].reshape(-1) if frame.conf is not None else None
    good = np.isfinite(xyz).all(1)
    if conf is not None:
        good &= conf <= float(cfg["perception"]["max_point_confidence"])
    P = apply_T(T_BC, xyz[good])
    col = rgb[good]
    d_xy = np.linalg.norm(P[:, :2] - nominal[:2], axis=1)
    base_r = float(v["base_radius_m"])
    # substrate surface: refine the taught level from the lowest dense layer near the cell centre
    soil_pts = P[(d_xy < base_r) & (np.abs(P[:, 2] - soil_z) < 0.02)]
    soil_est = float(np.percentile(soil_pts[:, 2], 30)) if len(soil_pts) > 30 and v.get("refine_soil", True) else soil_z
    if abs(soil_est - soil_z) > 0.015:
        soil_est = soil_z  # implausible, keep the taught level
    h = P[:, 2] - soil_est
    # a cone opening upward, so a tilted stem rooted in this cell stays inside
    roi = (h > float(v["band_low_m"])) & (h < float(v["band_high_m"])) & (d_xy < base_r + np.clip(h, 0, None) * np.tan(np.radians(float(v["max_tilt_deg"]))))
    est = CellEstimate(False, "", n_roi=int(roi.sum()), soil_z=soil_est, pixel_box=box)
    sel = roi
    if v.get("green_filter", True):
        sel = roi & (excess_green(col) > float(v["exg_threshold"])) & (col.astype(np.float32).sum(1) > float(v["min_brightness"]))
    est.n_green = int(sel.sum())
    Q = P[sel]
    min_pts = int(v["min_points"])
    if len(Q) < min_pts:
        est.status = f"too few stem points ({len(Q)})"
        return est
    gh = soil_est + float(grasp_height_m)
    r_axis = float(cfg["grasp"]["default_stem_diameter_m"]) / 2
    cam = T_BC[:3, 3]
    # sequential RANSAC: several lines may be present (neighbours leaning in); keep the one rooted in this cell
    best = None
    rest = Q.copy()
    for _ in range(4):
        if len(rest) < min_pts:
            break
        fit = fit_line_ransac(rest, float(v["ransac_threshold_m"]), 200)
        if fit is None or fit[2].sum() < min_pts:
            break
        c, d, inl = fit
        d = d if d[2] >= 0 else -d
        span = float(rest[inl, 2].max() - rest[inl, 2].min())
        if d[2] > 1e-3 and span >= float(v["min_vertical_span_m"]) and d[2] >= np.cos(np.radians(float(v["max_tilt_deg"]))):
            root = c + d * (soil_est - c[2]) / d[2]
            if np.linalg.norm(root[:2] - nominal[:2]) < base_r and (best is None or inl.sum() > best[2].sum()):
                r = rest[inl] - c
                best = (c, d, inl, float(np.sqrt(np.mean(np.linalg.norm(r - np.outer(r @ d, d), axis=1) ** 2)) * 1000))
        rest = rest[~inl]
    if best is not None:
        c, d, inl, rms = best
        p = c + d * (gh - c[2]) / d[2]
        est.rms_mm, est.n_inliers = rms, int(inl.sum())
        est.s_B, est.tilt_deg, est.status = d, float(np.degrees(np.arccos(np.clip(d[2], -1, 1)))), "line"
    else:
        # no reliable line: use the lowest slice near the cell centre (the stem base is anchored there)
        low = Q[(Q[:, 2] < soil_est + float(v["band_low_m"]) + 0.015) & (np.linalg.norm(Q[:, :2] - nominal[:2], axis=1) < base_r)]
        if len(low) < min_pts // 2:
            est.status = "no stem line rooted in this cell"
            return est
        p = np.array([*np.median(low[:, :2], 0), gh])
        est.s_B, est.tilt_deg, est.status = np.array([0, 0, 1.0]), None, "position_only"
    # visible surface -> axis: push back by the stem radius, horizontally away from the camera
    view = p - cam
    view[2] = 0
    if np.linalg.norm(view) > 1e-6:
        p = p + r_axis * unit(view)
    off = (p[:2] - nominal[:2]) * 1000
    if np.linalg.norm(off) > float(v["max_correction_m"]) * 1000:
        est.status = f"correction {np.linalg.norm(off):.0f} mm above limit"
        return est
    est.ok, est.p_B, est.offset_mm = True, p, (float(off[0]), float(off[1]))
    return est


# ------------------------------------------------------------------ targeter -----------------------
class TrayTargeter:
    """Targets one tray cell. mode='blind' uses the taught position; 'vision' corrects it per cell."""

    def __init__(self, tray: Tray, cell: str, mode: str):
        if mode not in {"blind", "vision"}:
            raise ValueError(mode)
        self.tray, self.cell, self.mode = tray, cell, mode
        self.name = f"tray:{mode}"
        self.last_estimate: CellEstimate | None = None
        self.used: str = ""

    def _estimate(self, task, log, prefix: str) -> CellEstimate:
        from .viz import save_rgb
        import cv2

        cfg = task.cfg
        nominal = self.tray.grasp_point(self.cell)
        if task.T_FC is not None and not self._go_observe(task, log, nominal, prefix):
            return CellEstimate(False, "no safe observation pose")
        frame = task.cam.grab(n_average=int(cfg["camera"].get("average_frames", 1)))
        if task.T_FC is not None:  # wrist camera: pose of this (stationary) capture
            from .calibration import flange_pose

            task.T_BC = flange_pose(task.robot) @ task.T_FC
        est = estimate_cell_stem(frame, task.T_BC, nominal, self.tray.soil_z(self.cell), self.tray.grasp_height_m, cfg)
        if cfg["tray"]["vision"].get("save_frames", True):
            frame.save(log.path(f"{prefix}_frame"), save_png=False)
        img = frame.rgb.copy()
        if est.pixel_box:
            u0, v0, u1, v1 = est.pixel_box
            cv2.rectangle(img, (u0, v0), (u1, v1), (255, 255, 0), 2)
        T_CB = inv_T(task.T_BC)
        for p, colr in ((nominal, (255, 0, 0)), (est.p_B, (0, 255, 0))):
            if p is not None:
                uv = frame.intr.project(apply_T(T_CB, p)).astype(int)
                cv2.drawMarker(img, tuple(int(x) for x in uv), colr, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, f"{self.cell} {est.status} tilt={est.tilt_deg} off={est.offset_mm}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        save_rgb(log.path(f"{prefix}_cell.jpg"), img)
        log.event("OBSERVE", "cell_vision", cell=self.cell, prefix=prefix, **est.summary())
        return est

    def _go_observe(self, task, log, nominal: np.ndarray, prefix: str) -> bool:
        """Wrist camera: move (joint motion) to an observation pose looking at the cell, then let it settle."""
        import time

        from .wrist import plan_observation

        cfg = task.cfg
        o = cfg["tray"]["observe"]
        floor = self.tray.soil_z(self.cell) + float(o["min_clearance_m"])
        plan = plan_observation(nominal, task.robot.q(), task.T_FC, floor, cfg)
        log.event("OBSERVE", "observation_pose", cell=self.cell, prefix=prefix, ok=plan.ok, desc=plan.desc,
                  max_joint_step_deg=round(plan.max_joint_step_deg, 1), lowest_z=round(plan.lowest_z, 3), lowest_part=plan.lowest_part,
                  tried=plan.tried)
        if not plan.ok:
            return False
        from .robot import MotionResult, UnsafeTarget

        try:
            r = task.robot.move_joints(plan.q, dynamics=float(cfg["robot"]["dynamics"]), description=f"observe {self.cell} ({plan.desc})")
        except UnsafeTarget as exc:
            r = MotionResult(False, f"unsafe_target: {exc}")
        log.event("OBSERVE", "motion", desc="observation pose", ok=r.ok, reason=r.reason)
        if not r.ok:
            return False
        time.sleep(float(o.get("settle_s", 0.5)))
        return True

    def can_refine(self, task) -> bool:
        return False  # the arm at pre-grasp usually hides the cell from a fixed camera

    def acquire(self, task, log, attempt: int) -> tuple[GraspCandidate | None, str]:
        cfg = task.cfg
        nominal = self.tray.grasp_point(self.cell)
        p, s, self.used = nominal, np.array([0, 0, 1.0]), "nominal"
        if self.mode == "vision":
            est = self._estimate(task, log, f"a{attempt}_cell")
            self.last_estimate = est
            if est.ok:
                p, s, self.used = est.p_B, est.s_B, f"vision_{est.status}"
            elif cfg["tray"]["vision"]["fallback"] == "reject":
                return None, f"cell vision failed: {est.status}"
            else:
                self.used = f"nominal_fallback ({est.status})"
        opts = [(float(y), float(pt)) for pt in cfg["tray"]["approach_pitch_deg"] for y in cfg["tray"]["approach_yaw_offsets_deg"]]
        cand = candidate_at(p, s, cfg, q_seed=task.robot.q(), approach_options=opts, T_FC=task.T_FC)
        log.event("OBSERVE", "tray_target", cell=self.cell, mode=self.mode, used=self.used, nominal=np.round(nominal, 4),
                  target=np.round(p, 4), feasible=cand.feasible, reason=cand.reject_reason)
        return (cand, "") if cand.feasible else (None, f"{self.cell}: {cand.reject_reason}")

    def refine(self, task, log, attempt, target):
        return "inconclusive", None, np.inf

    def still_at_origin(self, task, log, attempt: int, target) -> bool | None:
        if task.T_BC is None or task.cam is None or task.T_FC is not None:
            return None  # no camera, or a wrist camera (it moves with the plant in the gripper)
        est = self._estimate(task, log, f"a{attempt}_post_extract")
        # a stem still rooted in this cell => plant not removed (neighbours leaning in do not count)
        return bool(est.ok)
