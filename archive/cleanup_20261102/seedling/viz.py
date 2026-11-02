"""Debug overlays and point-cloud export."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .camera import Frame
from .grasp import GraspCandidate, finger_boxes_G
from .transforms import apply_T, inv_T


def _put(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_perception(frame: Frame, instances, geoms, cands: list[GraspCandidate] | None = None, T_BC=None, cfg=None, lines=None) -> np.ndarray:
    img = frame.rgb.copy()
    for inst in instances:
        col = np.array([0, 200, 255], np.float32)
        img[inst.mask] = (0.6 * img[inst.mask] + 0.4 * col).astype(np.uint8)
    for geo in geoms:
        if len(geo.path_uv):
            cv2.polylines(img, [geo.path_uv.reshape(-1, 1, 2).astype(np.int32)], False, (255, 255, 0), 1)
        for seg in geo.segments:
            u, v = (int(round(x)) for x in seg.pixel_center)
            color = (0, 255, 0) if seg.ok else (255, 60, 60)
            cv2.circle(img, (u, v), 4 if seg.ok else 2, color, -1)
            if seg.ok:
                _put(img, f"{geo.instance}.{seg.seg_index} d={seg.diameter_m*1000:.1f}mm z={seg.p_C[2]:.3f}", (u + 8, v + 4), 0.45, color)
    if cands and T_BC is not None:
        T_CB = inv_T(T_BC)
        best = next((c for c in cands if c.feasible), None)
        if best is not None:
            draw_gripper(img, frame, best, T_CB, cfg)
    y = 26
    for line in lines or []:
        _put(img, line, (12, y), 0.65)
        y += 26
    return img


def draw_gripper(img, frame: Frame, cand: GraspCandidate, T_CB: np.ndarray, cfg: dict) -> None:
    intr = frame.intr
    for T, col in ((cand.T_BG_pre, (255, 160, 0)), (cand.T_BG, (255, 0, 255))):
        T_CG = T_CB @ T
        for lo, hi in finger_boxes_G(cfg["grasp"], float(cfg["gripper"]["open_width_m"])):
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pc = apply_T(T_CG, corners)
            if np.any(pc[:, 2] <= 0.01):
                continue
            uv = intr.project(pc).astype(np.int32)
            hull = cv2.convexHull(uv)
            cv2.polylines(img, [hull], True, col, 2)
        o = apply_T(T_CG, np.zeros(3))
        if o[2] > 0.01:
            for axis, ac in ((0, (255, 0, 0)), (1, (0, 255, 0)), (2, (0, 0, 255))):
                tip = apply_T(T_CG, np.eye(3)[axis] * 0.02)
                cv2.line(img, tuple(intr.project(o).astype(int)), tuple(intr.project(tip).astype(int)), ac, 2)


def save_rgb(path: str | Path, rgb: np.ndarray, max_width: int = 1920) -> None:
    h, w = rgb.shape[:2]
    if w > max_width:
        rgb = cv2.resize(rgb, (max_width, int(h * max_width / w)))
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def save_ply(path: str | Path, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(pts, np.float32)
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for i, p in enumerate(pts):
            if colors is not None:
                c = colors[i]
                fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
