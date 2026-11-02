from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera_source import CameraIntrinsics


@dataclass
class StemDetection:
    mask: np.ndarray
    confidence: float
    mask_area_px: int
    grasp_pixel: tuple[int, int]
    stem_axis_deg_image: float
    gripper_axis_deg_image: float
    num_predictions: int
    depth_m: float | None = None
    depth_method: str = "not_used"
    camera_xyz_m: list[float] | None = None


def load_yolo_model(model_path: str | Path):
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return YOLO(str(model_path))


def select_yolo_device(requested: str = "auto") -> str | int:
    if requested != "auto":
        return requested

    try:
        import torch

        if not torch.cuda.is_available():
            print("[vision] CUDA is not available; using CPU for YOLO.")
            return "cpu"

        major, minor = torch.cuda.get_device_capability(0)
        device_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_arch not in supported_arches:
            print(
                "[vision] CUDA device capability "
                f"{device_arch} is not supported by this PyTorch build "
                f"({', '.join(sorted(supported_arches))}); using CPU for YOLO."
            )
            return "cpu"

        return 0
    except Exception as exc:
        print(f"[vision] CUDA auto-detect failed ({exc}); using CPU for YOLO.")
        return "cpu"


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask > 0.5
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0.5


def choose_grasp_pixel(mask: np.ndarray, height_percentile: float) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")

    target_y = np.percentile(ys, height_percentile)
    band = max(8, int(mask.shape[0] * 0.012))
    keep = np.abs(ys - target_y) <= band
    if int(keep.sum()) < 8:
        keep = np.ones_like(ys, dtype=bool)
    return int(np.median(xs[keep])), int(np.median(ys[keep]))


def mask_axis(mask: np.ndarray, center: tuple[int, int] | None = None, radius_px: int = 120) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 6:
        return 0.0, 90.0

    coords = np.column_stack([xs, ys]).astype(np.float32)
    if center is not None:
        u, v = center
        local = ((coords[:, 0] - u) ** 2 + (coords[:, 1] - v) ** 2) <= radius_px**2
        if int(local.sum()) >= 12:
            coords = coords[local]

    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    major = major / (np.linalg.norm(major) + 1e-9)

    stem_angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    stem_angle = normalize_axis_angle_deg(stem_angle)
    gripper_angle = normalize_axis_angle_deg(stem_angle + 90.0)
    return stem_angle, gripper_angle


def normalize_axis_angle_deg(angle: float) -> float:
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def robust_depth_at_mask(
    depth_mm: np.ndarray | None,
    mask: np.ndarray,
    grasp_pixel: tuple[int, int],
    radius_px: int = 18,
) -> tuple[float | None, str]:
    if depth_mm is None:
        return None, "no_depth_frame"

    u, v = grasp_pixel
    h, w = depth_mm.shape[:2]
    x1, x2 = max(0, u - radius_px), min(w, u + radius_px + 1)
    y1, y2 = max(0, v - radius_px), min(h, v + radius_px + 1)
    patch = depth_mm[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2] > 0

    valid = patch[np.isfinite(patch) & (patch > 0) & patch_mask]
    if valid.size >= 8:
        return float(np.median(valid)) / 1000.0, "local_mask_median"

    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size >= 8:
        return float(np.median(valid)) / 1000.0, "local_patch_median"

    valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0) & (mask > 0)]
    if valid.size >= 8:
        return float(np.median(valid)) / 1000.0, "full_mask_median"

    return None, "no_valid_depth"


def backproject_pixel(
    u: int,
    v: int,
    z_m: float | None,
    intrinsics: CameraIntrinsics | None,
) -> list[float] | None:
    if z_m is None or intrinsics is None:
        return None
    x = (float(u) - intrinsics.cx) * z_m / intrinsics.fx
    y = (float(v) - intrinsics.cy) * z_m / intrinsics.fy
    return [float(x), float(y), float(z_m)]


def detect_stem(
    model: Any,
    rgb: np.ndarray,
    depth_mm: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    confidence: float,
    image_size: int,
    max_detections: int,
    grasp_height_percentile: float,
    min_mask_area_px: int,
    device: str | int | None = None,
) -> StemDetection | None:
    h, w = rgb.shape[:2]

    # Ultralytics examples use OpenCV BGR arrays. Convert from our RGB frame.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    predict_kwargs = {
        "imgsz": int(image_size),
        "conf": float(confidence),
        "max_det": int(max_detections),
        "verbose": False,
    }
    if device is not None:
        predict_kwargs["device"] = device

    result = model.predict(bgr, **predict_kwargs)[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    masks = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    candidates: list[tuple[float, int, np.ndarray, int]] = []
    for idx, raw_mask in enumerate(masks):
        mask = _resize_mask(raw_mask, w, h)
        area = int(mask.sum())
        if area < min_mask_area_px:
            continue
        score = float(confs[idx]) * math.log1p(area)
        candidates.append((score, idx, mask, area))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, idx, mask, area = candidates[0]
    grasp_pixel = choose_grasp_pixel(mask, grasp_height_percentile)
    stem_angle, gripper_angle = mask_axis(mask, center=grasp_pixel)
    depth_m, depth_method = robust_depth_at_mask(depth_mm, mask, grasp_pixel)
    camera_xyz = backproject_pixel(grasp_pixel[0], grasp_pixel[1], depth_m, intrinsics)

    return StemDetection(
        mask=mask.astype(np.uint8),
        confidence=float(confs[idx]),
        mask_area_px=area,
        grasp_pixel=grasp_pixel,
        stem_axis_deg_image=stem_angle,
        gripper_axis_deg_image=gripper_angle,
        num_predictions=len(candidates),
        depth_m=depth_m,
        depth_method=depth_method,
        camera_xyz_m=camera_xyz,
    )


def draw_detection_overlay(
    rgb: np.ndarray,
    detection: StemDetection | None,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    out = rgb.copy()
    if detection is not None:
        color = np.zeros_like(out)
        color[:, :, 1] = 210
        mask = detection.mask > 0
        out[mask] = (0.65 * out[mask] + 0.35 * color[mask]).astype(np.uint8)

        u, v = detection.grasp_pixel
        cv2.circle(out, (u, v), 16, (255, 40, 40), 3)
        cv2.line(out, (u - 28, v), (u + 28, v), (255, 40, 40), 3)
        cv2.line(out, (u, v - 28), (u, v + 28), (255, 40, 40), 3)

        length = 80
        theta = math.radians(detection.stem_axis_deg_image)
        dx = int(math.cos(theta) * length)
        dy = int(math.sin(theta) * length)
        cv2.line(out, (u - dx, v - dy), (u + dx, v + dy), (255, 220, 0), 4)

    if status_lines:
        y = 26
        for line in status_lines:
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30

    return out

