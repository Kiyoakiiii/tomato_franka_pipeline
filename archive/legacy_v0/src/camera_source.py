from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str


@dataclass
class FrameBundle:
    rgb: np.ndarray
    depth_mm: np.ndarray | None
    intrinsics: CameraIntrinsics | None
    label: str


class OpenCVCamera:
    def __init__(self, camera_id: int = 0) -> None:
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera id {camera_id}")

    def frames(self) -> Iterator[FrameBundle]:
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            yield FrameBundle(
                rgb=rgb,
                depth_mm=None,
                intrinsics=CameraIntrinsics(
                    fx=float(w),
                    fy=float(w),
                    cx=float(w) / 2.0,
                    cy=float(h) / 2.0,
                    width=w,
                    height=h,
                    source="webcam_rough",
                ),
                label=f"webcam:{self.camera_id}",
            )

    def close(self) -> None:
        self.cap.release()


class ImageFileSource:
    def __init__(self, image_path: str | Path, depth_path: str | Path | None = None) -> None:
        self.image_path = Path(image_path)
        self.depth_path = Path(depth_path) if depth_path else None
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)

    def frames(self) -> Iterator[FrameBundle]:
        bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {self.image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = None
        if self.depth_path is not None:
            depth = np.load(self.depth_path).astype(np.float32)
        h, w = rgb.shape[:2]
        frame = FrameBundle(
            rgb=rgb,
            depth_mm=depth,
            intrinsics=CameraIntrinsics(
                fx=1061.7401123046875,
                fy=1061.7401123046875,
                cx=945.9121704101562,
                cy=536.445556640625,
                width=w,
                height=h,
                source="default_zed_hd1080",
            ),
            label=str(self.image_path),
        )
        while True:
            yield frame

    def close(self) -> None:
        return None


class ZedCamera:
    def __init__(
        self,
        resolution_name: str = "HD1080",
        fps: int = 30,
        depth_mode_name: str = "NEURAL",
    ) -> None:
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError(
                "pyzed is not installed/importable. Run this on the lab computer with the ZED SDK."
            ) from exc

        self.sl = sl
        self.zed = sl.Camera()
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, resolution_name)
        init_params.camera_fps = int(fps)
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_name)
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        info = self.zed.get_camera_information()
        cfg = info.camera_configuration
        calib = cfg.calibration_parameters.left_cam
        self.intrinsics = CameraIntrinsics(
            fx=float(calib.fx),
            fy=float(calib.fy),
            cx=float(calib.cx),
            cy=float(calib.cy),
            width=int(cfg.resolution.width),
            height=int(cfg.resolution.height),
            source=f"zed:{info.camera_model}",
        )

    def frames(self) -> Iterator[FrameBundle]:
        sl = self.sl
        while True:
            if self.zed.grab(self.runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                continue
            self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)

            image = np.array(self.image_mat.get_data(), copy=True)
            if image.shape[2] == 4:
                rgb = image[:, :, :3][:, :, ::-1]
            else:
                rgb = image[:, :, ::-1]
            depth = np.array(self.depth_mat.get_data(), copy=True).astype(np.float32)
            yield FrameBundle(rgb=rgb, depth_mm=depth, intrinsics=self.intrinsics, label="zed_left")

    def close(self) -> None:
        self.zed.close()


def make_camera_source(config: dict, source_override: str | None = None, image: str | None = None):
    source = source_override or config["camera"]["source"]
    if image:
        return ImageFileSource(image)
    if source == "zed":
        return ZedCamera(
            resolution_name=config["camera"]["zed_resolution"],
            fps=int(config["camera"]["zed_fps"]),
            depth_mode_name=config["camera"]["zed_depth_mode"],
        )
    if source == "webcam":
        return OpenCVCamera(int(config["camera"]["webcam_id"]))
    raise ValueError(f"Unknown camera source: {source}")

