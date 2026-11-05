"""RGB-D frame acquisition (ZED 2i), saving and offline replay.

A ``Frame`` always holds:
  rgb   (H, W, 3) uint8, RGB order, rectified left image
  xyz   (H, W, 3) float32, metres, camera frame C (x right, y down, z forward), NaN = invalid
  conf  (H, W) float32 ZED confidence (1 best .. 100 worst) or None
and optionally rgb_right, the rectified right image (same intrinsics; right camera at +baseline_m along x).
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def project(self, pts_c: np.ndarray) -> np.ndarray:
        pts_c = np.asarray(pts_c, dtype=float)
        return np.stack([self.fx * pts_c[..., 0] / pts_c[..., 2] + self.cx, self.fy * pts_c[..., 1] / pts_c[..., 2] + self.cy], -1)


@dataclass
class Frame:
    rgb: np.ndarray
    xyz: np.ndarray
    conf: np.ndarray | None
    intr: Intrinsics
    meta: dict[str, Any] = field(default_factory=dict)
    gt_masks: list[np.ndarray] | None = None  # only for synthetic frames
    rgb_right: np.ndarray | None = None  # rectified right image (meta["baseline_m"] gives the baseline)

    @property
    def depth(self) -> np.ndarray:
        return self.xyz[..., 2]

    def save(self, path: str | Path, save_png: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            rgb=self.rgb,
            xyz=self.xyz.astype(np.float32),
            conf=self.conf if self.conf is not None else np.zeros(0, np.float32),
            intr=np.array([self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy, self.intr.width, self.intr.height]),
            meta=json.dumps(self.meta),
            rgb_right=self.rgb_right if self.rgb_right is not None else np.zeros(0, np.uint8),
        )
        if save_png:
            cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        return path.with_suffix(".npz")

    @staticmethod
    def load(path: str | Path) -> "Frame":
        d = np.load(Path(path).with_suffix(".npz"), allow_pickle=False)
        fx, fy, cx, cy, w, h = d["intr"].tolist()
        conf = d["conf"] if d["conf"].size else None
        right = d["rgb_right"] if "rgb_right" in d.files and d["rgb_right"].size else None
        return Frame(d["rgb"], d["xyz"], conf, Intrinsics(fx, fy, cx, cy, int(w), int(h)), json.loads(str(d["meta"])), rgb_right=right)


def xyz_from_depth(depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    return np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], -1)


def load_legacy_sample(sample_dir: str | Path, intrinsics_json: str | Path | None = None) -> Frame:
    """Load an old dataset sample folder (stem_rgb.png + stem_depth_mm.npy)."""
    sample_dir = Path(sample_dir)
    bgr = cv2.imread(str(sample_dir / "stem_rgb.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(sample_dir / "stem_rgb.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_mm = np.load(sample_dir / "stem_depth_mm.npy").astype(np.float32)
    intrinsics_json = Path(intrinsics_json) if intrinsics_json else sample_dir.parent / "zed_intrinsics.json"
    lc = json.loads(intrinsics_json.read_text())["left_camera"]
    h, w = rgb.shape[:2]
    intr = Intrinsics(lc["fx"], lc["fy"], lc["cx"], lc["cy"], w, h)
    meta = {"source": "legacy", "sample": str(sample_dir), "intrinsics_file": str(intrinsics_json),
            "warning": "intrinsics recorded 2026-05-19, assumed valid for this sample"}
    return Frame(rgb, xyz_from_depth(depth_mm / 1000.0, intr), None, intr, meta)


def load_frame_any(path: str | Path) -> Frame:
    path = Path(path)
    if path.is_dir() and (path / "stem_rgb.png").exists():
        return load_legacy_sample(path)
    if path.suffix == ".npz" or path.with_suffix(".npz").exists():
        return Frame.load(path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        h, w = bgr.shape[:2]
        intr = Intrinsics(1061.74, 1061.74, w / 2, h / 2, w, h)
        xyz = np.full((h, w, 3), np.nan, np.float32)
        return Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), xyz, None, intr, {"source": "rgb_only", "path": str(path)})
    raise ValueError(f"unrecognised frame source: {path}")


def save_in_background(jobs) -> None:
    """Write frames without waiting: [(frame, path, save_png), ...] saved by a forked child (compressing a 26 MB frame
    took ~1 s per file in the middle of every locate / close-up, 2026-11-04); in this process if fork is not possible"""
    import os

    try:
        pid = os.fork()
    except (OSError, AttributeError):
        pid = -1
    if pid == 0:  # the child: write and leave at once (no cleanup of the parent's resources)
        try:
            for f, path, png in jobs:
                f.save(path, save_png=png)
        finally:
            os._exit(0)
    if pid < 0:
        for f, path, png in jobs:
            f.save(path, save_png=png)


class RemoteCamera:
    """Frames from scripts/63_camera_server.py (the ZED kept open for a whole run)."""

    def __init__(self, sock: str, cfg: dict):
        self.sock, self.cfg = sock, cfg

    def grab(self, n_average: int = 1, timeout_s: float = 15.0, with_right: bool = True) -> Frame:
        import io
        import socket
        import struct

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        s.connect(self.sock)
        try:
            s.sendall((json.dumps({"n_average": int(n_average), "with_right": bool(with_right)}) + "\n").encode())
            hdr = b""
            while len(hdr) < 8:
                hdr += s.recv(8 - len(hdr))
            n = struct.unpack("<Q", hdr)[0]
            chunks, got = [], 0
            while got < n:
                c = s.recv(min(1 << 22, n - got))
                if not c:
                    raise ConnectionError("camera server closed the connection")
                chunks.append(c)
                got += len(c)
        finally:
            s.close()
        data = b"".join(chunks)
        if data[:3] == b"ERR":
            raise RuntimeError(f"camera server: {data[3:].decode(errors='replace')}")
        d = np.load(io.BytesIO(data), allow_pickle=False)
        fx, fy, cx, cy, w, h = d["intr"]
        conf = d["conf"] if d["conf"].size else None
        rgb_r = d["rgb_right"] if d["rgb_right"].size else None
        return Frame(d["rgb"], d["xyz"], conf, Intrinsics(fx, fy, cx, cy, int(w), int(h)), json.loads(str(d["meta"])), rgb_right=rgb_r)

    def close(self) -> None:
        pass


class ZedCamera:
    """Thin pyzed wrapper returning metric XYZ in the IMAGE coordinate system. With SEEDLING_CAMERA_SOCK set (a run of
    72 with scripts/63_camera_server.py), a RemoteCamera instead: the camera stays open between pictures."""

    def __new__(cls, cfg: dict, *a, **k):
        import os

        sock = os.environ.get("SEEDLING_CAMERA_SOCK")
        if sock and os.path.exists(sock):
            return RemoteCamera(sock, cfg)
        return super().__new__(cls)

    def __init__(self, cfg: dict):
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError(
                "pyzed not importable. The account needs access to /usr/local/zed (group 'zed') and the "
                "pyzed wheel installed into .venvs/pipeline (see tools/install_pyzed.sh)."
            ) from exc
        self.sl = sl
        c = cfg["camera"]
        self.cfg = c
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, c["resolution"])
        init.camera_fps = int(c["fps"])
        init.depth_mode = getattr(sl.DEPTH_MODE, c["depth_mode"])
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init.depth_minimum_distance = float(c["depth_min_m"])
        init.depth_maximum_distance = float(c["depth_max_m"])
        if c.get("serial_number"):
            init.set_from_serial_number(int(c["serial_number"]))
        self.cam = sl.Camera()
        err = self.cam.open(init)
        self.open_warning = ""
        if err == getattr(sl.ERROR_CODE, "POTENTIAL_CALIBRATION_ISSUE", None):
            # the camera is open; the SDK's self-check doubts the stereo calibration for this scene
            self.open_warning = str(err)
            print(f"[camera] warning: {err} (camera opened; depth may be less accurate)")
        elif err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        self.rt = sl.RuntimeParameters()
        self.rt.confidence_threshold = int(c["confidence_threshold"])
        self.rt.texture_confidence_threshold = int(c["texture_confidence_threshold"])
        info = self.cam.get_camera_information()
        cc = info.camera_configuration
        lc = cc.calibration_parameters.left_cam
        self.intr = Intrinsics(float(lc.fx), float(lc.fy), float(lc.cx), float(lc.cy), int(cc.resolution.width), int(cc.resolution.height))
        self.base_meta = {
            "source": "zed",
            "serial": int(info.serial_number),
            "model": str(info.camera_model),
            "sdk_version": str(sl.Camera.get_sdk_version()),
            "resolution": c["resolution"],
            "depth_mode": c["depth_mode"],
            "confidence_threshold": int(c["confidence_threshold"]),
            "units": "m",
            "coordinate_system": "IMAGE (x right, y down, z forward)",
            "open_warning": self.open_warning,
            "baseline_m": float(cc.calibration_parameters.get_camera_baseline()),
        }
        self._img, self._img_r, self._xyz, self._conf = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
        for _ in range(int(c.get("warmup_frames", 0))):
            self.cam.grab(self.rt)

    def grab(self, n_average: int = 1, timeout_s: float = 5.0, with_right: bool = True) -> Frame:
        sl = self.sl
        xyzs, confs, rgb, rgb_r, ts = [], [], None, None, None
        t0 = time.time()
        for _ in range(int(self.cfg.get("flush_frames", 0))):  # drop frames buffered before the arm stopped
            self.cam.grab(self.rt)
        while len(xyzs) < max(1, n_average):
            if time.time() - t0 > timeout_s:
                raise TimeoutError("ZED grab timeout")
            if self.cam.grab(self.rt) != sl.ERROR_CODE.SUCCESS:
                continue
            self.cam.retrieve_image(self._img, sl.VIEW.LEFT)
            self.cam.retrieve_measure(self._xyz, sl.MEASURE.XYZ)
            self.cam.retrieve_measure(self._conf, sl.MEASURE.CONFIDENCE)
            if rgb is None:
                bgra = self._img.get_data()
                rgb = cv2.cvtColor(np.ascontiguousarray(bgra[:, :, :3]), cv2.COLOR_BGR2RGB)
                ts = self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                if with_right:  # same instant as the left image
                    self.cam.retrieve_image(self._img_r, sl.VIEW.RIGHT)
                    rgb_r = cv2.cvtColor(np.ascontiguousarray(self._img_r.get_data()[:, :, :3]), cv2.COLOR_BGR2RGB)
            xyzs.append(np.array(self._xyz.get_data()[:, :, :3], dtype=np.float32, copy=True))
            confs.append(np.array(self._conf.get_data(), dtype=np.float32, copy=True))
        with warnings.catch_warnings():  # pixels invalid in every frame stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            xyz = xyzs[0] if len(xyzs) == 1 else np.nanmedian(np.stack(xyzs), axis=0).astype(np.float32)
            conf = confs[0] if len(confs) == 1 else np.nanmedian(np.stack(confs), axis=0).astype(np.float32)
        xyz[~np.isfinite(xyz).all(-1)] = np.nan
        off = float(self.cfg.get("disparity_offset_px", 0.0))
        if off:  # measured stereo bias: disparity too large by `off` px; points move along their pixel rays
            fB = self.intr.fx * self.base_meta["baseline_m"]
            with np.errstate(divide="ignore", invalid="ignore"):
                z = xyz[..., 2]
                z_true = fB / (fB / z - off)
                xyz *= np.where(z_true > 0, z_true / z, np.nan)[..., None]
        meta = dict(self.base_meta, disparity_offset_px=off, timestamp_ms=int(ts), n_average=len(xyzs), wall_time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return Frame(rgb, xyz, conf, self.intr, meta, rgb_right=rgb_r)

    def close(self) -> None:
        self.cam.close()


class ReplayCamera:
    """Serves saved frames (npz or legacy folders) in order; used for offline/sim runs."""

    def __init__(self, paths: list[str | Path], loop: bool = True):
        if not paths:
            raise ValueError("ReplayCamera needs at least one frame")
        self.paths = [Path(p) for p in paths]
        self.loop = loop
        self.i = 0
        self.intr = load_frame_any(self.paths[0]).intr

    def grab(self, n_average: int = 1) -> Frame:
        if self.i >= len(self.paths):
            if not self.loop:
                raise StopIteration
            self.i = 0
        f = load_frame_any(self.paths[self.i])
        self.i += 1
        return f

    def close(self) -> None:
        pass
