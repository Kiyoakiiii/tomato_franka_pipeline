"""YOLO instance segmentation wrapper returning full-resolution stem masks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import project_path


@dataclass
class StemInstance:
    index: int
    mask: np.ndarray  # bool (H, W), aligned with the input image
    conf: float
    box_xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def select_device(requested) -> str | int:
    if requested not in (None, "auto"):
        return requested
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class StemSegmenter:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        p = cfg["perception"]
        self.p = p
        self.model_path = project_path(p["model"])
        self.model = YOLO(str(self.model_path))
        self.device = select_device(p.get("device", "auto"))

    def predict_frame(self, frame) -> list[StemInstance]:
        return self.predict(frame.rgb)

    def predict(self, rgb: np.ndarray) -> list[StemInstance]:
        h, w = rgb.shape[:2]
        res = self.model.predict(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            imgsz=int(self.p["imgsz"]),
            conf=float(self.p["conf"]),
            max_det=int(self.p["max_instances"]),
            retina_masks=True,
            device=self.device,
            verbose=False,
        )[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return []
        masks = res.masks.data.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        roi = self.p.get("roi_px")
        out = []
        for i, (m, c, b) in enumerate(zip(masks, confs, boxes)):
            if m.shape != (h, w):  # defensive: retina_masks should already give full resolution
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = m > 0.5
            if roi:
                x0, y0, x1, y1 = [int(v) for v in roi]
                keep = np.zeros_like(mask)
                keep[y0:y1, x0:x1] = True
                mask &= keep
            if mask.sum() < int(self.p["min_mask_area_px"]):
                continue
            out.append(StemInstance(i, mask, float(c), tuple(float(v) for v in b)))
        out.sort(key=lambda s: s.conf, reverse=True)
        return out
