from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import cv2

from src.camera_source import make_camera_source
from src.config import load_config, path_from_config
from src.localization import detection_to_robot_target, target_status_lines
from src.paths import PROJECT_DIR, resolve_project_path
from src.vision import detect_stem, draw_detection_overlay, load_yolo_model, select_yolo_device


def save_debug(rgb, overlay_rgb, detection, target, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cv2.imwrite(str(out_dir / f"{stamp}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stamp}_overlay.jpg"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    payload = {
        "detection": None if detection is None else {
            "confidence": detection.confidence,
            "mask_area_px": detection.mask_area_px,
            "grasp_pixel": list(detection.grasp_pixel),
            "stem_axis_deg_image": detection.stem_axis_deg_image,
            "gripper_axis_deg_image": detection.gripper_axis_deg_image,
            "depth_m": detection.depth_m,
            "depth_method": detection.depth_method,
            "camera_xyz_m": detection.camera_xyz_m,
        },
        "robot_target": None if target is None else asdict(target),
    }
    (out_dir / f"{stamp}_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir / (stamp + '_overlay.jpg')}")


class VisionWindow:
    def __init__(self, title: str):
        self.title = title
        self.backend = "opencv"
        self._key = 255
        self._closed = False
        self._root = None
        self._label = None
        self._image = None
        self._image_tk = None

        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"[vision] OpenCV HighGUI window unavailable: {exc}")
            print("[vision] Using Tk/Pillow window fallback.")
            self._init_tk()

    def _init_tk(self) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as exc:
            raise RuntimeError("No usable GUI backend: OpenCV HighGUI failed and Tk/Pillow import failed.") from exc

        self.backend = "tk"
        self._image = Image
        self._image_tk = ImageTk
        self._root = tk.Tk()
        self._root.title(self.title)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Key>", self._on_key)
        self._label = tk.Label(self._root)
        self._label.pack()

    def _on_key(self, event) -> None:
        if event.keysym == "Escape":
            self._key = 27
            return
        if event.char:
            self._key = ord(event.char.lower())

    def _on_close(self) -> None:
        self._closed = True

    def show(self, rgb, delay_ms: int) -> int:
        if self.backend == "opencv":
            cv2.imshow(self.title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return cv2.waitKey(delay_ms) & 0xFF

        photo = self._image_tk.PhotoImage(image=self._image.fromarray(rgb))
        self._label.configure(image=photo)
        self._label.image = photo
        self._root.update_idletasks()
        self._root.update()
        if self._closed:
            return ord("q")
        key = self._key
        self._key = 255
        return key

    def close(self) -> None:
        if self.backend == "opencv":
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            return

        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-only tomato stem check. No robot motion.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", choices=["zed", "webcam"], help="Override camera source")
    parser.add_argument("--image", help="Use a still image instead of a live camera")
    parser.add_argument("--conf", type=float, help="Override YOLO confidence")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, cuda, 0, etc.")
    parser.add_argument("--once", action="store_true", help="Process one frame and save debug output")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = path_from_config(config, "vision.model_path")
    model = load_yolo_model(model_path)
    device = select_yolo_device(args.device)
    source = make_camera_source(config, source_override=args.source, image=args.image)
    out_dir = PROJECT_DIR / "captures" / "vision_check"

    print(f"Project: {PROJECT_DIR}")
    print(f"Model:   {model_path}")
    print(f"Device:  {device}")
    print("Keys: q quit | s save current overlay")

    window = None
    try:
        window = VisionWindow("01 vision check - no robot motion")
        for frame in source.frames():
            vcfg = config["vision"]
            detection = detect_stem(
                model=model,
                rgb=frame.rgb,
                depth_mm=frame.depth_mm,
                intrinsics=frame.intrinsics,
                confidence=float(args.conf if args.conf is not None else vcfg["confidence"]),
                image_size=int(vcfg["image_size"]),
                max_detections=int(vcfg["max_detections"]),
                grasp_height_percentile=float(vcfg["grasp_height_percentile"]),
                min_mask_area_px=int(vcfg["min_mask_area_px"]),
                device=device,
            )

            target = None
            lines = [f"source: {frame.label}"]
            if detection is None:
                lines.append("YOLO: no stem detected")
            else:
                target = detection_to_robot_target(config, detection)
                u, v = detection.grasp_pixel
                depth_text = "NA" if detection.depth_m is None else f"{detection.depth_m:.3f} m"
                lines.extend([
                    f"YOLO conf {detection.confidence:.2f} | pixel ({u}, {v}) | area {detection.mask_area_px}px",
                    f"image stem axis {detection.stem_axis_deg_image:.1f} deg | depth {depth_text} ({detection.depth_method})",
                ])
                lines.extend(target_status_lines(target))

            overlay = draw_detection_overlay(frame.rgb, detection, lines)
            key = window.show(overlay, delay_ms=1 if not args.once else 100)
            if key == ord("s") or args.once:
                save_debug(frame.rgb, overlay, detection, target, out_dir)
                if args.once:
                    break
            if key == ord("q") or key == 27:
                break
    finally:
        source.close()
        if window is not None:
            window.close()


if __name__ == "__main__":
    main()

