from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera_source import make_camera_source
from src.config import load_config
from src.paths import PROJECT_DIR, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect pixel -> robot XY correspondences and save a planar homography."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", choices=["zed", "webcam"], help="Override camera source")
    parser.add_argument("--output", default="calibration/planar_pixel_to_robot.json")
    parser.add_argument("--update-config", action="store_true", help="Set localization.mode to homography in config")
    args = parser.parse_args()

    config = load_config(args.config)
    source = make_camera_source(config, source_override=args.source)
    out_path = resolve_project_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points_px: list[list[float]] = []
    points_xy: list[list[float]] = []
    current_frame = None
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    window = "04 calibration: click target point, then enter robot x y"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("Instructions:")
    print("  1. Put a visible marker/tool tip at a known robot XY point on the grasp plane.")
    print("  2. Click that point in the camera image.")
    print("  3. Type robot x y in meters in the terminal.")
    print("  4. Repeat at least 4 points, spread across the workspace.")
    print("Keys: q quit | c clear points | s save if >=4 points")

    try:
        for frame in source.frames():
            current_frame = frame.rgb.copy()
            display = current_frame.copy()
            for i, (u, v) in enumerate(points_px):
                cv2.circle(display, (int(u), int(v)), 8, (255, 40, 40), -1)
                cv2.putText(display, str(i + 1), (int(u) + 8, int(v) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"points: {len(points_px)}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

            while clicked:
                u, v = clicked.pop(0)
                text = input(f"Clicked pixel ({u}, {v}). Enter robot x y in meters, or blank to cancel: ").strip()
                if not text:
                    print("Cancelled this clicked point.")
                    continue
                try:
                    x_str, y_str = text.replace(",", " ").split()[:2]
                    x, y = float(x_str), float(y_str)
                except Exception:
                    print("Could not parse. Example: 0.500 0.120")
                    continue
                points_px.append([float(u), float(v)])
                points_xy.append([x, y])
                print(f"Added point {len(points_px)}: pixel=({u}, {v}) -> robot=({x:.4f}, {y:.4f})")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                points_px.clear()
                points_xy.clear()
                print("Cleared points.")
            elif key == ord("s"):
                if len(points_px) < 4:
                    print("Need at least 4 points to save homography.")
                    continue
                h, status = cv2.findHomography(
                    np.array(points_px, dtype=np.float64),
                    np.array(points_xy, dtype=np.float64),
                    method=0,
                )
                if h is None:
                    print("Homography solve failed.")
                    continue
                payload = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "type": "planar_pixel_to_robot_xy",
                    "note": "Maps image pixel [u,v,1] to robot base XY [x,y] on the fixed grasp plane.",
                    "pixel_points": points_px,
                    "robot_xy_points_m": points_xy,
                    "pixel_to_robot_homography": h.tolist(),
                    "status": None if status is None else status.reshape(-1).astype(int).tolist(),
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Saved calibration: {out_path}")
                if args.update_config:
                    config_path = resolve_project_path(args.config)
                    config_json = json.loads(config_path.read_text(encoding="utf-8"))
                    config_json.setdefault("localization", {})["mode"] = "homography"
                    config_json["localization"]["calibration_file"] = str(Path(args.output).as_posix())
                    config_path.write_text(json.dumps(config_json, indent=2), encoding="utf-8")
                    print(f"Updated config localization.mode='homography': {config_path}")
                else:
                    print("Set config localization.mode to \"homography\" before executing vision-guided grasp.")
            elif key == ord("q") or key == 27:
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
