from __future__ import annotations

import argparse
from dataclasses import replace

import cv2

from src.camera_source import make_camera_source
from src.config import load_config, path_from_config
from src.localization import check_workspace, detection_to_robot_target, target_status_lines
from src.paths import PROJECT_DIR
from src.robot_control import FrankaRobot, run_grasp_sequence
from src.vision import detect_stem, draw_detection_overlay, load_yolo_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-guided tomato/fake-stem grasp with manual confirmation.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", choices=["zed", "webcam"], help="Override camera source")
    parser.add_argument("--execute", action="store_true", help="Actually move Franka after pressing g")
    parser.add_argument(
        "--allow-rough-manual-scale",
        action="store_true",
        help="Allow robot execution when localization.mode is manual_scale. Use only after checking target pose.",
    )
    parser.add_argument("--host", help="Override robot FCI IP")
    parser.add_argument("--conf", type=float, help="Override YOLO confidence")
    parser.add_argument("--manual-step-mm", type=float, default=5.0, help="Keyboard correction step")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config["robot"]["host"] = args.host

    if args.execute and config["localization"]["mode"] == "manual_scale" and not args.allow_rough_manual_scale:
        raise SystemExit(
            "Refusing to execute with uncalibrated manual_scale localization.\n"
            "First run 04_collect_planar_calibration.py and set localization.mode='homography',\n"
            "or rerun with --allow-rough-manual-scale after you have manually checked the target."
        )

    model_path = path_from_config(config, "vision.model_path")
    model = load_yolo_model(model_path)
    source = make_camera_source(config, source_override=args.source)
    robot = None

    offsets = {"x": 0.0, "y": 0.0, "z": 0.0}
    step = args.manual_step_mm / 1000.0

    print(f"Project: {PROJECT_DIR}")
    print(f"Model:   {model_path}")
    print("Keys:")
    print("  q quit")
    print("  g dry-run/execute current grasp target")
    print("  i/k adjust robot x + / -")
    print("  j/l adjust robot y + / -")
    print("  u/o adjust robot z + / -")
    print("  r reset manual offsets")
    print("  p save overlay")
    print("\nDefault is safe: no motion unless --execute is used.")

    try:
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
            )

            target = None
            adjusted_target = None
            lines = [f"{'EXECUTE' if args.execute else 'DRY RUN'} | source: {frame.label}"]
            if detection is None:
                lines.append("YOLO: no stem detected")
            else:
                target = detection_to_robot_target(config, detection)
                adjusted_target = replace(
                    target,
                    xy_m=(target.xy_m[0] + offsets["x"], target.xy_m[1] + offsets["y"]),
                    z_m=target.z_m + offsets["z"],
                )
                workspace_warning = check_workspace(adjusted_target.xyz_m, config["robot"]["workspace_limits_m"])
                if workspace_warning:
                    adjusted_target = replace(adjusted_target, warning=workspace_warning)
                u, v = detection.grasp_pixel
                depth_text = "NA" if detection.depth_m is None else f"{detection.depth_m:.3f} m"
                lines.extend([
                    f"YOLO conf {detection.confidence:.2f} | pixel ({u}, {v}) | image axis {detection.stem_axis_deg_image:.1f} deg",
                    f"depth {depth_text} ({detection.depth_method}) | manual offset xyz "
                    f"{offsets['x']*1000:.0f},{offsets['y']*1000:.0f},{offsets['z']*1000:.0f} mm",
                ])
                lines.extend(target_status_lines(adjusted_target))

            overlay = draw_detection_overlay(frame.rgb, detection, lines)
            cv2.imshow("05 vision-guided grasp", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord("i"):
                offsets["x"] += step
            elif key == ord("k"):
                offsets["x"] -= step
            elif key == ord("j"):
                offsets["y"] += step
            elif key == ord("l"):
                offsets["y"] -= step
            elif key == ord("u"):
                offsets["z"] += step
            elif key == ord("o"):
                offsets["z"] -= step
            elif key == ord("r"):
                offsets = {"x": 0.0, "y": 0.0, "z": 0.0}
            elif key == ord("p"):
                out = PROJECT_DIR / "captures" / "vision_guided_overlay.jpg"
                cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved overlay: {out}")
            elif key == ord("g"):
                if adjusted_target is None:
                    print("No valid target to grasp.")
                    continue
                if args.execute and robot is None:
                    print("Connecting to Franka...")
                    robot = FrankaRobot(config)
                    print(robot.current_state_text())
                run_grasp_sequence(robot, adjusted_target, config, dry_run=not args.execute)
    finally:
        if robot is not None:
            robot.shutdown()
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
