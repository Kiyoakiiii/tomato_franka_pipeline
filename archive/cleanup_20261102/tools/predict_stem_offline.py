#!/usr/bin/env python3
"""Predict an existing project-local RGB file on CPU; never access camera or robot devices."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_runtime import configure, digest, load_libraries, local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="New project-local output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("Confidence must be between 0 and 1")
    if not 320 <= args.imgsz <= 1280 or args.imgsz % 32:
        parser.error("Image size must be a multiple of 32 in 320..1280")
    configure()
    weights, source, output = local(args.weights), local(args.image), local(args.output)
    if not weights.is_file() or not source.is_file() or source.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        parser.error("Existing local checkpoint and RGB image required")
    output.mkdir(parents=True, exist_ok=False)
    YOLO = load_libraries()
    import numpy as np
    from PIL import Image
    model = YOLO(str(weights))
    result = model.predict(str(source), imgsz=args.imgsz, conf=args.conf, device="cpu", retina_masks=True,
                           max_det=10, verbose=False, save=False)[0]
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    selected = np.zeros(rgb.shape[:2], dtype=bool)
    candidates = [{"confidence": float(box.conf), "box_xyxy": box.xyxy[0].tolist()} for box in result.boxes]
    selected_index = None
    if candidates and result.masks is not None:
        selected_index = int(result.boxes.conf.argmax())
        selected = result.masks.data[selected_index].cpu().numpy() > 0.5
        if selected.shape != rgb.shape[:2]:
            raise ValueError("Predicted mask is not aligned to original pixels")
    Image.fromarray(selected.astype(np.uint8)*255).save(output / "selected_mask.png")
    overlay = rgb.copy()
    overlay[selected] = (0.5*overlay[selected]+0.5*np.array([245, 65, 65])).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.jpg", quality=95)
    record = {"image": str(source), "image_sha256": digest(source), "weights": str(weights),
              "weights_sha256": digest(weights), "imgsz": args.imgsz, "confidence_threshold": args.conf,
              "device": "cpu", "retina_masks": True, "candidates": candidates,
              "selected_index": selected_index, "selected_pixels": int(selected.sum()),
              "status": "candidate_requires_validation" if selected.any() else "no_target_mask",
              "selection": "highest confidence; foreground single-plant work area assumed",
              "not_a_robot_target": True,
              "note": "No 3D depth, calibration, clearance or grasp feasibility validation is performed."}
    (output / "prediction.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"output": str(output), "predictions": len(candidates), "status": record["status"]}))


if __name__ == "__main__":
    main()
