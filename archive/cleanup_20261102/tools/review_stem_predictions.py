#!/usr/bin/env python3
"""Review the highest-confidence target mask on the provisional development validation set."""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_runtime import configure, digest, load_libraries, local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--threshold-diagnostics", action="store_true",
                        help="Also evaluate the predeclared 0.05/0.1/0.25/0.4/0.5 development thresholds")
    parser.add_argument("--settings-only", action="store_true", help="Read selected checkpoint training settings without image inference")
    args = parser.parse_args()
    if not (320 <= args.imgsz <= 1280 and args.imgsz % 32 == 0 and 0 < args.conf < 1):
        parser.error("Invalid image size or confidence")
    configure()
    weights, output = local(args.weights), local(args.output)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output.mkdir(parents=True, exist_ok=False)
    YOLO = load_libraries()
    import numpy as np
    from PIL import Image, ImageDraw
    model = YOLO(str(weights))
    stored_args = (model.ckpt or {}).get("train_args", {})
    keys = ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "nbs", "warmup_epochs",
            "warmup_bias_lr", "weight_decay", "mask_ratio", "mosaic", "seed", "patience", "freeze")
    training_settings = {"weights_sha256": digest(weights),
                         "saved_settings": {key: stored_args.get(key) for key in keys},
                         "note": "Requested settings saved in checkpoint; epochs is not proof of completed epoch count. Dataset paths intentionally omitted."}
    (output / "checkpoint_training_settings.json").write_text(json.dumps(training_settings, indent=2)+"\n")
    if args.settings_only:
        print(json.dumps(training_settings, indent=2))
        return
    dataset = local("data/curated/seedlings_dev_v3")
    with (dataset / "manifest.csv").open() as stream:
        samples = [r for r in csv.DictReader(stream) if r["split"] == "val"]
    rows = []
    diagnostic_rows = []
    panels = {}
    for sample in samples:
        sid = sample["sample_id"]
        image_path, label_path = local(sample["derived_image"]), local(sample["derived_label"])
        if digest(image_path) != sample["image_file_sha256"] or digest(label_path) != sample["derived_label_sha256"]:
            raise ValueError(f"Dataset changed: {sid}")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
        w, h = rgb.size
        gt_image = Image.new("1", (w, h))
        draw = ImageDraw.Draw(gt_image)
        for line in label_path.read_text().splitlines():
            values = list(map(float, line.split()[1:]))
            draw.polygon([(x*w, y*h) for x, y in zip(values[::2], values[1::2])], fill=1)
        gt = np.asarray(gt_image, dtype=bool)
        started = time.monotonic()
        inference_conf = min(0.05, args.conf) if args.threshold_diagnostics else args.conf
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=inference_conf, max_det=10,
                               retina_masks=True, device="cpu", save=False, verbose=False)[0]
        elapsed = time.monotonic() - started
        pred = np.zeros_like(gt)
        confidences = result.boxes.conf.cpu().numpy()
        count = int((confidences >= args.conf).sum())
        confidence = 0.0
        low_pred = np.zeros_like(gt)
        low_confidence = 0.0
        if len(result.boxes) and result.masks is not None:
            index = int(result.boxes.conf.argmax())
            low_confidence = float(result.boxes.conf[index])
            low_pred = result.masks.data[index].cpu().numpy() > 0.5
            if low_pred.shape != gt.shape:
                raise ValueError(f"Mask does not align with original image: {low_pred.shape} != {gt.shape}")
            if count:
                confidence, pred = low_confidence, low_pred
        if args.threshold_diagnostics:
            low_intersection = int((gt & low_pred).sum())
            low_union = int((gt | low_pred).sum())
            for threshold in (0.05, 0.1, 0.25, 0.4, 0.5):
                diagnostic_rows.append({"sample_id": sid, "capture_group": sample["capture_group"],
                                        "threshold": threshold,
                                        "predicted_instances": int((confidences >= threshold).sum()),
                                        "selected_mask_iou": low_intersection/low_union if low_confidence >= threshold and low_union else 0.0})
        intersection = int((gt & pred).sum())
        union = int((gt | pred).sum())
        row = dict(sample_id=sid, capture_group=sample["capture_group"],
                   predicted_instances=count, selected_confidence=confidence,
                   selected_mask_iou=intersection/union if union else 0.0,
                   selected_pixel_precision=intersection/int(pred.sum()) if pred.any() else 0.0,
                   selected_pixel_recall=intersection/int(gt.sum()), elapsed_s=elapsed)
        rows.append(row)
        # Two panels retain the original framing. No label-derived crop is used.
        canvas = Image.new("RGB", (1280, 396), "white")
        for column, mask, color, title in ((0, gt, (0, 210, 70), "ANNOTATION"),
                                          (1, pred, (240, 50, 60), f"PREDICTION conf={confidence:.2f}")):
            overlay = np.asarray(rgb).copy()
            overlay[mask] = (0.5*overlay[mask]+0.5*np.asarray(color)).astype(np.uint8)
            panel = Image.fromarray(overlay).resize((640, 360))
            canvas.paste(panel, (640*column, 36))
            ImageDraw.Draw(canvas).text((10+640*column, 6), title, fill="black")
        ImageDraw.Draw(canvas).text((220, 23), f"{sid}  {sample['capture_group']}  IoU={row['selected_mask_iou']:.3f}", fill="black")
        canvas.save(output / f"{sid}.jpg", quality=90)
        panels[sid] = canvas
    with (output / "per_image.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["capture_group"]].append(r)
    group_metrics = [{"capture_group": group, "images": len(rr),
                      "mean_selected_mask_iou": float(np.mean([r["selected_mask_iou"] for r in rr])),
                      "no_detection": sum(r["predicted_instances"] == 0 for r in rr)}
                     for group, rr in groups.items()]
    summary = {"weights": str(weights), "weights_sha256": digest(weights),
               "images": len(rows), "input_size": args.imgsz, "confidence_threshold": args.conf,
               "compute_device": "cpu", "retina_masks": True, "max_det": 10,
               "selection": "highest confidence; threshold fixed before examining pilot predictions",
               "mean_selected_mask_iou": float(np.mean([r["selected_mask_iou"] for r in rows])),
               "median_selected_mask_iou": float(np.median([r["selected_mask_iou"] for r in rows])),
               "no_detection_images": sum(r["predicted_instances"] == 0 for r in rows),
               "iou_at_least_0_5_images": sum(r["selected_mask_iou"] >= 0.5 for r in rows),
               "multiple_detection_images": sum(r["predicted_instances"] > 1 for r in rows),
               "capture_groups": group_metrics, "not_independent_test": True,
               "limitations": "Capture groups are not verified plant IDs; only selected foreground segments are labeled. IoU is not grasp success or mAP."}
    if args.threshold_diagnostics:
        with (output / "threshold_per_image.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        summary["threshold_diagnostics"] = []
        for threshold in (0.05, 0.1, 0.25, 0.4, 0.5):
            rr = [r for r in diagnostic_rows if r["threshold"] == threshold]
            summary["threshold_diagnostics"].append({"threshold": threshold,
                "mean_selected_mask_iou": float(np.mean([r["selected_mask_iou"] for r in rr])),
                "no_detection_images": sum(r["predicted_instances"] == 0 for r in rr),
                "iou_at_least_0_5_images": sum(r["selected_mask_iou"] >= 0.5 for r in rr)})
        summary["threshold_diagnostics_note"] = "Development diagnostics only; primary threshold remains 0.25. Highest-confidence mask selected without consulting the annotation."
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    ordered = sorted(rows, key=lambda r: r["selected_mask_iou"])
    cards = "\n".join(f'<figure><img loading="lazy" src="{r["sample_id"]}.jpg"><figcaption>{html.escape(r["sample_id"])} / IoU {r["selected_mask_iou"]:.3f}</figcaption></figure>' for r in ordered)
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>目标茎段开发预测检查</title>'
                '<style>body{font:16px sans-serif;max-width:1300px;margin:24px auto}img{width:100%}figure{margin:24px 0}</style>'
                '<h1>目标茎段开发预测检查</h1><p>按 IoU 从低到高排列。左：标注；右：最高置信度预测。'
                '保留原图视野，未用真实标签裁剪。当前是未确认植株身份的开发划分，不是独立测试；IoU 不代表抓取成功率。</p>' + cards + '</html>')
    (output / "review.html").write_text(document, encoding="utf-8")
    # Six lowest-IoU examples, including misses, chosen mechanically for review.
    sheet = Image.new("RGB", (1280, 396*6), "white")
    for i, row in enumerate(ordered[:6]):
        sheet.paste(panels[row["sample_id"]], (0, 396*i))
    sheet.save(output / "lowest_iou_six.jpg", quality=90)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
