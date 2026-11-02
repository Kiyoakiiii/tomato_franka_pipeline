#!/usr/bin/env python3
"""CPU-only audit of RGB-D samples and YOLO polygon labels; preserves raw files."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re

# Affect only this audit process, before importing numerical libraries.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
STAMP = re.compile(r"\d{8}_\d{6}_\d{3}")


def relative(path):
    return str(path.relative_to(PROJECT))


def guard(path):
    path.resolve().relative_to(PROJECT)
    for parent in (path, *path.parents):
        if parent == PROJECT:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink not accepted: {path}")
    return path


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def polygons(path, width, height):
    valid, issues = [], []
    if path is None:
        return valid, ["missing_label"]
    for number, line in enumerate(guard(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            tokens = [float(x) for x in line.split()]
            if len(tokens) < 7 or (len(tokens) - 1) % 2:
                raise ValueError("not_a_polygon")
            values = np.asarray(tokens[1:]).reshape(-1, 2)
            if tokens[0] != 0 or not np.isfinite(tokens).all():
                raise ValueError("invalid_class_or_nonfinite")
            if (values < 0).any() or (values > 1).any():
                raise ValueError("coordinate_out_of_range")
            area = abs(np.dot(values[:, 0], np.roll(values[:, 1], 1))
                       - np.dot(values[:, 1], np.roll(values[:, 0], 1))) / 2
            if area < 1e-10:
                raise ValueError("degenerate_polygon")
            valid.append(values * [width, height])
        except ValueError as exc:
            issues.append(f"line_{number}:{exc}")
    if not valid and not issues:
        issues.append("empty_label_needs_review")
    return valid, issues


def depth_stats(depth, selection):
    count = int(selection.sum())
    values = np.asarray(depth[selection])
    valid = values[np.isfinite(values) & (values > 0)]
    quantiles = np.percentile(valid, [5, 50, 95]).tolist() if len(valid) else [None] * 3
    return {"pixels": count, "valid_fraction": len(valid) / count if count else None,
            "p05_mm": quantiles[0], "p50_mm": quantiles[1], "p95_mm": quantiles[2]}


def fraction_text(value):
    return "NA" if value is None else f"{100 * value:.1f}%"


def audit_sample(sample, labels, output):
    sid = sample.name
    image_path = guard(sample / "stem_rgb.png")
    depth_path = guard(sample / "stem_depth_mm.npy")
    matched = labels.get(sid, [])
    label_path = matched[0] if len(matched) == 1 else None
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    points, issues = polygons(label_path, width, height)
    if len(matched) > 1:
        issues.append("ambiguous_label_match")
    mask_image = Image.new("L", image.size)
    draw = ImageDraw.Draw(mask_image)
    for poly in points:
        draw.polygon([tuple(p) for p in poly], fill=255)
    mask = np.asarray(mask_image) > 0
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
        & mask[1:-1, :-2] & mask[1:-1, 2:]
        & mask[:-2, :-2] & mask[:-2, 2:] & mask[2:, :-2] & mask[2:, 2:]
    )
    depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"Unexpected depth shape/dtype: {depth.shape}, {depth.dtype}")
    finite = np.isfinite(depth) & (depth > 0)
    stem = depth_stats(depth, mask)
    inside = depth_stats(depth, interior)
    gray = np.asarray(image.resize((17, 16)).convert("L"))
    dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest() if label_path else ""
    disposition = "candidate_rgb_training"
    if issues:
        disposition = "hold_label_review"
    if sid.startswith("20260519"):
        disposition = "exclude_mature_user_confirmed"
    row = {
        "sample_id": sid, "session_date": sid[:8], "rgb_path": relative(image_path),
        "depth_path": relative(depth_path), "label_path": relative(label_path) if label_path else "",
        "width": width, "height": height, "depth_dtype": str(depth.dtype),
        "instances": len(points), "mask_pixels": int(mask.sum()),
        "mask_fraction": float(mask.mean()), "full_depth_valid_fraction": float(finite.mean()),
        "depth_nan_fraction": float(np.isnan(depth).mean()),
        "depth_posinf_fraction": float(np.isposinf(depth).mean()),
        "depth_neginf_fraction": float(np.isneginf(depth).mean()),
        "depth_nonpositive_fraction": float((depth <= 0).mean()),
        **{f"stem_{key}": value for key, value in stem.items()},
        **{f"interior_{key}": value for key, value in inside.items()},
        "image_sha256": image_hash, "label_sha256": label_hash,
        "image_dhash": f"{dhash:064x}", "issues": ";".join(issues),
        "disposition": disposition, "plant_id": "", "split": "unassigned",
        "intrinsics_provenance": "same_date_metadata" if sid.startswith("20260519") else "not_confirmed_for_april",
    }
    thumb = image.resize((480, 270))
    overlay = Image.new("RGBA", thumb.size)
    od = ImageDraw.Draw(overlay)
    for poly in points:
        xy = [(float(x) * 480 / width, float(y) * 270 / height) for x, y in poly]
        od.polygon(xy, fill=(0, 255, 90, 65))
        od.line(xy + [xy[0]], fill=(255, 200, 20, 255), width=2)
    overlay = Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")
    tile = Image.new("RGB", (480, 314), "white")
    tile.paste(overlay, (0, 0))
    td = ImageDraw.Draw(tile)
    td.text((5, 273), f"{sid} | instances={len(points)}", fill="black")
    td.text((5, 287), f"stem valid={fraction_text(stem['valid_fraction'])} | full={fraction_text(float(finite.mean()))}", fill="black")
    td.text((5, 301), disposition, fill="black")
    tile.save(output / "thumbnails" / f"{sid}.jpg", quality=88)
    return row, tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New output directory within this project")
    parser.add_argument("--overlap-only", action="store_true", help="Inspect only labels containing multiple polygons")
    args = parser.parse_args()
    output = guard(PROJECT / args.output)
    if output.exists():
        raise SystemExit("Output already exists; choose a new directory to preserve the earlier audit.")
    label_root = guard(RAW / "image_annotations" / "labels")
    raw_root = guard(RAW / "tomato images")
    labels = defaultdict(list)
    unmatched_labels = []
    for path in sorted(label_root.rglob("*.txt")):
        guard(path)
        match = STAMP.search(path.stem)
        if match:
            labels[match.group()].append(path)
        else:
            unmatched_labels.append(relative(path))
    samples = [p for p in sorted(raw_root.iterdir()) if p.is_dir() and STAMP.fullmatch(p.name)]
    for sample in samples:
        guard(sample)
    output.mkdir(parents=True)
    if args.overlap_only:
        overlaps = []
        for sample in samples:
            paths = labels.get(sample.name, [])
            if len(paths) != 1 or len(paths[0].read_text().strip().splitlines()) < 2:
                continue
            with Image.open(guard(sample / "stem_rgb.png")) as opened:
                image = opened.convert("RGB")
            polys, issues = polygons(paths[0], *image.size)
            masks = []
            draw = ImageDraw.Draw(image)
            for i, poly in enumerate(polys):
                mask_image = Image.new("L", image.size)
                xy = [tuple(p) for p in poly]
                ImageDraw.Draw(mask_image).polygon(xy, fill=255)
                masks.append(np.asarray(mask_image) > 0)
                draw.line(xy + [xy[0]], fill=["red", "cyan", "yellow"][i % 3], width=4)
            for i, a in enumerate(masks):
                for j in range(i + 1, len(masks)):
                    b = masks[j]
                    intersection = int((a & b).sum())
                    overlaps.append({"sample_id": sample.name, "polygon_a": i + 1, "polygon_b": j + 1,
                                     "iou": intersection / int((a | b).sum()),
                                     "intersection_over_smaller": intersection / min(int(a.sum()), int(b.sum())),
                                     "review_status": "inspect_same_stem_or_distinct_plants"})
            image.save(output / f"{sample.name}_overlap.jpg", quality=92)
        write_csv(output / "instance_overlap.csv", overlaps,
                  ["sample_id", "polygon_a", "polygon_b", "iou", "intersection_over_smaller", "review_status"])
        print(json.dumps(overlaps, indent=2))
        return
    (output / "thumbnails").mkdir()
    rows, failures, sheets = [], [], []
    page = Image.new("RGB", (1920, 1884), "#dddddd")
    page_count = 0
    for index, sample in enumerate(samples):
        try:
            row, tile = audit_sample(sample, labels, output)
            rows.append(row)
            page.paste(tile, ((page_count % 4) * 480, (page_count // 4) * 314))
            page_count += 1
        except Exception as exc:
            failures.append({"sample_id": sample.name, "error": str(exc)})
        if page_count == 24 or index == len(samples) - 1:
            if page_count:
                name = f"contact_{len(sheets) + 1:02d}.jpg"
                page.crop((0, 0, 1920, ((page_count + 3) // 4) * 314)).save(output / name, quality=88)
                sheets.append(name)
            page_count = 0
            page = Image.new("RGB", (1920, 1884), "#dddddd")
        if (index + 1) % 25 == 0:
            print(f"Audited {index + 1}/{len(samples)} samples", flush=True)
    if rows:
        write_csv(output / "manifest.csv", rows)
    write_csv(output / "read_failures.csv", failures, ["sample_id", "error"])
    sample_ids = {p.name for p in samples}
    unmatched_labels += [relative(p) for sid, paths in labels.items() if sid not in sample_ids for p in paths]
    pairs = []
    hashes = [(r, int(r["image_dhash"], 16)) for r in rows]
    for i, (a, ha) in enumerate(hashes):
        for b, hb in hashes[i + 1:]:
            distance = (ha ^ hb).bit_count()
            exact = a["image_sha256"] == b["image_sha256"]
            if distance <= 8 or exact:
                pairs.append({"sample_a": a["sample_id"], "sample_b": b["sample_id"],
                              "dhash_distance": distance, "exact_pixels": int(exact)})
    write_csv(output / "similarity_candidates.csv", pairs,
              ["sample_a", "sample_b", "dhash_distance", "exact_pixels"])
    summary = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "raw_sample_count": len(samples), "audited_sample_count": len(rows),
        "annotation_file_count": sum(map(len, labels.values())),
        "sessions": dict(Counter(r["session_date"] for r in rows)),
        "dispositions": dict(Counter(r["disposition"] for r in rows)),
        "instance_counts": dict(Counter(str(r["instances"]) for r in rows)),
        "issues": dict(Counter(r["issues"] for r in rows if r["issues"])),
        "read_failures": failures, "unmatched_labels": unmatched_labels,
        "similarity_candidate_pairs": len(pairs),
        "exact_duplicate_pairs": sum(p["exact_pixels"] for p in pairs),
        "notes": ["Finite positive depth is not verified accurate depth.",
                  "Depth units inferred from file naming; April acquisition provenance needs confirmation.",
                  "Similarity pairs are review candidates, not automatic duplicates or plant identities.",
                  "No train/validation/test split is assigned before plant grouping.",
                  "Geometry checks do not establish polygon semantic correctness or detect every self-intersection."],
        "by_session": {},
    }
    for session in summary["sessions"]:
        rr = [r for r in rows if r["session_date"] == session]
        stats = {"count": len(rr), "labeled": sum(bool(r["label_path"]) for r in rr)}
        for key in ("full_depth_valid_fraction", "stem_valid_fraction", "interior_valid_fraction", "stem_p50_mm"):
            values = [r[key] for r in rr if r[key] is not None]
            stats[key] = dict(zip(["min", "median", "max"], map(float, np.percentile(values, [0, 50, 100])))) if values else None
        stats["stem_valid_under_50pct"] = sum(r["stem_valid_fraction"] is not None and r["stem_valid_fraction"] < 0.5 for r in rr)
        summary["by_session"][session] = stats
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    cards = []
    for row in rows:
        sid = row["sample_id"]
        cards.append(f'<article data-date="{sid[:8]}"><a href="thumbnails/{sid}.jpg"><img loading="lazy" src="thumbnails/{sid}.jpg" alt="{sid}"></a>'
                     f'<p>{html.escape(row["issues"] or "Polygon syntax valid; visual review required")}</p></article>')
    page_html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>番茄数据检查</title>
<style>body{font:16px sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}article{background:white;padding:8px}img{width:100%}button{margin:12px;padding:8px}</style>
<h1>番茄 RGB 与标注检查</h1><p>绿色区域与黄色边界为已有标注；深度有效率仅指有限正值，不代表定位准确。原件未修改。按植株分组前尚未分配训练集与测试集。</p>
<button onclick="filter('')">全部</button><button onclick="filter('20260423')">4 月 23 日</button><button onclick="filter('20260519')">5 月 19 日</button><main>'''
    page_html += "\n".join(cards) + '''</main><script>function filter(d){document.querySelectorAll('article').forEach(e=>e.hidden=!!d&&e.dataset.date!==d)}</script></html>'''
    (output / "review.html").write_text(page_html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
