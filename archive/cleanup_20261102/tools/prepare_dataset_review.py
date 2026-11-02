#!/usr/bin/env python3
"""Create review manifests; does not assign final plant identities or run training."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import pickletools
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
ACCIDENTAL = {"20260423_151640_764", "20260423_151641_469", "20260423_151642_151"}


def save_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="data/review/audit_20260930")
    parser.add_argument("--output", default="data/curated/seedlings_v2")
    args = parser.parse_args()
    audit = (PROJECT / args.audit).resolve()
    output = PROJECT / args.output
    audit.relative_to(PROJECT)
    output.resolve().relative_to(PROJECT)
    for path in (output, *output.parents):
        if path == PROJECT:
            break
        if path.is_symlink():
            raise ValueError("Refusing output through a symbolic link")
    if output.exists():
        raise SystemExit("Output already exists; use a new destination")
    rows = list(csv.DictReader((audit / "manifest.csv").open(encoding="utf-8")))
    for row in rows:
        sid = row["sample_id"]
        if sid.startswith("20260519"):
            disposition, reason = "exclude", "mature_session_user_confirmed"
        elif sid in ACCIDENTAL:
            disposition, reason = "exclude", "accidental_camera_view_visually_reviewed"
        elif not row["label_path"]:
            disposition, reason = "hold", "plant_visible_but_label_missing"
        elif row["instances"] == "2":
            disposition, reason = "hold", "duplicate_same_stem_instances_visually_reviewed"
        elif row["issues"]:
            disposition, reason = "hold", row["issues"]
        else:
            disposition, reason = "candidate", "target_definition_and_plant_group_review_required"
        row.update({"curation_status": disposition, "curation_reason": reason,
                    "plant_id_confirmed": "", "final_split": "", "training_ready": "0"})
        if row["instances"] == "2":
            row["curation_reason"] += ";two_polygons_check_instance_identity"
    output.mkdir(parents=True)
    fields = list(rows[0])
    for filename, subset in (
        ("all_samples.csv", rows),
        ("rgb_candidates.csv", [r for r in rows if r["curation_status"] == "candidate"]),
        ("excluded.csv", [r for r in rows if r["curation_status"] == "exclude"]),
        ("needs_review.csv", [r for r in rows if r["curation_status"] == "hold"]),
    ):
        save_csv(output / filename, subset, fields)

    groups = defaultdict(list)
    previous, group_id = None, 0
    group_rows = []
    for row in rows:
        if row["session_date"] != "20260423":
            continue
        now = datetime.strptime(row["sample_id"], "%Y%m%d_%H%M%S_%f")
        gap = (now - previous).total_seconds() if previous else None
        if previous is None or gap > 20:
            group_id += 1
        proposed = f"time_{group_id:02d}"
        groups[proposed].append(row)
        group_rows.append({"sample_id": row["sample_id"], "proposed_time_group": proposed,
                           "gap_before_s": gap, "plant_id_confirmed": "", "reviewed": "0",
                           "final_split": "", "notes": "Temporal suggestion only; merge/split using plant appearance"})
        previous = now
    save_csv(output / "plant_groups_review.csv", group_rows, list(group_rows[0]))
    ranges = [{"proposed_time_group": key, "first": rr[0]["sample_id"], "last": rr[-1]["sample_id"],
               "frames": len(rr), "labeled_candidates": sum(r["curation_status"] == "candidate" for r in rr),
               "plant_identity_confirmed": "0"} for key, rr in groups.items()]
    save_csv(output / "temporal_groups.csv", ranges, list(ranges[0]))

    # Inspection only: do not unpickle the checkpoint or import torch.
    checkpoint = PROJECT / "models" / "stem_yolo11n_v3_clean_one_stem.pt"
    with zipfile.ZipFile(checkpoint) as archive:
        member = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        if archive.getinfo(member).file_size > 10 * 1024 * 1024:
            raise ValueError("Checkpoint pickle metadata exceeds inspection limit")
        ops = [(op.name, arg) for op, arg, _ in pickletools.genops(archive.read(member))]
    metadata = {"checkpoint": str(checkpoint.relative_to(PROJECT)),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "inspection": "ZIP and pickle opcode strings only; model not loaded"}
    for key in ("yaml_file", "date", "version"):
        for i, (_, value) in enumerate(ops):
            if value == key:
                following = [v for _, v in ops[i + 1:i + 5] if isinstance(v, str)]
                if following:
                    metadata[key] = following[0]
    (output / "existing_checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plan = {
        "status": "proposal_only_no_training_started",
        "primary_model": "yolo26s-seg.pt",
        "baseline_model": "yolo11n-seg.pt",
        "optional_comparison": "RFDETRSegSmall",
        "data": None,
        "device": None,
        "initial_trial_settings": {"imgsz": 960, "batch": 2, "workers": 0, "cache": False,
                                   "epochs": 100, "patience": 20, "seed": 42, "mask_ratio": 2},
        "settings_note": "Starting proposal; resolve compatible versions and GPU budget before use. No accuracy guarantee.",
        "required_before_training": [
            "Confirm plant groups; no image-level random split",
            "Define consistently annotated target region or finish labels within a fixed inference ROI",
            "Resolve instance identity of the three two-polygon annotations",
            "Create train/validation/test manifests from reviewed plant groups",
            "Verify usable GPU and coordinate its memory/time allocation; observed NVML driver/library mismatch",
            "Install and lock a project-local vision environment without changing system drivers or robot environment",
        ],
        "evaluation": ["Held-out plants", "Mask IoU and boundary/centerline errors",
                       "Grasp candidate validity", "Per-plant failures and model latency"],
    }
    (output / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Relative URL from the curated directory to the immutable audit thumbnails.
    import os
    prefix = Path(os.path.relpath(audit / "thumbnails", output)).as_posix()
    cards = []
    for key, rr in groups.items():
        content = [f'<section><h2>{key}: {len(rr)} frames (unconfirmed)</h2><div class="grid">']
        for row in rr:
            sid = row["sample_id"]
            content.append(f'<article><img loading="lazy" src="{html.escape(prefix)}/{sid}.jpg" alt="{sid}">'
                           f'<p>{html.escape(row["curation_status"] + ": " + row["curation_reason"])}</p></article>')
        cards.append("".join(content) + "</div></section>")
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>植株分组复核</title>
<style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:10px}article{background:white;padding:8px}img{width:100%}p{overflow-wrap:anywhere}</style>
<h1>植株分组复核</h1><p>以下用超过 20 秒的拍摄间隔提出分组，组号不是植株身份。结合植株叶片、茎形和盆签合并或拆分；同一株若在后面再次出现，仍需使用同一 plant_id。尚未生成训练或测试划分。</p>
<p>232 张单实例标注图为候选；6 张后期苗图和 3 张相机误拍已从候选排除；另 3 张缺标注和 3 张重复实例标注暂缓。原件全部保留。修改 plant_groups_review.csv 填入经确认的植株编号。</p>'''
    (output / "plant_group_review.html").write_text(document + "\n".join(cards) + "</html>", encoding="utf-8")
    print(json.dumps({"output": str(output), "candidates": sum(r["curation_status"] == "candidate" for r in rows),
                      "excluded": sum(r["curation_status"] == "exclude" for r in rows),
                      "hold": sum(r["curation_status"] == "hold" for r in rows),
                      "unconfirmed_time_groups": len(groups), "checkpoint_metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
