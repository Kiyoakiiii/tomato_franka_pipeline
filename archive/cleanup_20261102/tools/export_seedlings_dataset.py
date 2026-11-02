#!/usr/bin/env python3
"""Export a traceable development dataset without modifying any source files."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = PROJECT / value
    path.resolve().relative_to(PROJECT)
    for parent in (path, *path.parents):
        if parent == PROJECT:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlinks are not accepted: {parent}")
    return path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_polygons(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        tokens = line.split()
        if tokens[0] != "0" or len(tokens) < 7 or len(tokens) % 2 != 1:
            raise ValueError("Expected class 0 and >=3 XY vertices")
        points = [float(value) for value in tokens[1:]]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in points):
            raise ValueError("Non-finite or out-of-range coordinates")
        xy = list(zip(points[::2], points[1::2]))
        area2 = sum(x*y2-x2*y for (x, y), (x2, y2) in zip(xy, xy[1:]+xy[:1]))
        if abs(area2) < 1e-12:
            raise ValueError("Degenerate polygon")
    if not lines:
        raise ValueError("Missing annotation is not a negative example")
    return lines


def curate(rows, groups, policy):
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample ID")
    decisions = []
    for source in rows:
        r = dict(source)
        sid = r["sample_id"]
        group = groups.get(sid, "")
        if r["session_date"] in policy["exclude_sessions"]:
            status, reason = "excluded", "mature_session_user_confirmed"
        elif sid in policy["exclude_accidental"]:
            status, reason = "excluded", "accidental_camera_view"
        elif not r["label_path"]:
            status, reason = "hold", "visible_plant_label_missing"
        else:
            if r["issues"]:
                raise ValueError(f"Unresolved audit issue: {sid}: {r['issues']}")
            if not group:
                raise ValueError(f"Missing capture group: {sid}")
            status, reason = "included", "selected_foreground_segment"
        r.update(status=status, reason=reason, capture_group=group, plant_id_confirmed="",
                 split=("val" if group in policy["validation_time_groups"] else "train")
                 if status == "included" else "", derived_image="", derived_label="",
                 image_file_sha256="", derived_label_sha256="", label_action="")
        decisions.append(r)
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/curated/seedlings_dev_v3")
    parser.add_argument("--policy", default="configs/seedlings_dev_v3.json")
    args = parser.parse_args()
    output = local_path(args.output)
    if output.exists():
        raise SystemExit("Output exists; choose a new versioned directory")
    policy_path = local_path(args.policy)
    policy = json.loads(policy_path.read_text())
    rows = read_csv(local_path("data/review/audit_20260930/manifest.csv"))
    group_rows = read_csv(local_path("data/curated/seedlings_v2/plant_groups_review.csv"))
    groups = {r["sample_id"]: r["proposed_time_group"] for r in group_rows}
    if len(groups) != len(group_rows):
        raise ValueError("Duplicate group assignment")
    if not set(policy["validation_time_groups"]) <= set(groups.values()):
        raise ValueError("Unknown validation group")
    decisions = curate(rows, groups, policy)
    # Complete all input checks before creating the output directory.
    labels = {}
    applied_fixes = set()
    for r in decisions:
        # The initial audit hashes decoded RGB pixels, not PNG container bytes.
        with Image.open(local_path(r["rgb_path"])) as image:
            pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        if pixel_hash != r["image_sha256"]:
            raise ValueError(f"Image pixels changed since audit: {r['rgb_path']}")
        r["image_file_sha256"] = sha256(local_path(r["rgb_path"]))
        if r["label_path"] and sha256(local_path(r["label_path"])) != r["label_sha256"]:
            raise ValueError(f"Label changed since audit: {r['label_path']}")
        if r["status"] != "included":
            continue
        sid = r["sample_id"]
        original = local_path(r["label_path"]).read_text()
        lines = checked_polygons(original)
        keep = policy["duplicate_keep_line_1based"].get(sid)
        if keep is not None:
            if len(lines) != 2 or keep not in (1, 2):
                raise ValueError(f"Duplicate correction does not match source: {sid}")
            labels[sid] = lines[keep - 1] + "\n"
            r["label_action"] = f"duplicate_same_stem_keep_original_line_{keep}"
            applied_fixes.add(sid)
        else:
            if len(lines) != 1:
                raise ValueError(f"Unexpected instance count: {sid}")
            labels[sid] = original
            r["label_action"] = "unchanged"
    if applied_fixes != set(policy["duplicate_keep_line_1based"]):
        raise ValueError("Not all planned corrections were applied")
    output.mkdir(parents=True)
    included = [r for r in decisions if r["status"] == "included"]
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for r in included:
        sid, split = r["sample_id"], r["split"]
        destination = output / "images" / split / f"{sid}.png"
        shutil.copyfile(local_path(r["rgb_path"]), destination)
        if sha256(destination) != r["image_file_sha256"]:
            raise ValueError("Image copy checksum mismatch")
        label = output / "labels" / split / f"{sid}.txt"
        label.write_text(labels[sid], encoding="utf-8")
        r.update(derived_image=str(destination.relative_to(PROJECT)),
                 derived_label=str(label.relative_to(PROJECT)), derived_label_sha256=sha256(label))
    write_csv(output / "manifest.csv", decisions)
    write_csv(output / "label_corrections.csv", [r for r in included if r["sample_id"] in applied_fixes])
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (output / "data.yaml").write_text(
        f"# Provisional capture-group development split; no independent test set.\n"
        f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\nnames:\n  0: stem\n")
    summary = {"counts": dict(Counter(r["status"] for r in decisions)),
               "split_counts": dict(Counter(r["split"] for r in included)),
               "split_groups": {s: sorted({r["capture_group"] for r in included if r["split"] == s})
                                for s in ("train", "val")},
               "corrections": len(applied_fixes), "purpose": policy["purpose"],
               "source_policy_sha256": sha256(policy_path), "raw_checksums_verified": True,
               "test_set": None, "plant_identities_confirmed": False}
    if set(summary["split_groups"]["train"]) & set(summary["split_groups"]["val"]):
        raise ValueError("Capture group crosses split")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
