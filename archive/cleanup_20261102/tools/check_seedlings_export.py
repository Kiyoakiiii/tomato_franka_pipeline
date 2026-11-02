#!/usr/bin/env python3
"""Verify the exported dataset contract and original-label preservation without training."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exporter", ROOT / "tools/export_seedlings_dataset.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def main():
    data = ROOT / "data/curated/seedlings_dev_v3"
    rows = exporter.read_csv(data / "manifest.csv")
    policy = json.loads((data / "policy.json").read_text())
    assert len(rows) == len({r["sample_id"] for r in rows}) == 247
    included = [r for r in rows if r["status"] == "included"]
    assert len(included) == 235
    assert sum(r["status"] == "excluded" for r in rows) == 9
    assert sum(r["status"] == "hold" for r in rows) == 3
    group_splits = {}
    image_splits = {}
    for r in rows:
        assert exporter.sha256(exporter.local_path(r["rgb_path"])) == r["image_file_sha256"]
        if r["label_path"]:
            assert exporter.sha256(exporter.local_path(r["label_path"])) == r["label_sha256"]
        if r["status"] != "included":
            assert not r["split"] and not r["derived_image"] and not r["derived_label"]
            continue
        assert not r["plant_id_confirmed"]  # never fabricate plant identities
        split = r["split"]
        assert group_splits.setdefault(r["capture_group"], split) == split
        assert image_splits.setdefault(r["image_sha256"], split) == split
        image = exporter.local_path(r["derived_image"])
        label = exporter.local_path(r["derived_label"])
        assert not image.is_symlink() and not label.is_symlink()
        assert image.stat().st_ino != exporter.local_path(r["rgb_path"]).stat().st_ino
        assert exporter.sha256(image) == exporter.sha256(exporter.local_path(r["rgb_path"])) == r["image_file_sha256"]
        assert exporter.sha256(label) == r["derived_label_sha256"]
        derived = exporter.checked_polygons(label.read_text())
        original = exporter.checked_polygons(exporter.local_path(r["label_path"]).read_text())
        assert len(derived) == 1
        if r["sample_id"] in policy["duplicate_keep_line_1based"]:
            assert len(original) == 2 and derived[0] == original[0]
        else:
            assert label.read_bytes() == exporter.local_path(r["label_path"]).read_bytes()
    assert sum(r["split"] == "train" for r in rows) == 186
    assert sum(r["split"] == "val" for r in rows) == 49
    assert len(list((data / "images").rglob("*.png"))) == 235
    assert len(list((data / "labels").rglob("*.txt"))) == 235
    old_model = json.loads((ROOT / "data/curated/seedlings_v2/existing_checkpoint_metadata.json").read_text())
    assert exporter.sha256(exporter.local_path(old_model["checkpoint"])) == old_model["sha256"]
    assert not list((ROOT / "data/raw").rglob("*.cache"))
    pixel_cache_record = data / "pixel_cache_manifest.json"
    if pixel_cache_record.exists():
        caches = json.loads(pixel_cache_record.read_text())["files"]
        assert len(caches) == len(included)
        by_id = {r["sample_id"]: r for r in included}
        assert len({r["sample_id"] for r in caches}) == len(included)
        for cache in caches:
            source = by_id[cache["sample_id"]]
            assert cache["pixel_equality_verified"]
            assert cache["source_png"] == source["derived_image"]
            assert cache["png_sha256"] == source["image_file_sha256"]
            path = exporter.local_path(cache["cache"])
            assert path == exporter.local_path(source["derived_image"]).with_suffix(".npy")
            assert exporter.sha256(path) == cache["cache_sha256"]
    # Regression checks for dangerous or silently misleading inputs.
    for bad in ("", "0 0 0 1 0 nan 1", "0 0 0 0.5 0.5 1 1", "1 0 0 1 0 0 1", "0 0 0 1.1 0 0 1"):
        try:
            exporter.checked_polygons(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid label accepted: {bad}")
    for bad in ("../../outside_project", "/tmp/outside_project"):
        try:
            exporter.local_path(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("External path accepted")
    print(json.dumps({"result": "PASS", "samples": 247, "exported_pairs": 235,
                      "raw_images_unchanged": 247, "raw_labels_unchanged": 235,
                      "original_model_unchanged": True, "duplicate_fixes_verified": 3,
                      "verified_decoded_pixel_caches": len(caches) if pixel_cache_record.exists() else 0,
                      "group_overlap": 0, "exact_image_overlap": 0,
                      "warning": "No claim of verified plant independence"}, indent=2))


if __name__ == "__main__":
    main()
