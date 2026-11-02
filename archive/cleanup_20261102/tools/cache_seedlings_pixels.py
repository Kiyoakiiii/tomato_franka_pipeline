#!/usr/bin/env python3
"""Create verified lossless BGR caches beside derived PNGs, never beside raw captures."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_runtime import configure, digest, local


def main():
    configure()
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    dataset = local("data/curated/seedlings_dev_v3")
    record = dataset / "pixel_cache_manifest.json"
    if record.exists():
        raise FileExistsError(record)
    rows = list(csv.DictReader((dataset / "manifest.csv").open()))
    output = []
    for row in rows:
        if row["status"] != "included":
            continue
        png = local(row["derived_image"])
        png.relative_to(dataset / "images")
        if digest(png) != row["image_file_sha256"]:
            raise ValueError(f"Changed image: {png}")
        pixels = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if pixels is None or pixels.shape != (1080, 1920, 3) or pixels.dtype != np.uint8:
            raise ValueError(f"Unexpected decoded pixels: {png}")
        target = local(png.with_suffix(".npy"))
        if target.exists():
            if not np.array_equal(np.load(target, allow_pickle=False), pixels):
                raise ValueError(f"Existing pixel cache differs: {target}")
        else:
            partial = local(target.with_suffix(".npy.partial"))
            with partial.open("xb") as stream:
                np.save(stream, pixels, allow_pickle=False)
            if not np.array_equal(np.load(partial, allow_pickle=False), pixels):
                raise ValueError(f"Cache readback differs: {partial}")
            partial.rename(target)
        output.append({"sample_id": row["sample_id"], "source_png": str(png.relative_to(local('.'))),
                       "png_sha256": digest(png), "cache": str(target.relative_to(local('.'))),
                       "cache_sha256": digest(target), "pixel_equality_verified": True})
    with record.open("x") as stream:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(),
                   "format": "Original-resolution uint8 BGR, exactly equal to OpenCV PNG decode",
                   "purpose": "Avoid repeated lossless PNG decoding; no resize, augmentation or relabeling",
                   "library_behavior": "Ultralytics BaseDataset.load_image uses an existing adjacent NPY even when cache=False",
                   "files": output}, stream, indent=2)
    print(json.dumps({"cached_images": len(output), "record": str(record), "all_pixels_equal": True}))


if __name__ == "__main__":
    main()
