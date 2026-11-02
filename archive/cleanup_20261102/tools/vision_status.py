#!/usr/bin/env python3
"""Read this project's vision run records and matching processes; never signal a process."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def task_processes():
    matches = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
            if not any(x in ("tools/train_stem_cpu.py", "tools/train_stem_gpu.py") for x in args):
                continue
            if (path / "cwd").resolve() != ROOT:
                continue
            if "--name" in args:
                matches[args[args.index("--name") + 1]] = int(path.name)
        except (OSError, IndexError):
            continue
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.run and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Invalid run name")
    runs = ROOT / "experiments/vision"
    paths = [runs / args.run] if args.run else sorted(p for p in runs.iterdir() if p.is_dir())
    processes = task_processes()
    output = []
    for path in paths:
        if not (path / "run_metadata.json").is_file():
            continue
        path.resolve().relative_to(ROOT)
        metrics = csv_rows(path / "train/results.csv")
        resource = csv_rows(path / "resources.csv")
        metadata = json.loads((path / "run_metadata.json").read_text())
        summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() else None
        output.append({"run": path.name, "status": "completed" if summary else
                       "running" if path.name in processes else
                       "interrupted_checkpoint_saved" if (path / "interruption.json").is_file() else "incomplete_no_matching_process",
                       "pid": processes.get(path.name), "device": metadata["compute_device"],
                       "last_epoch": metrics[-1] if metrics else None,
                       "last_resources": resource[-1] if resource else None,
                       "summary": summary})
    if args.compact:
        output = [{"run": r["run"], "status": r["status"], "pid": r["pid"],
                   "epoch": (r["last_epoch"] or {}).get("epoch"),
                   "elapsed_s": (r["last_resources"] or {}).get("elapsed_s"),
                   "last_mask_map50": (r["last_epoch"] or {}).get("metrics/mAP50(M)"),
                   "last_mask_map50_95": (r["last_epoch"] or {}).get("metrics/mAP50-95(M)"),
                   "gpu_peak_reserved_mib": (r["last_resources"] or {}).get("gpu_peak_reserved_mib"),
                   "gpu_free_mib": (r["last_resources"] or {}).get("gpu_free_mib")}
                  for r in output]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
