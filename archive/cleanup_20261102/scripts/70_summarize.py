#!/usr/bin/env python
"""Summarise experiment CSVs with raw counts and 95% Wilson intervals.

    .venvs/pipeline/bin/python scripts/70_summarize.py experiments/runs/<session>/end_to_end_trials.csv
    .venvs/pipeline/bin/python scripts/70_summarize.py experiments/tolerance/*/tolerance_trials.csv
    .venvs/pipeline/bin/python scripts/70_summarize.py experiments/localization/*/localization_checks.csv

Frames/retries of the same plant are not independent; intervals here treat rows as independent
trials and are only valid when each row is a separate plant (see docs/experiment_protocol_CN.md).
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n=0 (no data)"
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {100*p:.0f}%  [95% CI {100*max(0, c-h):.0f}–{100*min(1, c+h):.0f}%]"


def ints(rows, col):
    return [int(r[col]) for r in rows if r.get(col, "") not in ("", None)]


def e2e(rows):
    n = len(rows)
    rej = [r for r in rows if r.get("rejected") == "1"]
    att = [r for r in rows if r.get("acquisition_attempted") == "1"]
    print(f"presented targets: {n}, rejected before acquisition: {len(rej)}, acquisition attempted: {len(att)}")
    acq = ints(att, "acquisition_success")
    print("acquisition success (denominator: attempted, unknown excluded):", wilson(sum(acq), len(acq)), f" unknown={len(att)-len(acq)}")
    ext = ints([r for r in rows if r.get("extraction_attempted") == "1"], "extraction_success")
    print("extraction success (denominator: extraction attempted):", wilson(sum(ext), len(ext)))
    e = ints(rows, "end_to_end_success")
    print("end-to-end success (denominator: all presented):", wilson(sum(e), n))
    print("end-to-end success (denominator: executed):", wilson(sum(ints(att, "end_to_end_success")), len(att)))
    dmg = [r for r in rows if r.get("damage_assessable") == "1"]
    sd = ints(dmg, "visible_stem_damage")
    print("visible stem damage (denominator: assessable):", wilson(sum(sd), len(sd)), f" not assessed={n-len(dmg)}")
    ct = [float(r["cycle_time_s"]) for r in rows if r.get("cycle_time_s")]
    if ct:
        ct.sort()
        print(f"cycle time s: median {ct[len(ct)//2]:.1f}, min {ct[0]:.1f}, max {ct[-1]:.1f}")
    reasons = defaultdict(int)
    for r in rows:
        if r.get("failure_reason") or r.get("rejection_reason"):
            reasons[(r.get("failure_stage") or "REJECTED") + ": " + (r.get("failure_reason") or r.get("rejection_reason"))[:70]] += 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {v:3d} x {k}")


def tolerance(rows):
    by = defaultdict(list)
    for r in rows:
        for ax in "cas":
            v = r.get(f"offset_{ax}_mm", "")
            if v not in ("", "0", "0.0"):
                by[(r["stage"], ax, float(v))].append(r)
                break
        else:
            by[(r["stage"], "-", 0.0)].append(r)
    print("stage, axis, offset_mm -> acquisition | transport/extraction | stem damage")
    for key in sorted(by):
        rs = by[key]
        acq = ints(rs, "acquisition_success")
        sec = ints(rs, "transport_success") or ints(rs, "extraction_success")
        dm = ints(rs, "visible_stem_damage")
        print(f"  {key[0]:<18} {key[1]} {key[2]:+6.1f}: {wilson(sum(acq), len(acq)):<34} | {wilson(sum(sec), len(sec)):<34} | {sum(dm)}/{len(dm)}")


def localization(rows):
    import numpy as np

    for role in sorted({r["reference_role"] for r in rows}):
        rs = [r for r in rows if r["reference_role"] == role and r.get("error_c_mm")]
        rej = [r for r in rows if r["reference_role"] == role and r.get("rejected") == "1"]
        print(f"[{role}] n={len(rs)} rejected={len(rej)}")
        for ax in "cas":
            e = np.array([float(r[f"error_{ax}_mm"]) for r in rs])
            if len(e):
                print(f"  e_{ax}: mean signed {e.mean():+.2f} mm, median |e| {np.median(abs(e)):.2f}, p90 |e| {np.percentile(abs(e), 90):.2f}, max |e| {abs(e).max():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in args.csv:
        with open(path, newline="") as fh:
            groups[Path(path).name] += list(csv.DictReader(fh))
    for name, rows in groups.items():
        print(f"\n===== {name} ({len(rows)} rows from {len(args.csv)} file(s))")
        {"end_to_end_trials.csv": e2e, "tolerance_trials.csv": tolerance, "localization_checks.csv": localization}.get(name, lambda r: print("unknown table"))(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
