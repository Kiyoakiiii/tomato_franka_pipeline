#!/usr/bin/env python
"""Replay the overview locate (seedling/overview.py, what scripts/locate.py runs) on recorded pictures, offline:
no robot, no camera. Each experiments/tray/<id>/frame.npz with the camera pose saved next to it (tray_grid.json) is
located again with today's code; one JSON line per picture, a summary, and optionally a comparison with an earlier
replay (--compare) or with what the run itself found that day (--compare online).

    .venvs/pipeline/bin/python tools/replay_overview.py                       # all pictures -> experiments/replay/<id>/
    .venvs/pipeline/bin/python tools/replay_overview.py --only 20261106 --overlays
    .venvs/pipeline/bin/python tools/replay_overview.py --compare experiments/replay/<earlier>/results.jsonl

--pose saved (default) uses the pose of that day (the camera was remounted on 2026-11-06: today's calibration is
15-20 mm off for older pictures); --z-hint none (default) makes every picture take the full height search (the live
run hints the last tray top: --z-hint saved gives each picture the tray top found that day).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frames(only: list[str]) -> list[Path]:
    out = []
    for d in sorted((ROOT / "experiments" / "tray").iterdir()):
        if (d / "frame.npz").exists() and (d / "tray_grid.json").exists() and (not only or any(d.name.startswith(o) for o in only)):
            out.append(d)
    return out


def replay_one(d: Path, pose: str, z_hint: str, overlays: Path | None, config: str | None) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np

    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.camera import Frame
    from seedling.config import load_config
    from seedling.overview import locate_overview

    cfg = load_config(config)
    saved = json.loads((d / "tray_grid.json").read_text())
    frame = Frame.load(d / "frame.npz")
    if pose == "saved":
        T_BC = np.array(saved["T_BC"])
    else:
        T_BC = kinematics.fk_flange(np.array(saved["q"])) @ load_T_FC(cfg)[0]
    hint = float(saved["z_top"]) if z_hint == "saved" and saved.get("ok") and saved.get("z_top") else None
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        res = locate_overview(frame, T_BC, cfg, z_hint=hint, prev_disc=None)
    t = time.perf_counter() - t0
    g, rec, disc = res["grid"], res["rec"], res["disc"]
    if overlays is not None and g.cells:
        import cv2

        from seedling.traygrid import draw_cells

        img = draw_cells(frame, T_BC, g.cells)
        cv2.imwrite(str(overlays / f"{d.name}.jpg"), cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"id": d.name, "t_s": round(t, 2), "ok": bool(g.ok), "status": g.status,
            "z_top": None if g.z_top is None else round(float(g.z_top), 4), "yaw_deg": g.yaw_deg,
            "score": None if g.score is None else round(float(g.score), 4), "cells": g.cells,
            "occupied": rec["occupied"], "planted": rec["planted"],
            "soil_z": {k: (None if v is None else round(float(v), 4)) for k, v in rec["soil_z_overview"].items()},
            "disc": {"ok": bool(disc.get("ok")), "status": disc.get("status"),
                     "centre": None if disc.get("centre") is None else [round(float(c), 4) for c in disc["centre"]],
                     "z_top": disc.get("z_top"), "free_holes": disc.get("free_holes"), "occupied_holes": disc.get("occupied_holes")},
            "log": buf.getvalue().splitlines()}


def kind(r: dict) -> str:
    s = r["status"] or ""
    if not r["ok"]:
        return "failed"
    if s.startswith("outline"):
        return "outline"
    if "moved by" in s:
        return "corrected"
    return "grid"


def cell_shift_mm(a: dict | None, b: dict | None) -> float | None:
    """median distance between the same cells of two fits (mm); None if either has none"""
    import numpy as np

    if not a or not b:
        return None
    ks = [k for k in a if k in b]
    return round(float(np.median([np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in ks])) * 1000, 1)


def online(d: Path) -> dict:
    s = json.loads((d / "tray_grid.json").read_text())
    return {"id": d.name, "ok": bool(s.get("ok")), "status": s.get("status", ""), "cells": s.get("cells"),
            "occupied": s.get("occupied", []), "z_top": s.get("z_top")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[], help="picture ids (or prefixes, e.g. a day 20261106) to replay")
    ap.add_argument("--pose", choices=["saved", "current"], default="saved")
    ap.add_argument("--z-hint", choices=["none", "saved"], default="none")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--overlays", action="store_true", help="also write a small overlay per picture")
    ap.add_argument("--compare", default=None, help="an earlier results.jsonl, or 'online' (what each run found that day)")
    ap.add_argument("--out", default=None, help="output directory (default experiments/replay/<time>)")
    args = ap.parse_args()

    ds = frames(args.only)
    out = Path(args.out) if args.out else ROOT / "experiments" / "replay" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    ov = None
    if args.overlays:
        ov = out / "overlays"
        ov.mkdir(exist_ok=True)
    print(f"{len(ds)} pictures, pose {args.pose}, z hint {args.z_hint}, {args.workers} workers -> {out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(replay_one, ds, [args.pose] * len(ds), [args.z_hint] * len(ds), [ov] * len(ds), [args.config] * len(ds)))
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    from collections import Counter

    import numpy as np

    c = Counter(kind(r) for r in results)
    ts = np.array([r["t_s"] for r in results])
    print(f"tray: {dict(c)}; disc found {sum(r['disc']['ok'] for r in results)}/{len(results)}; "
          f"time per picture median {np.median(ts):.1f} s, max {ts.max():.1f} s (wall {time.time() - t0:.0f} s)")
    if args.compare:
        if args.compare == "online":
            ref = {d.name: online(d) for d in ds}
        else:
            ref = {r["id"]: r for r in map(json.loads, Path(args.compare).read_text().splitlines())}
        changed = 0
        for r in results:
            o = ref.get(r["id"])
            if o is None:
                continue
            sh = cell_shift_mm(r["cells"], o["cells"]) if r["ok"] and o["ok"] else None
            occ = set(r["occupied"]) ^ set(o["occupied"])
            if r["ok"] != o["ok"] or (sh is not None and sh > 2) or occ:
                changed += 1
                print(f"  {r['id']}: {'ok' if o['ok'] else 'FAILED'} -> {'ok' if r['ok'] else 'FAILED'}"
                      + (f", cells moved {sh} mm" if sh is not None and sh > 2 else "")
                      + (f", plugs differ {sorted(occ)}" if occ else "") + f"  [{r['status'][:70]}]")
        print(f"{changed}/{len(results)} pictures differ from {args.compare}")
    print(f"results {out / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
