#!/usr/bin/env python
"""Offline check of the whole transplanting over many random layouts (no robot, no camera).

The goal: every plant within the arm's reach is transplanted, wherever the tray, the disc and the plants are. Each
layout moves and turns the located tray (calibration/tray_grid.json) at random, puts the disc in front of it or to
its left or right (at random), and plants a random set of cells. 72_transplant_all.py --dry-run then plans the full
sequence for it as the real run would: for every plant the observation pose, the grasp from there, the placement
(nearest plant first; a full disc is replaced by an empty one), and the way back to the overview. Each layout runs
in a private copy of the scripts (experiments/sim/<run>/<layout>/proj), so nothing in calibration/ or
experiments/grasp/ is touched.

Every plant left over is then classified on its own, with all other plants removed and an empty disc:
  - out of reach: no safe grasp (or placement) exists even then: the arm's limit for a flat grasp;
  - PLANNER FAILURE: it can be done alone, but not in the sequence: a bug to fix (should be none).

    .venvs/pipeline/bin/python scripts/74_simulate_layouts.py --layouts 24 --workers 4
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR

LINKED = ("configs", "models", "data", "tools", ".venvs")


def make_layouts(n, seed, shift, yaw_deg, plants, tray_top=(0.05, 0.20), disc_top=(0.03, 0.15)):
    g0 = json.loads((PROJECT_DIR / "calibration" / "tray_grid.json").read_text())
    d0 = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    rng = np.random.default_rng(seed)
    names = sorted(g0["cells"])
    C = np.array([g0["cells"][k][:2] for k in names])
    c0, top = C.mean(0), float(g0["z_top"])
    out = []
    for _ in range(n):
        th = np.radians(rng.uniform(-yaw_deg, yaw_deg))
        top = float(rng.uniform(*tray_top))  # the tray raised (or not) on something
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t = c0 + rng.uniform(-shift, shift, 2)
        Cn = (C - c0) @ R.T + t
        g = dict(g0)
        g["cells"] = {k: [float(x), float(y), top] for k, (x, y) in zip(names, Cn)}
        g["A1"] = g["cells"]["A1"]
        occ = sorted(rng.choice(names, size=int(rng.integers(plants[0], plants[1] + 1)), replace=False).tolist())
        g.update(z_top=top, occupied=occ, planted=occ, green=occ, soil_z_overview={k: float(top + rng.uniform(0.008, 0.020)) for k in occ})
        half = np.abs((Cn - t) @ R).max(0) + 0.0448 / 2 + 0.015
        while True:  # the disc: in front of the tray or beside it, its rim at least 6 cm from the tray, within 0.8 m
            side = str(rng.choice(["front", "right", "left"]))
            xy = (np.array([rng.uniform(0.38, 0.52), rng.uniform(-0.15, 0.15)]) if side == "front" else
                  np.array([rng.uniform(0.50, 0.75), rng.uniform(0.36, 0.50) * (1 if side == "left" else -1)]))
            if np.max(np.abs(R.T @ (xy - t)) - half) > 0.085 + 0.06 and np.hypot(*xy) < 0.8:
                break
        ph = rng.uniform(0, np.pi / 3)
        holes = [[float(xy[0] + 0.0575 * np.cos(ph + k * np.pi / 3)), float(xy[1] + 0.0575 * np.sin(ph + k * np.pi / 3))] for k in range(6)]
        d = dict(d0)
        dt = float(rng.uniform(*disc_top))
        d.update(centre=[float(xy[0]), float(xy[1])], holes=holes, holes_seen=holes, free_holes=[1, 2, 3, 4, 5, 6], ok=True,
                 z_top=dt, z_top_depth=dt, hole_bottom_z=dt - (float(d0["z_top"]) - float(d0.get("hole_bottom_z", float(d0["z_top"]) - 0.055))))
        d.pop("ok_last_picture", None)
        desc = (f"tray centre {np.round(t, 3).tolist()} turned {np.degrees(th):+.1f} deg, top at z {top:.3f}; disc {side} at "
                f"{np.round(xy, 3).tolist()}, top at z {dt:.3f}; {len(occ)} plants")
        out.append((g, d, desc))
    return out


def private_copy(L: Path, g: dict, d: dict) -> Path:
    W = L / "proj"
    if W.exists():
        shutil.rmtree(W)
    (W / "calibration").mkdir(parents=True)
    (W / "experiments" / "grasp").mkdir(parents=True)
    for d_ in ("scripts", "seedling"):
        shutil.copytree(PROJECT_DIR / d_, W / d_, ignore=shutil.ignore_patterns("__pycache__"))
    for d_ in LINKED:
        if (PROJECT_DIR / d_).exists():
            (W / d_).symlink_to(PROJECT_DIR / d_)
    for f in (PROJECT_DIR / "calibration").iterdir():
        if f.name not in ("tray_grid.json", "disc.json"):
            (W / "calibration" / f.name).symlink_to(f)
    (W / "calibration" / "tray_grid.json").write_text(json.dumps(g))
    (W / "calibration" / "disc.json").write_text(json.dumps(d))
    return W


def run_layout(L: Path, g, d, q_json: Path, discs: int):
    W = private_copy(L, g, d)
    env = dict(os.environ, SEEDLING_SIM_Q=str(q_json))  # seedling.robot: planning only, never connects
    py = sys.executable

    def run(cmd):
        r = subprocess.run([py, "-u", *cmd], cwd=W, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    t0 = time.time()
    _, out = run(["scripts/72_transplant_all.py", "--dry-run", "--max-plants", "999", "--dry-run-discs", str(discs)])
    (L / "sequence.txt").write_text(out)
    if "libfranka" in out or "franky" in out:
        raise SystemExit(f"{L.name}: a script reached franky in the simulation: stopped (see sequence.txt)")
    if "Traceback" in out:
        print(f"{L.name}: the sequence crashed: {out.strip().splitlines()[-1][:200]}", flush=True)
    back = re.findall(r"^  (\w+): \(dry run\) NO SAFE WAY BACK.*$", out, re.M)
    done = [k for k, _ in re.findall(r"^  (\w+): grasp pitched .* into H(\d)", out, re.M) if k not in back]
    left = [k for k in g["occupied"] if k not in done and k not in back]
    # each plant left over on its own: all other plants gone, an empty disc
    verdict = {}
    for k in left:
        others = ",".join(x for x in g["occupied"] if x != k)
        rg, og = run(["scripts/65_side_grasp.py", "plan", "--from-observe", "--cell", k, "--lift-mm", "100", "--ignore-cells", others])
        if "Traceback" in og:
            verdict[k] = ("PLANNER FAILURE", "crashed: " + og.strip().splitlines()[-1][:160])
            continue
        if rg != 0:
            why = [ln.strip() for ln in og.splitlines() if "NOT SAFE at" in ln or "observation pose" in ln or "not reachable" in ln]
            verdict[k] = ("out of reach (grasp)", why[-1][:160] if why else og.strip()[-160:])
            continue
        rp, op = run(["scripts/71_place_in_disc.py", "plan", "--cell", k, "--free", "1,2,3,4,5,6"])
        if "Traceback" in op:
            verdict[k] = ("PLANNER FAILURE", "crashed: " + op.strip().splitlines()[-1][:160])
            continue
        if rp != 0:
            verdict[k] = ("out of reach (placement)", op.strip().splitlines()[-1][:160] if op.strip() else "")
            continue
        verdict[k] = ("PLANNER FAILURE", "possible alone, not in the sequence")
    shutil.rmtree(W, ignore_errors=True)
    return {"done": len(done), "plants": len(g["occupied"]), "left": verdict, "no_way_back": back, "s": round(time.time() - t0)}


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--layouts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4, help="layouts planned at the same time (one CPU core each)")
    ap.add_argument("--tray-shift-m", type=float, default=0.06, help="the tray moved up to this far in x and y from where it was located")
    ap.add_argument("--tray-yaw-deg", type=float, default=10.0, help="... and turned up to this much")
    ap.add_argument("--plants", type=int, nargs=2, default=[8, 20], help="plants per layout (min max)")
    ap.add_argument("--tray-top-m", type=float, nargs=2, default=[0.05, 0.20], help="tray top height range (base frame, m)")
    ap.add_argument("--disc-top-m", type=float, nargs=2, default=[0.03, 0.15], help="disc top height range (base frame, m)")
    ap.add_argument("--discs", type=int, default=99, help="a full disc is replaced by an empty one, up to this many")
    args = ap.parse_args()
    cfg = config_from_args(args)

    run_dir = PROJECT_DIR / "experiments" / "sim" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    q_json = run_dir / "overview_q.json"
    q_json.write_text(json.dumps({"q": cfg["tray"]["overview_q"]}))  # the runs start at the overview pose
    layouts = make_layouts(args.layouts, args.seed, args.tray_shift_m, args.tray_yaw_deg, args.plants, args.tray_top_m, args.disc_top_m)
    print(f"{len(layouts)} layouts, {args.workers} at a time -> {run_dir.relative_to(PROJECT_DIR)}", flush=True)

    def job(i):
        g, d, desc = layouts[i]
        L = run_dir / f"{i:03d}"
        L.mkdir()
        (L / "layout.txt").write_text(desc + "\n" + " ".join(g["occupied"]) + "\n")
        r = run_layout(L, g, d, q_json, args.discs)
        r["desc"] = desc
        (L / "result.json").write_text(json.dumps(r, indent=1))
        bad = {k: v for k, v in r["left"].items() if v[0] == "PLANNER FAILURE"}
        print(f"{i:03d} {r['done']}/{r['plants']} transplanted ({r['s']} s); {desc}"
              + "".join(f"\n      {k}: {v[0]}: {v[1]}" for k, v in r["left"].items())
              + (f"\n      no safe way back after {r['no_way_back']}" if r["no_way_back"] else ""), flush=True)
        return r, bad

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(job, range(len(layouts))))
    tot = sum(r["plants"] for r, _ in res)
    done = sum(r["done"] for r, _ in res)
    oor = sum(1 for r, _ in res for v in r["left"].values() if v[0].startswith("out of reach"))
    fails = sum(len(b) for _, b in res) + sum(len(r["no_way_back"]) for r, _ in res)
    summary = (f"SUMMARY: {done}/{tot} plants transplanted in {len(res)} layouts; {oor} out of the arm's reach for a flat grasp "
               f"(not possible even alone); {fails} planner failures (possible alone, not in the sequence, or no way back)")
    print(summary)
    (run_dir / "summary.txt").write_text(summary + "\n")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
