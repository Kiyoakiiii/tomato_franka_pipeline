#!/usr/bin/env python
"""How general and how fast is the program? Random layouts (the tray and the disc each moved by a few cm and raised /
lowered) are each simulated as a whole job (76 build: teaching, every plant's program with its observation pose and
turn, the placement; a full tray, an empty disc); the plants transplanted, the reasons plants were left, and the
planning time are summed up. Planning only (no robot, no camera).

    .venvs/pipeline/bin/python scripts/79_program_eval.py --n 12 --seed 1
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import collections
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grid", default="experiments/tray/20261105_174439/tray_grid.json", help="the tray (its plants) to start from")
    ap.add_argument("--disc-r", type=float, nargs=2, default=[-0.02, 0.08], help="disc moved away from the robot (m)")
    ap.add_argument("--disc-l", type=float, nargs=2, default=[-0.08, 0.08], help="disc moved to the robot's left (m)")
    ap.add_argument("--disc-z", type=float, nargs=2, default=[-0.02, 0.04])
    ap.add_argument("--tray-r", type=float, nargs=2, default=[-0.03, 0.04])
    ap.add_argument("--tray-l", type=float, nargs=2, default=[-0.05, 0.05])
    ap.add_argument("--tray-z", type=float, nargs=2, default=[-0.03, 0.05])
    ap.add_argument("--out", default=".tmp/program_eval.json")
    args = ap.parse_args()
    cfg = config_from_args(args)
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    grid = json.loads((PROJECT_DIR / args.grid).read_text())
    n_holes = len(disc["holes"])
    qf = PROJECT_DIR / ".tmp" / "overview_q.json"
    qf.parent.mkdir(exist_ok=True)
    qf.write_text(json.dumps({"q": cfg["tray"]["overview_q"]}))
    env = dict(os.environ, SEEDLING_SIM_Q=str(qf))
    rng = np.random.default_rng(args.seed)

    def frame(c):
        c = np.asarray(c, float)[:2]
        u = c / max(np.linalg.norm(c), 1e-9)
        return u, np.array([-u[1], u[0]])

    ud, ld = frame(disc["centre"])
    ut, lt = frame(np.mean([v[:2] for v in grid["cells"].values()], axis=0))
    layouts = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    while len(layouts) < args.n:
        layouts.append(tuple(float(rng.uniform(*r)) for r in (args.disc_r, args.disc_l, args.disc_z, args.tray_r, args.tray_l, args.tray_z)))

    def run(lay):
        dr, dl, dz, tr, tl, tz = lay
        d = ud * dr + ld * dl
        t = ut * tr + lt * tl
        cmd = [sys.executable, "-u", "scripts/76_program.py", "build", "--reteach", "--grid", args.grid,
               "--free", ",".join(str(h) for h in range(1, n_holes + 1)),
               f"--shift-disc={d[0]:.4f},{d[1]:.4f},{dz:.4f}", f"--shift-tray={t[0]:.4f},{t[1]:.4f},{tz:.4f}"]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        done = re.findall(r"^  (\w+\d+) -> H(\d+): (.*)$", out, re.M)
        left = re.findall(r"^  (\w+\d+): no way found \((.*)\)$", out, re.M)
        full = "the disc counts as full" in out
        tm = re.search(r"program built in ([\d.]+) s", out)
        teach = "teaching: no arm configuration" in out
        if r.returncode not in (0, 1) or "error:" in out and "usage:" in out:
            teach = False
        return {"layout": lay, "done": len(done), "holes": sorted(int(h) for _, h, _ in done),
                "variants": sum("default template" not in n for *_, n in done), "left": left, "disc_full": full,
                "not_robust": sum("NOT checked robust" in n for *_, n in done), "no_close_up": sum("no close-up" in n for *_, n in done),
                "no_teach": teach, "build_s": float(tm.group(1)) if tm else time.time() - t0, "wall_s": time.time() - t0,
                "error": "Traceback" in out or "usage:" in out, "tail": out.strip().splitlines()[-3:] if ("Traceback" in out or "usage:" in out) else []}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run, layouts))
    print(f"{len(res)} layouts (disc moved r/l/z, tray moved r/l/z in cm), a full tray ({len(grid.get('occupied', []))} plants) "
          f"into an empty disc ({n_holes} holes)")
    for r in res:
        lay = " ".join(f"{v * 100:+4.0f}" for v in r["layout"])
        note = "NO PROGRAM (teaching)" if r["no_teach"] else ("ERROR " + " | ".join(r["tail"]) if r["error"] else "")
        print(f"  [{lay}]  {r['done']}/{n_holes} plants ({r['variants']} variants, {r.get('not_robust', 0)} not robust, "
              f"{r.get('no_close_up', 0)} without close-up)  build {r['build_s']:5.1f} s  {note}")
    reasons = collections.Counter()
    for r in res:
        for _k, why in r["left"]:
            for part in why.split("; "):
                part = re.sub(r"approach [+-]?\d+ deg from the program's, ", "", part)
                part = re.sub(r"[+-]?\d+(\.\d+)? mm", "N mm", part)
                part = re.sub(r"heading [+-]?\d+(\.\d+)?", "heading N", part)
                part = re.sub(r"H\d", "H?", part)
                reasons[part[:110]] += 1
    tot = sum(r["done"] for r in res)
    print(f"TOTAL {tot}/{n_holes * len(res)} ({100 * tot / (n_holes * len(res)):.0f}%); layouts with no program "
          f"{sum(r['no_teach'] for r in res)}; errors {sum(r['error'] for r in res)}; not robust {sum(r.get('not_robust', 0) for r in res)}; "
          f"without close-up {sum(r.get('no_close_up', 0) for r in res)}; build mean {np.mean([r['build_s'] for r in res]):.1f} s, "
          f"max {max(r['build_s'] for r in res):.1f} s; wall {time.time() - t0:.0f} s")
    print("reasons plants were left (most common):")
    for part, n in reasons.most_common(12):
        print(f"  {n:3d}  {part}")
    (PROJECT_DIR / args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
