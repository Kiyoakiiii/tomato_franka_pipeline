#!/usr/bin/env python
"""Where do the tray and the disc work best? The whole job (the plants now in the tray, an empty disc) is simulated
with the program (76 build: teaching, each plant's program with its observation pose and turn, the placement) for the
layout as located and for the disc (or the tray) moved by a few centimetres; the table tells how many plants each
layout transplants and which way to move what (a reachability analysis with the real program, 2026-11-05).
Planning only (no robot, no camera); several layouts at a time.

    .venvs/pipeline/bin/python scripts/78_layout_check.py            # the disc moved
    .venvs/pipeline/bin/python scripts/78_layout_check.py --tray     # the tray moved
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

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
    ap.add_argument("--tray", action="store_true", help="move the tray instead of the disc")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cfg = config_from_args(args)
    disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    n_holes = len(disc["holes"])
    qf = PROJECT_DIR / ".tmp" / "overview_q.json"
    qf.parent.mkdir(exist_ok=True)
    qf.write_text(json.dumps({"q": cfg["tray"]["overview_q"]}))
    env = dict(os.environ, SEEDLING_SIM_Q=str(qf))
    # offsets: radial (away from the robot), lateral (to the robot's left), up; in cm
    c = np.array(disc["centre"][:2], float) if not args.tray else None
    if args.tray:
        grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
        c = np.mean([v[:2] for v in grid["cells"].values()], axis=0)
    u = c / max(np.linalg.norm(c), 1e-9)
    left = np.array([-u[1], u[0]])
    cands = [(0, 0, 0), (3, 0, 0), (6, 0, 0), (-3, 0, 0), (0, 5, 0), (0, -5, 0), (3, 5, 0), (3, -5, 0), (0, 0, 3), (0, 0, -3)]
    if args.tray:
        cands = [(0, 0, 0), (-3, 0, 0), (3, 0, 0), (0, 5, 0), (0, -5, 0), (0, 0, 3), (0, 0, 5), (0, 0, -3)]

    def run(cand):
        dr, dl, dz = cand
        d = u * dr / 100 + left * dl / 100
        shift = f"{d[0]:.4f},{d[1]:.4f},{dz / 100:.4f}"
        cmd = [sys.executable, "-u", "scripts/76_program.py", "build", "--reteach", "--free", ",".join(str(h) for h in range(1, n_holes + 1)),
               "--shift-tray" if args.tray else "--shift-disc", shift]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        done = re.findall(r"^  (\w+\d+) -> H(\d+): (.*)$", out, re.M)
        m = re.search(r"the job: (\d+) plants", out)
        return cand, len(done), int(m.group(1)) if m else 0, sorted(int(h) for _, h, _ in done), \
            sum("default template" in n for *_, n in done), time.time() - t0, out

    what = "tray" if args.tray else "disc"
    print(f"the {what} as located and moved; the job: the plants now in the tray, an empty disc ({n_holes} holes)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run, cands))
    print(f"{'move the ' + what + ' (cm)':34s} plants done   holes used          default / variant")
    for cand, n, n_pl, holes, n_def, dt, out in res:
        dr, dl, dz = cand
        desc = "as it is" if cand == (0, 0, 0) else ", ".join(
            p for p in ([f"{abs(dr)} {'away from' if dr > 0 else 'towards'} the robot"] if dr else []) +
            ([f"{abs(dl)} to the robot's {'left' if dl > 0 else 'right'}"] if dl else []) +
            ([f"{abs(dz)} {'up' if dz > 0 else 'down'}"] if dz else []))
        if "no program for" in out or "teaching: no arm configuration" in out:
            print(f"  {desc:32s} no program")
            continue
        print(f"  {desc:32s} {n:2d} of {min(n_pl, n_holes):2d}       {str(holes):18s}  {n_def} / {n - n_def}")
    best = max(res, key=lambda r_: (r_[1], r_[4], -sum(abs(v) for v in r_[0])))
    base = res[0]
    if best[1] > base[1]:
        dr, dl, dz = best[0]
        print(f"BEST: move the {what} {best[0]} cm (away from the robot, to its left, up): {best[1]} plants instead of {base[1]}")
    else:
        print(f"the {what} is fine where it is ({base[1]} plants)")
    print(f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
