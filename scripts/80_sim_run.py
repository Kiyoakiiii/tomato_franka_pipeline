#!/usr/bin/env python
"""The whole transplant run simulated, end to end, for random layouts: 77 with every script it calls (68, 76, 69, 65,
71), the arm a stateful stand-in that checks every move against the world model (seedling.robot.StateSimRobot), the
close-up "measuring" the soil a few mm off the overview's value, no camera and no gripper. Each layout runs in its own
copy of the project under .tmp/simruns/ (the run rewrites the tray and disc files), so nothing of the real state is
touched. Tells per layout how many plants went in, where plants were left and why, simulated collisions, and the
planning time per step.

    .venvs/pipeline/bin/python scripts/80_sim_run.py --n 8 --seed 3
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-plants", type=int, default=6)
    ap.add_argument("--grid", default="experiments/tray/20261105_174439/tray_grid.json", help="the tray (its plants) to start from")
    ap.add_argument("--disc-r", type=float, nargs=2, default=[-0.02, 0.08])
    ap.add_argument("--disc-l", type=float, nargs=2, default=[-0.08, 0.08])
    ap.add_argument("--disc-z", type=float, nargs=2, default=[-0.02, 0.04])
    ap.add_argument("--tray-r", type=float, nargs=2, default=[-0.03, 0.04])
    ap.add_argument("--tray-l", type=float, nargs=2, default=[-0.05, 0.05])
    ap.add_argument("--tray-z", type=float, nargs=2, default=[-0.03, 0.05])
    ap.add_argument("--soil-dz", type=float, nargs=2, default=[-0.005, 0.013], help="close-up soil minus overview soil (m)")
    ap.add_argument("--out", default=".tmp/sim_run.json")
    args = ap.parse_args()
    cfg = config_from_args(args)
    base_grid = json.loads((PROJECT_DIR / args.grid).read_text())
    base_disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
    n_holes = len(base_disc["holes"])
    rng = np.random.default_rng(args.seed)

    def frame(c):
        c = np.asarray(c, float)[:2]
        u = c / max(np.linalg.norm(c), 1e-9)
        return u, np.array([-u[1], u[0]])

    ud, ld = frame(base_disc["centre"])
    ut, lt = frame(np.mean([v[:2] for v in base_grid["cells"].values()], axis=0))
    layouts = [(0.0,) * 6]
    while len(layouts) < args.n:
        layouts.append(tuple(float(rng.uniform(*r)) for r in (args.disc_r, args.disc_l, args.disc_z, args.tray_r, args.tray_l, args.tray_z)))
    root = PROJECT_DIR / ".tmp" / "simruns"
    root.mkdir(parents=True, exist_ok=True)

    def sandbox(i, lay):
        sb = root / f"L{i:02d}"
        if sb.exists():
            shutil.rmtree(sb)
        sb.mkdir()
        for d in ("scripts", "seedling", "configs"):
            shutil.copytree(PROJECT_DIR / d, sb / d, ignore=shutil.ignore_patterns("__pycache__"))
        (sb / "data").symlink_to(PROJECT_DIR / "data")
        (sb / "calibration").mkdir()
        for f in (PROJECT_DIR / "calibration").iterdir():
            if f.is_file() and f.suffix in (".yaml", ".json"):
                shutil.copy(f, sb / "calibration" / f.name)
        (sb / ".tmp").mkdir()
        wc = PROJECT_DIR / ".tmp" / "fr3_world_cache.pkl"
        if wc.exists():
            shutil.copy(wc, sb / ".tmp" / wc.name)
        (sb / "experiments" / "grasp").mkdir(parents=True)
        # the layout: the tray and the disc moved; a full tray, an empty disc
        dr, dl, dz, tr, tl, tz = lay
        d = ud * dr + ld * dl
        t = ut * tr + lt * tl
        g = json.loads(json.dumps(base_grid))
        g["cells"] = {k: [v[0] + t[0], v[1] + t[1], v[2] + tz] for k, v in g["cells"].items()}
        g["z_top"] = float(g["z_top"]) + tz
        g["soil_z_overview"] = {k: v + tz for k, v in (g.get("soil_z_overview") or {}).items()}
        (sb / cfg["tray"]["grid_file"]).parent.mkdir(parents=True, exist_ok=True)
        (sb / cfg["tray"]["grid_file"]).write_text(json.dumps(g, indent=1))
        dd = json.loads(json.dumps(base_disc))
        dd["centre"] = [dd["centre"][0] + d[0], dd["centre"][1] + d[1]]
        dd["holes"] = [[h[0] + d[0], h[1] + d[1]] for h in dd["holes"]]
        dd["z_top"] = float(dd["z_top"]) + dz
        dd["free_holes"] = list(range(1, n_holes + 1))
        dd["occupied_holes"] = []
        dd.pop("ok_last_picture", None)
        (sb / "calibration" / "disc.json").write_text(json.dumps(dd, indent=1))
        (sb / ".tmp" / "sim_state.json").write_text(json.dumps({"q": cfg["tray"]["overview_q"]}))
        return sb

    def run(item):
        i, lay = item
        sb = sandbox(i, lay)
        env = dict(os.environ, SEEDLING_SIM_STATE=str(sb / ".tmp" / "sim_state.json"),
                   SEEDLING_SIM_SOIL_DZ=f"{args.soil_dz[0]},{args.soil_dz[1]},{args.seed * 100 + i}")
        env.pop("SEEDLING_SIM_Q", None)
        env.pop("SEEDLING_CAMERA_SOCK", None)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", "scripts/77_program_run.py", "--max-plants", str(args.max_plants)], cwd=sb,
                           capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        (sb / "run.log").write_text(out)
        st = json.loads((sb / ".tmp" / "sim_state.json").read_text())
        moves = st.get("log", [])
        done = re.findall(r"^(\w\d+) -> H(\d+) done", out, re.M)
        left = re.findall(r"^the cycle for (\w\d+) found no safe plan after the close-up.*$", out, re.M)
        stopped = re.findall(r"^the cycle for (\w\d+) stopped \(exit (\d+)\).*$", out, re.M)
        steps = [json.loads(m) for m in re.findall(r"^STEP TIMES (.*)$", out, re.M)]
        ready = [float(x) for x in re.findall(r"^>>> transplant \w\d+ -> H\d+ \(ready in ([\d.]+) s\)", out, re.M)]
        prog_ready = re.findall(r"\(the program's arm configuration ready in ([\d.]+) s\)", out)
        fails = [ln for ln in out.splitlines() if "failed or refused" in ln or "Traceback" in ln or "simulated collision" in ln]
        return {"layout": lay, "done": len(done), "placed": done, "left_after_close_up": left, "stopped": stopped,
                "collisions": [m_ for m_ in moves if m_.get("failed")], "min_margin_mm": min([m_["margin_mm"] for m_ in moves] or [0]),
                "steps": steps, "ready_s": ready, "program_ready_s": float(prog_ready[0]) if prog_ready else None,
                "fails": fails[:6], "end": (out.strip().splitlines() or [""])[-1], "wall_s": round(time.time() - t0, 1),
                "sandbox": str(sb.relative_to(PROJECT_DIR))}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run, enumerate(layouts)))
    want = min(args.max_plants, n_holes)
    print(f"{len(res)} layouts simulated end to end (disc moved r/l/z, tray moved r/l/z in cm); up to {want} plants each")
    for r in res:
        lay = " ".join(f"{v * 100:+4.0f}" for v in r["layout"])
        st_ = r["stopped"]
        print(f"  [{lay}]  {r['done']}/{want} in; left after the close-up {r['left_after_close_up'] or '-'}; "
              f"{'STOPPED ' + str(st_) + '; ' if st_ else ''}collisions {len(r['collisions'])}; closest move {r['min_margin_mm']:+.0f} mm; "
              f"program {r['program_ready_s']} s; {r['wall_s']:.0f} s  ({r['sandbox']})")
        for f in r["fails"][:3]:
            print(f"      {f[:160]}")
    st = collections.defaultdict(list)
    for r in res:
        for s in r["steps"]:
            for k, v in s.items():
                st[k].append(v)
    tot = sum(r["done"] for r in res)
    print(f"TOTAL {tot}/{want * len(res)} ({100 * tot / max(1, want * len(res)):.0f}%); collisions {sum(len(r['collisions']) for r in res)}; "
          f"stops {sum(len(r['stopped']) for r in res)}; left after a close-up {sum(len(r['left_after_close_up']) for r in res)}; wall {time.time() - t0:.0f} s")
    print("planning steps in the cycle (s, mean / max): " + ", ".join(
        f"{k} {np.mean(v):.1f}/{max(v):.1f}" for k, v in st.items() if k in ("plan", "plan the placement")))
    rd = [x for r in res for x in r["ready_s"]]
    if rd:
        print(f"waiting for the next plant's plan between cycles (s, mean / max): {np.mean(rd):.1f}/{max(rd):.1f}; the program's "
              f"configuration ready after (s): {[r['program_ready_s'] for r in res]}")
    (PROJECT_DIR / args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
