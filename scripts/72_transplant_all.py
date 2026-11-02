#!/usr/bin/env python
"""Transplant plants from the tray into the disc, nearest first, until the disc is full (or nothing reachable is
left). Each round:

1. overview pose (68), locate tray and disc from one picture (60): stop if either is not found;
2. the disc's free holes (a hole with a plug shows no red interior): none left -> done;
3. the plants still in the tray, nearest the robot base first (a plug seen in the cell; cells whose only soil
   lies at the cell bottom, deeper than --scrap-depth-mm below the tray top, are soil scraps, not plugs);
   for each in turn a grasp plan (65, with an approach heading clear of the other plants) and a placement plan
   (71) are made without moving; the first plant with both safe is taken, the others wait for a later round
   (a plant behind others often becomes reachable once those are gone);
4. the full cycle for it (69 --place): observe, grasp, lift, carry, insert, open, retreat, overview;
5. check on the next overview picture: its cell must be empty and one more hole taken. Otherwise stop (the
   gripper cannot feel whether it holds the plant: only the picture tells).

touch experiments/grasp/STOP to stop after the current step (the running cycle stops at its next step).

    .venvs/pipeline/bin/python scripts/72_transplant_all.py --dry-run     # plans only, from the saved files
    .venvs/pipeline/bin/python scripts/72_transplant_all.py
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import subprocess
import sys
import time

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--max-plants", type=int, default=6)
    ap.add_argument("--lift-mm", type=float, default=100.0, help="straight up after the grasp (60 until 2026-10-28: leaves brushed)")
    ap.add_argument("--scrap-depth-mm", type=float, default=30.0)
    ap.add_argument("--skip", default="", help="comma list of cells not to take (e.g. a plant just missed)")
    ap.add_argument("--dry-run", action="store_true", help="no motion, no picture: plan the sequence from the saved tray/disc files")
    args = ap.parse_args()
    cfg = config_from_args(args)

    py = sys.executable
    grasp_dir = PROJECT_DIR / "experiments" / "grasp"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    stop = grasp_dir / "STOP"
    if stop.exists():
        stop.unlink()
    log_file = grasp_dir / f"transplant_all_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log = {"rounds": []}

    def run(cmd, quiet=False):
        if quiet:
            r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR, capture_output=True, text=True)
            return r.returncode, r.stdout + r.stderr
        r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR)
        return r.returncode, ""

    def load():
        grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
        disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
        return grid, disc

    def plants(grid):
        top = float(grid["z_top"])
        soil = grid.get("soil_z_overview") or {}
        out = []
        for k in grid.get("occupied", []):
            z = soil.get(k)
            if z is not None and z < top - args.scrap_depth_mm / 1000:
                continue  # soil scraps at the cell bottom
            x, y = grid["cells"][k][:2]
            out.append((float(np.hypot(x, y)), k))
        return [k for _, k in sorted(out)]

    done, prev = 0, None  # prev: (cell, free holes before) of the last cycle, to check it
    sim_removed, sim_taken = set(), set()
    while True:
        if stop.exists():
            print("STOPPED on request")
            break
        if not args.dry_run:
            print(">>> overview and locate", flush=True)
            if run(["scripts/68_goto_overview.py"])[0] != 0:
                print("cannot go to the overview pose: stopped")
                break
            rc, out = run(["scripts/60_tray_locate.py"], quiet=True)
            print("\n".join(ln[:200] for ln in out.splitlines() if ln.startswith(("grid", "disc", "overlay")) or "cells with" in ln))
            if rc != 0:
                print("the tray was not located: stopped (is it fully in view?)")
                break
        grid, disc = load()
        if not disc.get("ok") or disc.get("ok_last_picture") is False:
            print(f"the disc was not located: stopped ({disc.get('last_status', '')})")
            break
        free = [h for h in disc.get("free_holes", []) if h not in sim_taken]
        todo = [k for k in plants(grid) if k not in sim_removed and k not in args.skip.split(",")]
        if prev is not None and not args.dry_run:
            cell, free_before = prev
            ok_cell = cell not in grid.get("occupied", [])
            ok_hole = len(free) < len(free_before)
            log["rounds"][-1]["check"] = {"cell_empty": ok_cell, "hole_taken": ok_hole}
            if ok_cell and ok_hole:
                done += 1
                print(f"CHECK ok: {cell} is empty and a hole was taken ({done} transplanted)")
            else:
                print(f"CHECK FAILED for {cell}: cell empty {ok_cell}, one more hole taken {ok_hole}: stopped "
                      "(the plant may have slipped; please look)")
                break
        print(f"free holes {free}; plants left (nearest first) {todo}")
        if not free:
            print("DISC FULL")
            break
        if done >= args.max_plants:
            print(f"{done} plants done (--max-plants)")
            break
        chosen = None
        for k in todo:
            obs = grasp_dir / f"{k}_observe.json"
            if obs.exists():
                obs.unlink()  # an old close-up of this cell is stale: the plan uses the overview's soil height
            rg, og = run(["scripts/65_side_grasp.py", "plan", "--cell", k, "--lift-mm", str(args.lift_mm),
                          "--ignore-cells", ",".join(sorted(sim_removed))], quiet=True)
            if rg != 0:
                why = [ln for ln in og.splitlines() if "not reachable" in ln or "NOT SAFE" in ln or "TOO LOW" in ln]
                if "Connection to FCI refused" in og or "NetworkException" in og:
                    print("cannot connect to the robot (is FCI enabled in Desk?): stopped")
                    return 1
                last = [ln for ln in og.splitlines() if ln.strip()]
                print(f"  {k}: no safe grasp now ({why[-1].strip() if why else (last[-1].strip()[:160] if last else 'refused')})")
                continue
            rp, op = run(["scripts/71_place_in_disc.py", "plan", "--cell", k, "--free", ",".join(map(str, free))], quiet=True)
            if rp != 0:
                why = [ln for ln in op.splitlines() if ln.strip()]
                print(f"  {k}: grasp ok, no safe placement ({why[-1].strip() if why else 'refused'})")
                continue
            hole = json.loads((grasp_dir / f"{k}_place_plan.json").read_text())["hole"]
            head = [ln for ln in og.splitlines() if "heading" in ln and "from straight out" in ln]
            chosen = (k, hole)
            print(f"  {k}: grasp {head[0].strip().rstrip(':') if head else ''}, into H{hole}")
            break
        if chosen is None:
            print("no plant left that can be grasped and placed safely: stopped")
            break
        k, hole = chosen
        log["rounds"].append({"cell": k, "hole": hole, "free_before": free})
        log_file.write_text(json.dumps(log, indent=1))
        if args.dry_run:
            sim_removed.add(k)
            sim_taken.add(hole)
            done += 1
            if done >= args.max_plants:
                print(f"(dry run) {done} plants planned")
                break
            continue
        if stop.exists():
            print("STOPPED on request")
            break
        print(f">>> transplant {k} -> H{hole}", flush=True)
        rc, _ = run(["scripts/69_pick_cycle.py", "--cell", k, "--lift-mm", str(args.lift_mm), "--place", str(hole)])
        if rc != 0:
            print(f"the cycle for {k} stopped (exit {rc}): stopped here, the arm is where that step left it")
            break
        prev = (k, free)
    log["done"] = done
    log_file.write_text(json.dumps(log, indent=1))
    print(f"TRANSPLANT RUN END: {done} plants {'planned' if args.dry_run else 'transplanted and checked'}; log {log_file.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
