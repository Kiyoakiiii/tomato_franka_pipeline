#!/usr/bin/env python
"""Transplant plants from the tray into the disc, nearest first, until the disc is full (or nothing reachable is
left). Each round:

1. overview pose (68), locate tray and disc from one picture (60): stop if either is not found;
2. the disc's free holes (a hole with a plug shows no red interior): none left -> done;
3. the plants still in the tray, the row nearest the robot first, within a row from right to left (as seen from the
   robot) (a plug seen in the cell; cells whose only soil
   lies at the cell bottom, deeper than --scrap-depth-mm below the tray top, are soil scraps, not plugs);
   for each in turn a grasp plan (65, with an approach heading clear of the other plants) and a placement plan
   (71) are made without moving; the first plant with both safe is taken, the others wait for a later round
   (a plant behind others often becomes reachable once those are gone);
4. the full cycle for it (69 --place): observe, grasp, lift, carry, insert, open, retreat, overview;
5. check on the next overview picture: its cell must be empty and one more hole taken. Otherwise stop (the
   gripper cannot feel whether it holds the plant: only the picture tells).

touch experiments/grasp/STOP to stop after the current step (the running cycle stops at its next step).
With --pause it waits for a key before the first move to the overview and before each plant (space: go on, q: stop).

    .venvs/pipeline/bin/python scripts/72_transplant_all.py --dry-run     # plans only, from the saved files
    .venvs/pipeline/bin/python scripts/72_transplant_all.py
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import re
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
    ap.add_argument("--dry-run-discs", type=int, default=1, help="dry run: when the disc is full, go on with an empty one in its place "
                    "(as if swapped by hand), up to this many discs (with --max-plants large: every plant of the tray)")
    ap.add_argument("--pause", action="store_true", help="wait for space (go on) or q (stop) before the first move and before each plant")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.pause and not sys.stdin.isatty():
        print("--pause needs a terminal (keys are read from it)")
        return 1

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

    def go_on(what):  # --pause: space goes on, q stops; keys pressed earlier are dropped
        if not args.pause:
            return True
        import termios
        import tty

        print(f"=== 下一步: {what}   [空格 继续 / q 停止]", flush=True)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            termios.tcflush(fd, termios.TCIFLUSH)
            while True:
                c = sys.stdin.read(1)
                if c == " ":
                    return True
                if c in "qQ":
                    print("stopped (q)")
                    return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def load():
        grid = json.loads((PROJECT_DIR / cfg["tray"]["grid_file"]).read_text())
        disc = json.loads((PROJECT_DIR / "calibration" / "disc.json").read_text())
        return grid, disc

    def plants(grid):
        top = float(grid["z_top"])
        soil = grid.get("soil_z_overview") or {}
        # the row nearest the robot first (its cells' mean distance from the base), within a row from right to left
        # as seen from the robot (smallest base-frame y first) (user's rule, 2026-11-04)
        cells = grid["cells"]
        rows_ = {}
        for k in cells:
            rows_.setdefault(k[0], []).append(float(np.hypot(*cells[k][:2])))
        row_dist = {r: float(np.mean(v)) for r, v in rows_.items()}
        out = []
        for k in grid.get("occupied", []):
            z = soil.get(k)
            if z is not None and z < top - args.scrap_depth_mm / 1000:
                continue  # soil scraps at the cell bottom
            out.append((row_dist[k[0]], float(cells[k][1]), k))
        return [k for *_, k in sorted(out)]

    # the camera and the gripper's serial port kept open for the whole run (scripts/63_camera_server.py,
    # 64_gripper_server.py): each script asks them instead of opening the device itself (each camera open + close
    # took 3-20 s; each gripper connect waited 3 s for a banner and then opened the gripper once more, ~20 s per plant)
    servers = []
    if not args.dry_run:
        import os

        for script, sock_name, env_name in (("scripts/63_camera_server.py", "camera.sock", "SEEDLING_CAMERA_SOCK"),
                                            ("scripts/64_gripper_server.py", "gripper.sock", "SEEDLING_GRIPPER_SOCK")):
            sock = PROJECT_DIR / ".tmp" / sock_name
            logf = PROJECT_DIR / ".tmp" / sock_name.replace(".sock", "_server.log")
            logf.parent.mkdir(exist_ok=True)
            p_ = subprocess.Popen([py, "-u", script, "--sock", str(sock)], cwd=PROJECT_DIR,
                                  stdout=open(logf, "w"), stderr=subprocess.STDOUT)
            servers.append(p_)
            t_end = time.time() + 30
            ready = False
            while time.time() < t_end and p_.poll() is None:
                if "server ready" in logf.read_text(errors="replace"):
                    os.environ[env_name] = str(sock)
                    ready = True
                    break
                time.sleep(0.1)
            if not ready:
                print(f"{script} did not start: the scripts open the device themselves")
    try:
        return _run(args, cfg, py, grasp_dir, stop, log_file, log, run, go_on, load, plants)
    finally:
        for p_ in servers:
            if p_.poll() is None:
                p_.terminate()
                try:
                    p_.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p_.kill()


def _run(args, cfg, py, grasp_dir, stop, log_file, log, run, go_on, load, plants) -> int:
    done, prev = 0, None  # prev: (cell, free holes before) of the last cycle, to check it
    sim_removed, sim_taken = set(), set()
    discs_left = [args.dry_run_discs]
    failed = {}  # cell -> what was around it when its plan last failed
    while True:
        if stop.exists():
            print("STOPPED on request")
            break
        if not args.dry_run:
            if prev is None and not go_on("回总览位，拍照定位托盘和圆盘"):
                break
            print(">>> overview and locate", flush=True)
            if run(["scripts/68_goto_overview.py"])[0] != 0:
                print("cannot go to the overview pose: stopped")
                break
            rc, out = run(["scripts/60_tray_locate.py"], quiet=True)
            print("\n".join(ln[:200] for ln in out.splitlines() if ln.startswith(("grid", "disc", "overlay", "cross-check", "CORRECTED", "REJECTED")) or "cells with" in ln))
            if rc != 0:
                err = [ln for ln in out.splitlines() if ln.startswith(("RuntimeError", "FileNotFoundError")) or "Exception:" in ln]
                if err:
                    print(f"locating failed: {err[-1][:200]}: stopped")
                else:
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
        # what a flat hand in a natural posture can reach at all (seedling.feasibility: a few IK solutions, seconds):
        # told before anything moves, with what to change; those plants / holes are not planned
        from seedling.feasibility import check_layout

        # only when twisted postures are not allowed (grasp.max_posture_deg <= 60): with them allowed the plans decide
        feas = check_layout(grid, disc, todo, free, cfg) if float(cfg["grasp"].get("max_posture_deg", 60.0)) <= 60.0 else {"plants": {}, "holes": {}}
        bad_p = {k: v for k, v in feas["plants"].items() if v}
        bad_h = {h: v for h, v in feas["holes"].items() if v}
        if bad_p or bad_h:
            print(f"FEASIBILITY: {len(todo) - len(bad_p)}/{len(todo)} plants and {len(free) - len(bad_h)}/{len(free)} free holes "
                  "can be reached with a flat hand in a natural posture")
            if bad_p:
                print(f"  plants not reachable flat: {sorted(bad_p)}; to change: {feas.get('tray_change', '?')}")
            if bad_h:
                print(f"  holes not reachable flat: {sorted(bad_h)}; to change: {feas.get('disc_change', '?')}")
        if float(cfg["grasp"].get("max_posture_deg", 60.0)) <= 60.0:
            # twisted postures not allowed: those plants / holes are not planned at all
            todo = [k for k in todo if k not in bad_p]
            free = [h for h in free if h not in bad_h]
            if not todo or not free:
                print("nothing can be done in this layout: change it as suggested above")
                break
        print(f"free holes {free}; plants left (nearest first) {todo}")
        if not free and args.dry_run and todo and discs_left[0] > 1:
            discs_left[0] -= 1
            sim_taken.clear()
            free = [h for h in disc.get("free_holes", [])]
            print(f"DISC FULL: (dry run) an empty disc in its place, {discs_left[0]} left")
        if not free:
            print("DISC FULL")
            break
        if done >= args.max_plants:
            print(f"{done} plants done (--max-plants)")
            break
        chosen, no_place = None, []
        pitches = [float(x) for x in cfg["grasp"].get("side_pitch_deg", [26.0])]
        for k in todo:
            # a plant whose plan failed is planned again only when something changed around it: a plant within 15 cm
            # taken away, or the free holes (a plan that cannot succeed took ~1 min every round, 2026-11-04)
            ck = np.array(grid["cells"][k][:2])
            key = (frozenset(j for j in todo if j != k and np.hypot(*(np.array(grid["cells"][j][:2]) - ck)) < 0.15), tuple(free))
            if failed.get(k) == key:
                print(f"  {k}: not planned again (nothing changed around it since its plan failed)")
                continue
            obs = grasp_dir / f"{k}_observe.json"
            if obs.exists():
                obs.unlink()  # an old close-up of this cell is stale: the plan uses the overview's soil height
            # the flattest safe grasp first; when the plant cannot be placed from it (the hand stays as flat while
            # carrying and letting go), the grasp is planned again from the next steeper pitch
            left = pitches
            while left:
                # planned from the observation pose the cycle goes to first (the cycle plans again from there after the
                # close-up): a plan from the overview could pass where the one from the observation pose fails (2026-11-04, C7)
                rg, og = run(["scripts/65_side_grasp.py", "plan", "--from-observe", "--cell", k, "--lift-mm", str(args.lift_mm),
                              "--ignore-cells", ",".join(sorted(sim_removed)), "--pitch-deg", *map(str, left)], quiet=True)
                if rg != 0:
                    lines_ = og.splitlines()
                    # the one-line verdicts of the steepest pitch tried (the full report of the fallback plan is long)
                    why = [ln for ln in lines_ if "NOT SAFE at" in ln] or [ln for ln in lines_ if "not reachable" in ln or "in the way" in ln]
                    if "Connection to FCI refused" in og or "NetworkException" in og:
                        print("cannot connect to the robot (is FCI enabled in Desk?): stopped")
                        print("\n".join(og.strip().splitlines()[-6:]))
                        return 1
                    last = [ln for ln in og.splitlines() if ln.strip()]
                    print(f"  {k}: no safe grasp now ({why[-1].strip() if why else (last[-1].strip()[:160] if last else 'refused')})")
                    break
                m = [re.search(r"pitched (-?\d+) deg down, heading ([-+]?\d+) deg", ln) for ln in og.splitlines() if ln.startswith("approach pitched")]
                m = m[0] if m and m[0] else None
                rp, op = run(["scripts/71_place_in_disc.py", "plan", "--cell", k, "--free", ",".join(map(str, free)),
                              "--ignore-cells", ",".join(sorted(sim_removed))], quiet=True)
                if rp == 0:
                    hole = json.loads((grasp_dir / f"{k}_place_plan.json").read_text())["hole"]
                    chosen = (k, hole, f"俯角 {m[1]}°，方向 {m[2]}°" if m else "", [p_ for p_ in left if m is None or p_ >= float(m[1])])
                    print(f"  {k}: grasp pitched {m[1] if m else '?'} deg, heading {m[2] if m else '?'} deg, into H{hole}")
                    break
                why = [ln for ln in op.splitlines() if ln.strip()]
                steeper = [p_ for p_ in left if m is not None and p_ > float(m[1])]
                print(f"  {k}: grasp pitched {m[1] if m else '?'} deg ok, no safe placement from it ({why[-1].strip() if why else 'refused'})"
                      + ("; trying steeper" if steeper else ""))
                left = steeper
                if not left:
                    no_place.append(k)
            if chosen is not None:
                break
            failed[k] = key
        if chosen is None:
            print("no plant left that can be grasped and placed safely: stopped")
            if no_place:
                print(f"({len(no_place)} could be grasped but no free hole reached: the disc may be too close to the robot; "
                      "about 0.65 m from the base worked on 2026-10-28)")
            break
        k, hole, head, pitch_list = chosen
        log["rounds"].append({"cell": k, "hole": hole, "free_before": free})
        log_file.write_text(json.dumps(log, indent=1))
        if args.dry_run:
            # the way back to the overview after letting go (68 from the place plan's last pose)
            q_ua = json.loads((grasp_dir / f"{k}_place_plan.json").read_text())["q"]["up after"]
            r68, o68 = run(["scripts/68_goto_overview.py", "--dry-run", "--start-q", *map(str, q_ua),
                            "--free", ",".join(str(h) for h in free if h != hole) or "0",
                            "--ignore-cells", ",".join(sorted(sim_removed | {k}))], quiet=True)
            if r68 != 0:
                print(f"  {k}: (dry run) NO SAFE WAY BACK to the overview after H{hole}: "
                      + next((ln.strip() for ln in o68.splitlines() if "NOT SAFE" in ln or "cannot" in ln), o68.strip()[-160:]))
                break
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
        if not go_on(f"夹取 {k}，放进 H{hole}（{head}）"):
            break
        print(f">>> transplant {k} -> H{hole}", flush=True)
        rc, _ = run(["scripts/69_pick_cycle.py", "--cell", k, "--lift-mm", str(args.lift_mm), "--place", str(hole),
                     "--pitch-deg", *map(str, pitch_list)])
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
