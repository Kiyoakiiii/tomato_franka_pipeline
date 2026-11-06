#!/usr/bin/env python
"""Transplant run as an industrial program (seedling.program, scripts/76_program.py), 2026-11-05:

1. overview pose (68), one picture (60): the tray and the disc located ONCE (they do not move);
2. the program: taught for this layout (or the one taught before, if the layout is the same), checked for the job;
3. per plant, in the user's order (nearest row first, right to left) into the holes in the program's order (farthest
   first): the template moved to its cell and hole (76 grasp / place, ~2 s, no search), then the cycle (69
   --program --no-return): observe close up, the template at the measured soil height, pre-grasp, approach, close,
   lift, carry (the hand flat), insert, open, retreat; from there straight to the next plant's observation pose. After
   each plant its cell counts as empty and its hole as taken (written to the grid and disc files);
4. at the end, back at the overview, one picture: each plant's cell empty and its hole taken.

touch experiments/grasp/STOP to stop between steps (never stop a script while the arm moves).

    .venvs/pipeline/bin/python scripts/77_program_run.py --max-plants 3
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import os
import subprocess
import sys
import time

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--max-plants", type=int, default=6)
    ap.add_argument("--skip", default="", help="comma list of cells not to take")
    ap.add_argument("--reteach", action="store_true", help="teach the program again even if the layout is the same")
    args = ap.parse_args()
    cfg = config_from_args(args)
    py = sys.executable
    gdir = PROJECT_DIR / "experiments" / "grasp"
    stop = gdir / "STOP"
    if stop.exists():
        stop.unlink()

    def run(cmd, quiet=False, env=None):
        if quiet:
            r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR, capture_output=True, text=True, env=env)
            return r.returncode, r.stdout + r.stderr
        r = subprocess.run([py, "-u", *cmd], cwd=PROJECT_DIR, env=env)
        return r.returncode, ""

    # planning without a robot connection, as from the overview pose (the observation pose is worked out so)
    qf = PROJECT_DIR / ".tmp" / "overview_q.json"
    qf.parent.mkdir(exist_ok=True)
    qf.write_text(json.dumps({"q": cfg["tray"]["overview_q"]}))
    env_plan = dict(os.environ, SEEDLING_SIM_Q=str(qf))

    if os.environ.get("SEEDLING_SIM_STATE"):
        # the whole-run simulation (scripts/80): no camera, no robot; the tray and disc files as they are
        return _run(args, cfg, gdir, stop, run, env_plan, sim=True)
    # the camera kept open for the run (63); its output into a file
    sock = PROJECT_DIR / ".tmp" / "camera.sock"
    cam_log_path = PROJECT_DIR / ".tmp" / "camera_server.log"
    cam_log = open(cam_log_path, "w")
    cam_srv = subprocess.Popen([py, "-u", "scripts/63_camera_server.py", "--sock", str(sock)], cwd=PROJECT_DIR,
                               stdout=cam_log, stderr=subprocess.STDOUT, text=True)
    t_end = time.time() + 30
    while time.time() < t_end and cam_srv.poll() is None:
        if "camera server ready" in cam_log_path.read_text(errors="replace"):
            os.environ["SEEDLING_CAMERA_SOCK"] = str(sock)
            break
        time.sleep(0.2)
    else:
        print("camera server did not start: the scripts open the camera themselves")
    try:
        return _run(args, cfg, gdir, stop, run, env_plan)
    finally:
        if cam_srv.poll() is None:
            cam_srv.terminate()
            try:
                cam_srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cam_srv.kill()


def _run(args, cfg, gdir, stop, run, env_plan, sim=False) -> int:
    from seedling.order import plant_order

    def write_json(path, d):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, path)

    gf = PROJECT_DIR / cfg["tray"]["grid_file"]
    df = PROJECT_DIR / "calibration" / "disc.json"
    print(">>> overview and locate (once for the run)", flush=True)
    if run(["scripts/68_goto_overview.py"])[0] != 0:
        print("cannot go to the overview pose: stopped")
        return 1
    rc, out = run(["scripts/60_tray_locate.py"], quiet=True) if not sim else (0, "(simulated: the files as they are)")
    print("\n".join(ln[:200] for ln in out.splitlines() if ln.startswith(("grid", "disc", "overlay", "CORRECTED", "REJECTED")) or "cells with" in ln))
    disc = json.loads(df.read_text())
    if rc != 0 or not disc.get("ok") or disc.get("ok_last_picture") is False:
        print("the tray or the disc was not located: stopped")
        return 1
    print(">>> the program (worked out in the background: the first plant starts as soon as its program is there)", flush=True)
    t0 = time.time()
    pf = gdir / "program.json"
    t_launch = time.time()
    blog = open(PROJECT_DIR / ".tmp" / "program_build.log", "w")
    build = subprocess.Popen([sys.executable, "-u", "scripts/76_program.py", "build", *(["--reteach"] if args.reteach else [])],
                             cwd=PROJECT_DIR, env=env_plan, stdout=blog, stderr=subprocess.STDOUT, text=True)

    def program():
        """the program as written by the build of this run (None before its first write)"""
        try:
            if pf.stat().st_mtime < t_launch:
                return None
            return json.loads(pf.read_text())
        except (OSError, ValueError):
            return None

    def wait_for(k):
        """until the build has decided about plant k (k None: until the taught configurations are there), or finished"""
        while True:
            pr = program()
            if pr is not None and ((k is None and (pr.get("configs") or pr.get("cells"))) or k in pr.get("cells", {}) or k in pr.get("no_way", {})
                                   or not pr.get("building", True)):
                return pr
            if build.poll() is not None:
                return program()
            time.sleep(0.3)

    def build_lines():
        txt = (PROJECT_DIR / ".tmp" / "program_build.log").read_text(errors="replace")
        return [ln for ln in txt.splitlines() if not ln.startswith((" - ", "nanobind")) and "leaked" not in ln and "Random" not in ln
                and "closing point raised" not in ln]

    def choose(todo, free, ignore=(), extra_taken=()):
        """the next plant in the user's order into the first free hole the program takes it to: (cell, hole) or None"""
        ig = ["--ignore-cells", ",".join(ignore)] if ignore else []
        for k in todo:
            pr = wait_for(k)
            if pr is None or k not in pr.get("cells", {}):
                continue
            rg, _o = run(["scripts/76_program.py", "grasp", "--cell", k, "--from-observe", *ig], quiet=True, env=env_plan)
            if rg != 0:
                continue
            # the hole the program meant for it first (the others' holes kept for them), then the other free ones
            mine = pr["cells"][k].get("hole")
            for hole in ([mine] if mine in free else []) + [h for h in free if h != mine]:
                if hole in extra_taken:
                    continue
                rp, _o = run(["scripts/76_program.py", "place", "--cell", k, "--hole", str(hole), *ig,
                              *(["--taken", ",".join(map(str, extra_taken))] if extra_taken else [])], quiet=True, env=env_plan)
                if rp == 0:
                    return (k, hole)
        return None

    hole_order = None
    done, placed, failed = 0, [], set()
    skip = set(filter(None, args.skip.split(",")))
    end_check = False
    ahead = None  # the next plant chosen in the background while the arm moves the current one
    try:
        while True:
            if stop.exists():
                print("STOPPED on request")
                break
            if done >= args.max_plants:
                print(f"{done} plants done (--max-plants)")
                end_check = True
                break
            grid, disc = json.loads(gf.read_text()), json.loads(df.read_text())
            if hole_order is None:
                pr = wait_for(None)
                if pr is None or not (pr.get("configs") or pr.get("cells")):
                    build.wait()
                    print("\n".join(build_lines()[-6:]))
                    print("no program for this layout: stopped")
                    return 1
                hole_order = list(pr["hole_order"])
                print(f"(the program's arm configuration ready in {time.time() - t0:.1f} s)")
            free = [h for h in hole_order if h in disc.get("free_holes", [])]
            todo = [k for k in plant_order(grid, grid.get("occupied", [])) if k not in skip and k not in failed]
            if not free:
                print("DISC FULL")
                end_check = True
                break
            t1 = time.time()
            if ahead is not None:
                ahead["thread"].join()
                chosen = ahead["chosen"] if (ahead["free"] == free and ahead["todo"] == todo) else None
                ahead = None
            else:
                chosen = None
            if chosen is None:
                chosen = choose(todo, free)
            if chosen is None:
                print(f"no plant left the program takes into a free hole (plants {todo}, holes {free}): stopped")
                print("  (scripts/78_layout_check.py tells how far to move the disc or the tray so that more of them work)")
                end_check = True
                break
            k, hole = chosen
            print(f">>> transplant {k} -> H{hole} (ready in {time.time() - t1:.1f} s)", flush=True)
            # the next one meanwhile, for the state after this one (its cell empty, its hole taken)
            if done + 1 < args.max_plants:
                import threading

                todo_n, free_n = [x for x in todo if x != k], [h for h in free if h != hole]
                ahead = {"todo": todo_n, "free": free_n, "chosen": None}

                def plan_ahead(a=ahead, todo_n=todo_n, free_n=free_n, k=k, hole=hole):
                    a["chosen"] = choose(todo_n, free_n, ignore=[k], extra_taken=[hole])

                ahead["thread"] = threading.Thread(target=plan_ahead, daemon=True)
                ahead["thread"].start()
            no_obs = (program() or {}).get("cells", {}).get(k, {}).get("observe") == "none"
            rc, _ = run(["scripts/69_pick_cycle.py", "--cell", k, "--lift-mm", "100", "--place", str(hole), "--no-return", "--program",
                         *(["--no-observe"] if no_obs else [])])
            if rc == 3:
                print(f"the cycle for {k} found no safe plan after the close-up (nothing touched): {k} left, on with the next")
                failed.add(k)
                if ahead is not None:
                    ahead["thread"].join()
                    ahead = None
                continue
            if rc != 0:
                print(f"the cycle for {k} stopped (exit {rc}): stopped here, the arm is where that step left it")
                break
            try:  # the hole it went into (76 place takes another free one when the close-up moved the grasp)
                hole = int(json.loads((gdir / f"{k}_place_plan.json").read_text())["hole"])
            except (OSError, ValueError, KeyError):
                pass
            done += 1
            placed.append((k, hole))
            g_ = json.loads(gf.read_text())
            for key_ in ("occupied", "planted"):
                if key_ in g_:
                    g_[key_] = [x for x in g_[key_] if x != k]
            write_json(gf, g_)
            d_ = json.loads(df.read_text())
            d_["free_holes"] = [h for h in d_.get("free_holes", []) if h != hole]
            d_["occupied_holes"] = sorted(set(d_.get("occupied_holes", [])) | {hole})
            write_json(df, d_)
            print(f"{k} -> H{hole} done ({done}); straight on", flush=True)
    finally:
        if ahead is not None:
            ahead["thread"].join()
        if build.poll() is None:
            build.wait()
        print("the program:\n" + "\n".join(build_lines()))
    if end_check and placed and not stop.exists() and not sim:
        print(">>> back to the overview, one picture to check", flush=True)
        if run(["scripts/68_goto_overview.py", "--after-place", placed[-1][0]])[0] == 0:
            rc, out = run(["scripts/60_tray_locate.py"], quiet=True)
            grid, disc = json.loads(gf.read_text()), json.loads(df.read_text())
            if rc == 0 and disc.get("ok") and disc.get("ok_last_picture") is not False:
                for k, hole in placed:
                    ok_cell, ok_hole = k not in grid.get("occupied", []), hole not in disc.get("free_holes", [])
                    print(f"CHECK {'ok' if ok_cell and ok_hole else 'FAILED'}: {k} -> H{hole}: cell empty {ok_cell}, hole taken {ok_hole}")
            else:
                print("CHECK: the tray or the disc was not located on the last picture; please look")
    print(f"TRANSPLANT RUN END: {done} plants transplanted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
