"""The steps of a run executed in this process instead of one new process each (2026-11-10: a plant's cycle started ~10
processes, each importing, loading the plan and the world model and connecting to the robot again).

run_script("scripts/65_side_grasp.py", ["--cell", "C5", "approach"]) runs that script's main() here, with sys.argv set
as on the command line; the script's module is loaded once and kept (its imports and module-level work done once).
The return value is what the process's exit code would have been: main()'s return value or SystemExit's code, 1 on an
exception (its traceback printed). Set SEEDLING_SHARED_ROBOT for the robot connection to be shared too
(seedling.robot.FrankyRobot)."""
from __future__ import annotations

import importlib.util
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

from .config import PROJECT_DIR

_MODS: dict = {}


def _module(path):
    p = (PROJECT_DIR / path).resolve()
    if p not in _MODS:
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))
        spec = importlib.util.spec_from_file_location(f"_step_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[p] = mod
    return _MODS[p]


def run_script(path, argv, quiet: bool = False):
    """main() of the script at path (relative to the project) with these arguments: its exit code; with quiet, also
    what it printed: (code, text)"""
    buf = io.StringIO() if quiet else None
    old = sys.argv
    try:
        mod = _module(path)
        sys.argv = [str(PROJECT_DIR / path), *map(str, argv)]
        if quiet:
            with redirect_stdout(buf), redirect_stderr(buf):
                rc = _call(mod)
        else:
            rc = _call(mod)
    finally:
        sys.argv = old
        sys.stdout.flush()
    return (rc, buf.getvalue()) if quiet else rc


def _call(mod) -> int:
    try:
        r = mod.main()
    except SystemExit as e:
        r = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:  # noqa: BLE001  (as a process would: the traceback, exit code 1)
        traceback.print_exc()
        r = 1
    sys.stdout.flush()
    return int(r or 0)
