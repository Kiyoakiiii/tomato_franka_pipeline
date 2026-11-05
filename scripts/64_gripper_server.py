#!/usr/bin/env python
"""Keep the pin-brake gripper's serial port open for a whole run and take its commands over a UNIX socket (72 starts
and stops it; seedling.gripper.make_gripper connects to it when SEEDLING_GRIPPER_SOCK is set).

Each script used to open the port itself: 3 s waiting for a firmware banner that never came (the Arduino is not
reset), and not knowing whether the gripper was open, a full open (3 s deflating + 2.6 s servo) before every close
(2026-11-04: ~20 s per plant). Here the port stays open and the state is known.

Commands (one JSON line, one JSON line back): {"cmd": "open", "wait": true|false}, {"cmd": "close"},
{"cmd": "pressure"}.

    .venvs/pipeline/bin/python scripts/64_gripper_server.py --sock .tmp/gripper.sock
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import json
import os
import signal
import socket
import time

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--sock", default=str(PROJECT_DIR / ".tmp" / "gripper.sock"))
    args = ap.parse_args()
    cfg = config_from_args(args)

    os.environ.pop("SEEDLING_GRIPPER_SOCK", None)  # this process talks to the gripper itself
    from seedling.gripper import make_gripper

    g = make_gripper(cfg, sim=False, step_mode=False)
    path = args.sock
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    stop = [False]

    def on_term(*_):
        stop[0] = True
        srv.close()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    print(f"gripper server ready on {path}", flush=True)
    try:
        while not stop[0]:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                f = conn.makefile("rw")
                req = json.loads(f.readline() or "{}")
                cmd = req.get("cmd")
                try:
                    if cmd == "open":
                        g.open(wait=bool(req.get("wait", True)))
                        out = {"ok": True, "pressure_kpa": g.pressure_kpa()}
                    elif cmd == "close":
                        r = g.grasp(0.003)
                        time.sleep(0.3)
                        out = {"ok": bool(r.command_ok), "reason": r.reason, "pressure_kpa": g.pressure_kpa()}
                    elif cmd == "pressure":
                        out = {"ok": True, "pressure_kpa": g.pressure_kpa()}
                    else:
                        out = {"ok": False, "reason": f"unknown command {cmd!r}"}
                except Exception as exc:  # noqa: BLE001  (tell the client, keep serving)
                    out = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
                f.write(json.dumps(out) + "\n")
                f.flush()
    finally:
        # the gripper stays as it is (open or holding): no reset of the Arduino on the way out
        g.release_port()
        if os.path.exists(path):
            os.remove(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
