#!/usr/bin/env python
"""Keep the ZED open and hand out frames: the scripts of a run (locate.py, pick.py) ask it for a picture over a UNIX socket
instead of opening the camera themselves (each open + close took 3-20 s; 2026-11-04). run.py starts it and stops it;
with it running, seedling.camera.ZedCamera(cfg) connects to it (env SEEDLING_CAMERA_SOCK) instead of opening the
camera. Nothing else may open the camera while it runs.

    .venvs/pipeline/bin/python scripts/camera_server.py --sock .tmp/camera.sock
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import io
import json
import os
import signal
import socket
import struct

import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--sock", default=str(PROJECT_DIR / ".tmp" / "camera.sock"))
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.camera import ZedCamera

    os.environ.pop("SEEDLING_CAMERA_SOCK", None)  # this process opens the camera itself
    cam = ZedCamera(cfg)
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
    print(f"camera server ready on {path}", flush=True)
    try:
        while not stop[0]:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                req = json.loads(conn.makefile("r").readline() or "{}")
                try:
                    f = cam.grab(n_average=int(req.get("n_average", 1)), with_right=bool(req.get("with_right", True)))
                    buf = io.BytesIO()
                    np.savez(buf, rgb=f.rgb, xyz=f.xyz, conf=f.conf if f.conf is not None else np.zeros(0, np.float32),
                             rgb_right=f.rgb_right if f.rgb_right is not None else np.zeros(0, np.uint8),
                             intr=np.array([f.intr.fx, f.intr.fy, f.intr.cx, f.intr.cy, f.intr.width, f.intr.height], float),
                             meta=np.array(json.dumps(f.meta)))
                    data = buf.getvalue()
                except Exception as exc:  # noqa: BLE001  (tell the client, keep serving)
                    data = b"ERR" + str(exc).encode()
                try:
                    conn.sendall(struct.pack("<Q", len(data)) + data)
                except OSError as exc:  # the client gave up waiting (e.g. while the ZED recovered from a USB drop): keep serving
                    print(f"client gone before the frame was sent ({exc})", flush=True)
    finally:
        cam.close()
        if os.path.exists(path):
            os.remove(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
