#!/usr/bin/env python
"""Gripper checks (arm does not move). Works with gripper.type franka_hand or pin_brake.

    --homing                Franka Hand homing (fingers open fully then close)
    --open [MM]             open to width (default gripper.open_width_m; pin_brake always opens fully)
    --grasp-empty N         close on nothing N times: Franka Hand -> width for gripper.empty_close_width_m;
                            pin_brake -> stop time / pressure per close (it cannot tell empty from held)
    --grasp MM              grasp an object held between the pads (e.g. a dowel / stem of known diameter);
                            pin_brake closes with gripper.pin_brake (pressure_kpa / close_width_mm), MM is only logged
    --hold-s S              pin_brake with --grasp: keep holding S s and print the pad pressure every 0.5 s
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import time

import numpy as np

from _common import base_parser, config_from_args


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--homing", action="store_true")
    ap.add_argument("--open", type=float, nargs="?", const=-1.0, default=None)
    ap.add_argument("--grasp-empty", type=int, default=0)
    ap.add_argument("--grasp", type=float, default=None, help="expected object diameter in mm")
    ap.add_argument("--force", type=float, default=None)
    ap.add_argument("--hold-s", type=float, default=0.0)
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.gripper import make_gripper
    from seedling.robot import _console_confirm

    g = make_gripper(cfg, sim=False, step_mode=True, confirm_fn=_console_confirm)
    if g.width() is not None:
        print("width now:", g.width())
    if args.homing and hasattr(g, "homing"):
        print("homing ->", g.homing(), "width", g.width())
    if args.open is not None:
        print("open ->", g.open(None if args.open < 0 else args.open / 1000.0), "width", g.width())
    pin = cfg["gripper"]["type"] == "pin_brake"
    if args.grasp_empty:
        ws, hs = [], []
        for i in range(args.grasp_empty):
            g.open()
            r = g.grasp(0.0, args.force)
            ws.append(r.width_m)
            hs.append(r.holding)
            print(f"empty grasp {i+1}: {r}", getattr(g, "last", ""))
            time.sleep(0.3)
        ws = [w for w in ws if w is not None]
        if pin:
            print("\n(pin_brake: compare these stop times / pressures with --grasp on a stem)")
        if ws:
            print(f"\nempty close width: mean {np.mean(ws)*1000:.2f} mm  max {np.max(ws)*1000:.2f} mm")
            print(f"-> set gripper.empty_close_width_m: {np.max(ws):.4f} in configs/pipeline.yaml")
        g.open()
    if args.grasp is not None:
        g.open()
        _console_confirm("把测试物（如已知直径的木棍）放在两指之间，按 Enter 夹紧")
        w = float(cfg["gripper"]["empty_close_width_m"]) + 0.8 * args.grasp / 1000.0
        r = g.grasp(w, args.force)
        print(r, getattr(g, "last", ""))
        t_end = time.time() + args.hold_s
        while pin and time.time() < t_end:
            print("  pressure", g.pressure_kpa(), "kPa")
            time.sleep(0.5)
        _console_confirm("观察夹持情况/压痕后按 Enter 松开")
        g.open()
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
