#!/usr/bin/env python
"""Pin-brake gripper, one command per call (the arm does not move). Parameters from configs/pipeline.yaml
(gripper.pin_brake: stop pressure 3 kPa, nominal width 2 mm; the values tested on tomato stems).

    .venvs/pipeline/bin/python scripts/66_gripper.py close    # close on the stem, then leave it closed
    .venvs/pipeline/bin/python scripts/66_gripper.py open     # deflate, pump off, open
    .venvs/pipeline/bin/python scripts/66_gripper.py status   # pressure for 2 s (does not move the fingers)

`close` leaves the gripper closed and the firmware keeps the brake pressurised (~100 kPa, the pump runs now and
then: that is the noise) until `open`. The port is closed without dropping DTR, so the Arduino is not reset
(a reset would open the gripper). Do not run two of these at the same time.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import termios
import time

from _common import base_parser, config_from_args


def detach(g) -> None:
    """Stop reading and close the port, keeping DTR as it is (no Arduino reset): the gripper stays as it is."""
    ser = g.dev.ser
    g.dev.running = False
    time.sleep(0.2)
    attrs = termios.tcgetattr(ser.fd)
    attrs[2] &= ~termios.HUPCL
    termios.tcsetattr(ser.fd, termios.TCSANOW, attrs)
    ser.close()


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("command", choices=["close", "open", "status"])
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.gripper import make_gripper

    g = make_gripper(cfg, sim=False, step_mode=False)
    if args.command == "close":
        r = g.grasp(0.003)
        print(f"closed: {r.reason}")
        time.sleep(1.0)
        print(f"brake pressure {g.pressure_kpa() or 0:.1f} kPa (the firmware holds it until 'open')")
        detach(g)
    elif args.command == "open":
        g.open()
        time.sleep(0.3)
        print(f"open, pressure {g.pressure_kpa() or 0:.1f} kPa, pump off")
        detach(g)
    else:
        g.dev._start_stream()
        t_end = time.time() + 2.0
        while time.time() < t_end:
            time.sleep(0.5)
            print(f"pressure {g.pressure_kpa() or 0:.1f} kPa")
        g.dev._stop_stream()
        detach(g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
