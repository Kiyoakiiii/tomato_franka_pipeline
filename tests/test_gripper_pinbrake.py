"""Pin-brake backend against a fake firmware: stop at pressure = ok (object unknown); full close with pressure on = failure."""
import threading
import time

import pytest

pytest.importorskip("serial")
import example_pinBrakeGripper as pb  # noqa: E402

from seedling.config import load_config  # noqa: E402
from seedling.gripper import PinBrakeGripper  # noqa: E402


class FakeFirmware:
    """Answers G<deg> with GS: shortly after pressurizing if the pneumatics work, else at full close; streams 0 kPa."""

    port = "fake"

    def __init__(self, pneumatics_ok: bool, width_mm: float):
        self.obj = pneumatics_ok
        self.contact = pb.calculate_gripper_angles(width_mm)[0]
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.streaming = False
        self.pressure_on = False
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.buf)

    def read(self, n=1):
        time.sleep(0.002)
        with self.lock:
            if self.streaming:
                self.buf += bytes([pb.FRAME_START, 0, 0, 0, 0, pb.FRAME_END])
            out, self.buf[:] = bytes(self.buf[:n]), self.buf[n:]
        return out

    def write(self, data):
        cmd = data.decode().strip()
        self.writes.append(cmd)
        if cmd == "s":
            self.streaming = True
        elif cmd == "q":
            self.streaming = False
        elif cmd.startswith("P"):
            self.pressure_on = float(cmd[1:]) > 0
        elif cmd.startswith("G") and int(cmd[1:]) > 0:
            # servo model: fast to the contact angle, slow after; a held object stops it shortly after contact
            theta_contact, theta_max = int(self.contact), int(cmd[1:])
            t_slow = theta_contact / pb.SPEED_FAST_DPS + pb.SLOW_DOWN_DELAY_S
            t_full = t_slow + max(0.0, theta_max - pb.SPEED_FAST_DPS * t_slow) / pb.SPEED_SLOW_DPS
            delay = t_slow + 0.05 if self.obj else t_full

            def answer():
                time.sleep(delay)
                if self.obj:  # the pads only build pressure once the host enables it
                    t_end = time.monotonic() + 1.0
                    while not self.pressure_on and time.monotonic() < t_end:
                        time.sleep(0.005)
                    if not self.pressure_on:
                        return
                with self.lock:
                    self.buf += b"GS\n"

            threading.Thread(target=answer, daemon=True).start()

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fast_gripper(monkeypatch):
    # twice the real speeds (a close takes ~1 s instead of ~2 s)
    monkeypatch.setattr(pb, "SPEED_FAST_DPS", 2 * pb.SPEED_FAST_DPS)
    monkeypatch.setattr(pb, "SPEED_SLOW_DPS", 2 * pb.SPEED_SLOW_DPS)
    monkeypatch.setattr(pb, "SLOW_DOWN_DELAY_S", pb.SLOW_DOWN_DELAY_S / 2)
    monkeypatch.setattr(pb, "PRESSURIZE_DELAY_S", pb.PRESSURIZE_DELAY_S / 2)
    monkeypatch.setattr(pb, "GRIP_STOP_MARGIN_S", 0.3)


@pytest.mark.parametrize("obj, ok", [(True, True), (False, False)])
def test_pin_brake_stop_reason(obj, ok):
    cfg = load_config()
    fw = FakeFirmware(obj, cfg["gripper"]["pin_brake"]["close_width_mm"])
    g = PinBrakeGripper(cfg, ser=fw)
    try:
        r = g.grasp(0.003)
        assert r.command_ok is ok and r.holding is None, r
        assert f"P{cfg['gripper']['pin_brake']['pressure_kpa']:g}" in fw.writes
        assert g.open()
        assert fw.writes[-1] == "q"  # stream stopped after opening
    finally:
        g.dev.running = False
