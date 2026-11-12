"""Pin-brake gripper fast connect (seedling/gripper.py, gripper.pin_brake.fast_connect): a fake firmware on a fake
port. No hardware, no pyserial: what is sent and how long is waited when the Arduino was / was not reset, and the
open / closed state handed from one step (process) to the next."""
from __future__ import annotations

import time
import types

from seedling import gripper as G

STX, ETX = 0x02, 0x03
FRAME = bytes([STX, 0, 0, 0, 0, ETX])


class FakePort:
    """a gripper Arduino: reset -> banner after boot_s; running -> pressure frames after 's'"""

    def __init__(self, reset: bool, boot_s: float = 0.2, answers: bool = True):
        self.t0 = time.monotonic()
        self.reset, self.boot_s, self.answers = reset, boot_s, answers
        self.out = bytearray()
        self.sent = []
        self.banner_sent = False

    def _pump(self):
        if self.reset and not self.banner_sent and time.monotonic() - self.t0 >= self.boot_s:
            self.out += b"Gripper ready\n"
            self.banner_sent = True

    @property
    def in_waiting(self):
        self._pump()
        return len(self.out)

    def read(self, n=1):
        self._pump()
        if not self.out:
            time.sleep(0.005)
            return b""
        b, self.out[:] = bytes(self.out[:n]), self.out[n:]
        return b

    def write(self, data):
        self.sent.append(data)
        if data == b"s\n" and self.answers and (not self.reset or self.banner_sent):
            self.out += FRAME

    def flush(self):
        pass


def fake_pb(port: FakePort):
    def wait_for_banner(ser, timeout=3.0):
        got = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got += ser.read(ser.in_waiting or 1)
            if b"Gripper ready" in got:
                return True
        return False

    return types.SimpleNamespace(open_port=lambda dev: port, wait_for_banner=wait_for_banner, FRAME_START=STX, FRAME_END=ETX,
                                 FRAME_LENGTH=6, FIRMWARE_BANNER=b"Gripper ready", BANNER_TIMEOUT_S=3.0)


def test_not_reset_answers_quickly():
    port = FakePort(reset=False)
    t0 = time.monotonic()
    ser, banner, how = G.quick_connect(fake_pb(port), "/dev/x")
    dt = time.monotonic() - t0
    assert ser is port and banner is False and "no reset" in how
    assert dt < 0.5, f"{dt:.2f} s (the driver waits 3 s)"
    assert port.sent == [b"s\n", b"q\n"]  # the stream started for the probe and stopped again


def test_reset_waits_for_banner():
    port = FakePort(reset=True, boot_s=1.0)
    ser, banner, how = G.quick_connect(fake_pb(port), "/dev/x")
    assert banner is True and "reset" in how


def test_silent_port_falls_back_to_the_full_wait():
    port = FakePort(reset=False, answers=False)
    t0 = time.monotonic()
    ser, banner, _ = G.quick_connect(fake_pb(port), "/dev/x")
    assert banner is False and time.monotonic() - t0 >= 2.9  # as the driver: up to 3 s


def test_state_handed_over(tmp_path):
    p = tmp_path / "st.json"
    G.save_state(True, 0, time.monotonic() + 1.0, path=p)
    st = G.load_state(path=p)
    assert st["is_open"] and 0.5 < st["open_done_wall"] - time.time() <= 1.0
    dev = types.SimpleNamespace(is_open=False, last_close_angle=180, open_done_time=0.0)
    G.restore_state(dev, st)
    assert dev.is_open and dev.last_close_angle == 0 and dev.open_done_time - time.monotonic() <= 1.0
    G.save_state(False, 97, 0.0, path=p)
    dev = types.SimpleNamespace(is_open=True, last_close_angle=180, open_done_time=0.0)
    G.restore_state(dev, G.load_state(path=p))
    assert not dev.is_open and dev.last_close_angle == 97


def test_old_or_broken_state_ignored(tmp_path):
    p = tmp_path / "st.json"
    G.save_state(True, 0, 0.0, path=p)
    assert G.load_state(path=p, max_age_s=-1) is None
    p.write_text("{not json")
    assert G.load_state(path=p) is None
    assert G.load_state(path=tmp_path / "none.json") is None
