"""Gripper backends: Franka Hand (via franky), lab pin-brake gripper, serial (Arduino-style), none, and sim."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class GraspResult:
    command_ok: bool
    width_m: float | None
    holding: bool | None  # None = cannot tell from gripper state alone
    reason: str = ""


class GripperBase:
    def __init__(self, cfg: dict, confirm_fn=None, step_mode: bool = False):
        self.g = cfg["gripper"]
        self.step_mode = step_mode
        self.confirm_fn = confirm_fn

    def _confirm(self, text: str) -> None:
        if self.step_mode and self.confirm_fn:
            self.confirm_fn(text)

    def width(self) -> float | None:
        return None

    def open(self, width_m: float | None = None) -> bool:
        raise NotImplementedError

    def grasp(self, width_m: float, force_n: float | None = None) -> GraspResult:
        raise NotImplementedError

    def release(self) -> bool:
        return self.open()

    def holding_from_width(self, width: float | None) -> bool | None:
        if width is None:
            return None
        return width > float(self.g["empty_close_width_m"]) + float(self.g["holding_margin_m"])

    def close(self) -> None:
        pass


class FrankaHand(GripperBase):
    def __init__(self, cfg, confirm_fn=None, step_mode=False):
        super().__init__(cfg, confirm_fn, step_mode)
        import franky

        self.fr = franky
        self.hand = franky.Gripper(str(cfg["robot"]["host"]))

    def width(self):
        return float(self.hand.width)

    def homing(self) -> bool:
        self._confirm("gripper homing（手指会完全张开再闭合）")
        return bool(self.hand.homing())

    def open(self, width_m=None):
        w = float(width_m if width_m is not None else self.g["open_width_m"])
        self._confirm(f"gripper open -> {w*1000:.1f} mm")
        return bool(self.hand.move(w, float(self.g["grasp_speed_m_s"])))

    def grasp(self, width_m, force_n=None):
        f = float(force_n if force_n is not None else self.g["grasp_force_n"])
        self._confirm(f"gripper grasp width={width_m*1000:.1f} mm force={f:.1f} N")
        try:
            ok = bool(self.hand.grasp(float(width_m), float(self.g["grasp_speed_m_s"]), f,
                                      float(self.g["epsilon_inner_m"]), float(self.g["epsilon_outer_m"])))
        except self.fr.GripperException as exc:
            return GraspResult(False, self.width(), None, f"gripper_exception: {exc}")
        w = self.width()
        return GraspResult(ok, w, self.holding_from_width(w) and ok, "" if ok else "grasp_width_out_of_epsilon")


class SerialGripper(GripperBase):
    """Open/close-only gripper on a serial line. Cannot report width, so holding is unknown."""

    def __init__(self, cfg, confirm_fn=None, step_mode=False):
        super().__init__(cfg, confirm_fn, step_mode)
        import serial

        s = self.g["serial"]
        self.s = s
        self.port = serial.Serial(s["port"], int(s["baudrate"]), timeout=1.0)
        time.sleep(2.0)  # many Arduino boards reset on connect

    def _send(self, cmd: str) -> bool:
        self.port.write((cmd.strip() + "\n").encode())
        self.port.flush()
        time.sleep(float(self.s["settle_s"]))
        return True

    def open(self, width_m=None):
        self._confirm("serial gripper open")
        return self._send(self.s["open_command"])

    def grasp(self, width_m, force_n=None):
        self._confirm("serial gripper close")
        return GraspResult(self._send(self.s["close_command"]), None, None, "no_width_feedback")

    def close(self):
        self.port.close()


class PinBrakeGripper(GripperBase):
    """Lab pin-brake gripper (Arduino on USB serial), driven through the lab's example_pinBrakeGripper.py.

    close: fast to the predicted contact, then slow; the host enables pressure and the firmware stops the servo
    (GS) once gripper.pin_brake.pressure_kpa is reached. Tested 2026-10-15: an empty close also stops ~20 ms after
    pressurizing (~1.09 s, before full close at ~1.99 s), so the stop says nothing about an object; there is no
    width readout either -> holding is always None (unknown). Reaching full close with pressure on means the
    pressure never built up (pneumatics problem).
    Opening the port resets the Arduino, which opens the gripper: open it once per process, never while holding.
    """

    def __init__(self, cfg, confirm_fn=None, step_mode=False, ser=None):
        super().__init__(cfg, confirm_fn, step_mode)
        import example_pinBrakeGripper as pb

        self.pb = pb
        self.s = self.g["pin_brake"]
        banner = True
        if ser is None:
            ser, banner = pb.connect(self.s.get("port"))
            if ser is None:
                raise RuntimeError("pin-brake gripper not found (USB cable / power / permission on /dev/ttyUSB*)")
        self.dev = _pin_brake_device_class()(ser, is_open=banner)
        self.dev.set_pressure(float(self.s["pressure_kpa"]))
        self.last = {}

    def pressure_kpa(self) -> float | None:
        """Latest pad pressure (streamed while closed), None if no sample yet."""
        t, kpa = self.dev.latest_pressure
        return kpa if t > 0 else None

    def open(self, width_m=None):
        self._confirm("pin-brake gripper open (deflate, pneumatics off, servo to 0)")
        self.dev.open()
        self.pb.sleep_until(self.dev.open_done_time)  # G0 returns at once; wait until the pins are clear
        return True

    def grasp(self, width_m, force_n=None):
        # the tested setting closes on a fixed nominal width; the estimated stem diameter is not used
        w = float(self.s["close_width_mm"])
        self._confirm(f"pin-brake gripper close {w:g} mm, stop at {self.dev.pressure_kpa:g} kPa")
        if not self.dev.is_open:
            self.dev.open()
        self.last = r = self.dev.close_and_report(w)
        r["pressure_kpa_after"] = self.pressure_kpa()
        if r["stop"] == "pressure":
            return GraspResult(True, None, None, f"stopped at pressure after {r['t_stop_s']:.2f} s (object not detectable)")
        if r["stop"] == "full_close" and self.dev.pressure_kpa > 0:
            return GraspResult(False, None, None, "reached full close: stop pressure never reached (check air supply)")
        if r["stop"] == "full_close":
            return GraspResult(True, None, None, "reached full close (pressure disabled)")
        return GraspResult(False, None, None, "no gripper stop (GS) received")

    def close(self):
        self.dev.shutdown()


def _pin_brake_device_class():
    import example_pinBrakeGripper as pb

    class Device(pb.PinBrakeGripper):
        def close_and_report(self, width_mm: float) -> dict:
            """pb.PinBrakeGripper.close (same commands and timing), returning how the gripper stopped."""
            sleep_until = pb.sleep_until
            sleep_until(self.open_done_time)
            theta_contact, theta_max = pb.calculate_gripper_angles(width_mm)
            t_contact = theta_contact / pb.SPEED_FAST_DPS
            t_slow = t_contact + pb.SLOW_DOWN_DELAY_S
            t_full_close = t_slow + max(0.0, theta_max - pb.SPEED_FAST_DPS * t_slow) / pb.SPEED_SLOW_DPS
            self.recording = {"start_wall_time_s": time.time(), "width_mm": width_mm, "stop_pressure_kpa": self.pressure_kpa,
                              "theta_contact_deg": theta_contact, "theta_max_deg": theta_max,
                              "commands": [], "events": [], "pressure": []}
            self.print_pressure = False
            self._start_stream()
            self._send("P0")
            self._send(f"V{pb.SPEED_FAST_DPS}")
            time.sleep(0.05)
            self._clear_events()
            t_start = time.monotonic()
            self._send(f"G{theta_max}")
            self.is_open = False
            self.last_close_angle = theta_max
            sleep_until(t_start + t_slow)
            self._send(f"V{pb.SPEED_SLOW_DPS}")
            if self.pressure_kpa > 0:
                sleep_until(t_start + t_contact + pb.PRESSURIZE_DELAY_S)
                self._send(f"P{self.pressure_kpa:g}")
            t_stop = self._wait_for_grip_stop(t_start + t_full_close + pb.GRIP_STOP_MARGIN_S)
            out = {"theta_contact_deg": theta_contact, "theta_max_deg": theta_max, "t_full_close_s": round(t_full_close, 3)}
            if t_stop is None:
                return {**out, "stop": None, "t_stop_s": None}
            dt = t_stop - t_start
            early = self.pressure_kpa > 0 and dt < t_full_close - 0.1
            return {**out, "stop": "pressure" if early else "full_close", "t_stop_s": round(dt, 3)}

    return Device


class NoGripper(GripperBase):
    def open(self, width_m=None):
        print("[gripper] type=none: skip open")
        return True

    def grasp(self, width_m, force_n=None):
        print("[gripper] type=none: skip grasp")
        return GraspResult(True, None, None, "no_gripper")


class SimGripper(GripperBase):
    def __init__(self, cfg, world=None, confirm_fn=None, step_mode=False):
        super().__init__(cfg, confirm_fn, step_mode)
        self.world = world
        self._w = float(self.g["open_width_m"])

    def width(self):
        return self._w

    def open(self, width_m=None):
        self._w = float(width_m if width_m is not None else self.g["open_width_m"])
        if self.world:
            self.world.on_release()
        return True

    def grasp(self, width_m, force_n=None):
        held = self.world.try_grasp() if self.world else None
        empty = float(self.g["empty_close_width_m"])
        self._w = held if held else empty
        return GraspResult(True, self._w, self.holding_from_width(self._w))


def make_gripper(cfg: dict, sim: bool, step_mode: bool, confirm_fn=None, world=None) -> GripperBase:
    if sim:
        return SimGripper(cfg, world=world)
    t = cfg["gripper"]["type"]
    if t == "franka_hand":
        return FrankaHand(cfg, confirm_fn, step_mode)
    if t == "pin_brake":
        return PinBrakeGripper(cfg, confirm_fn, step_mode)
    if t == "serial":
        return SerialGripper(cfg, confirm_fn, step_mode)
    if t == "none":
        return NoGripper(cfg)
    if t == "sim":
        return SimGripper(cfg, world=world)
    raise ValueError(f"unknown gripper type {t}")
