#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone interactive controller for the pin-brake gripper.

Only needs pyserial. The gripper Arduino is found automatically: every USB
serial port is opened in turn and the one that prints the firmware banner
("Gripper ready ...") after its reset is used, so the ttyUSB number does not
matter. Use --port to skip the search.

Usage:
  python example_pinBrakeGripper.py
  python example_pinBrakeGripper.py --port /dev/ttyUSB1

Commands (enter interactively):
  init                Vent, pump off, default speed, gripper fully open
  open                Deflate, shut down the pneumatics and open the gripper
  p <kPa>             Gripper stop pressure for 'close' (0 = no pressure)
  close <width_mm>    Close on an object of the given width
  saveLog             Toggle log saving (default off)
  help                Show this help
  quit                Open the gripper, shut down the pneumatics and exit

Examples:
  init
  p 5
  saveLog
  close 30
  open
  quit

While the gripper is closed with pressure enabled, the pressure is printed every
0.5 s; 'open' stops the printing and deflates until the pressure is gone. With saveLog on, each grasp (close ... open)
is saved to data/pinBrakeGripper_<timestamp>.json with the command timings,
the GS / PR events and the time-pressure data.
"""

import argparse
import json
import math
import os
import queue
import readline     # line editing for input(); lets background prints redraw the prompt
import sys
import threading
import time

import serial
from serial.tools import list_ports


# ------------------------------------------------------------
# Serial settings
# ------------------------------------------------------------

BAUDRATE = 115200
FIRMWARE_BANNER = b"Gripper ready"
BANNER_TIMEOUT_S = 3.0      # Arduino reset + bootloader + banner
CH340_VID = 0x1A86          # USB-serial chip on the gripper Arduino


# ------------------------------------------------------------
# Gripper geometry
# ------------------------------------------------------------

LINK_LENGTH_MM = 43.0
OPEN_WIDTH_MM = 40.0
CONTACT_BUFFER_MM = 5.0     # pin travel per side before "contact"
SLIDE_MAX_MM = 20.0         # pin travel per side between contact and full close

MAX_OBJECT_WIDTH_MM = OPEN_WIDTH_MM


# ------------------------------------------------------------
# Grasp settings
# ------------------------------------------------------------

SPEED_FAST_DPS = 70         # until predicted contact
SPEED_SLOW_DPS = 20         # after predicted contact

# Delays after the predicted contact time (theta_contact / SPEED_FAST_DPS).
# These reproduce the effective timing of exp_berrySorting_pinBrake.py.
SLOW_DOWN_DELAY_S = 0.11
PRESSURIZE_DELAY_S = 0.16

DEFAULT_PRESSURE_KPA = 5.0
MAX_PRESSURE_KPA = 100.0    # firmware holds 100 kPa once the gripper stops

GRIP_STOP_MARGIN_S = 1.0


# ------------------------------------------------------------
# Pressure readout and logging
# ------------------------------------------------------------

PRESSURE_PRINT_INTERVAL_S = 0.5

# 'open' keeps deflating until the pressure is below DEFLATE_DONE_KPA.
DEFLATE_DONE_KPA = 2.0
DEFLATE_TIMEOUT_S = 3.0
PSI_TO_KPA = 6.89476
FRAME_START, FRAME_END, FRAME_LENGTH = 0x02, 0x03, 6
LOG_DIR = "data"


# ------------------------------------------------------------
# Console
# ------------------------------------------------------------

PROMPT = "gripper> "
at_prompt = threading.Event()


def read_command_line():
    at_prompt.set()
    try:
        return input(PROMPT)
    finally:
        at_prompt.clear()


def print_from_thread(message):
    """Print without breaking up a command that is being typed at the prompt."""
    if at_prompt.is_set() and sys.stdin.isatty():
        typed = readline.get_line_buffer()
        sys.stdout.write(f"\r\033[K{message}\n{PROMPT}{typed}")
        sys.stdout.flush()
    else:
        print(message)


# ------------------------------------------------------------
# Port detection
# ------------------------------------------------------------

def usb_serial_ports():
    """USB serial devices, CH340 adapters first."""
    ports = [p for p in list_ports.comports() if p.vid is not None]
    ports.sort(key=lambda p: (p.vid != CH340_VID, p.device))
    return [p.device for p in ports]


def wait_for_banner(ser, timeout=BANNER_TIMEOUT_S):
    """Opening the port resets the Arduino; return True once it prints its banner."""
    received = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        received += ser.read(ser.in_waiting or 1)
        if FIRMWARE_BANNER in received:
            return True
    return False


def open_port(device):
    return serial.Serial(device, BAUDRATE, timeout=0.05, write_timeout=1.0, exclusive=True)


def connect(port=None):
    """Return (serial port, banner seen), or (None, False) if no gripper was found."""
    if port is not None:
        ser = open_port(port)
        banner_seen = wait_for_banner(ser)
        if not banner_seen:
            print(f"Warning: no firmware banner from {port}; using it anyway.")
        return ser, banner_seen

    devices = usb_serial_ports()
    if not devices:
        print("Error: no USB serial devices found. Check the cable and power.")
        return None, False

    for device in devices:
        print(f"Checking {device} ...")
        try:
            ser = open_port(device)
        except serial.SerialException as e:
            print(f"  could not open: {e}")
            continue
        if wait_for_banner(ser):
            return ser, True
        ser.close()

    print("Error: no gripper found on " + ", ".join(devices))
    print("If the port is known, pass it with --port.")
    return None, False


# ------------------------------------------------------------
# Gripper
# ------------------------------------------------------------

def calculate_gripper_angles(width_mm):
    """Servo angles (deg) at predicted contact and at full close for an object width."""
    r = LINK_LENGTH_MM

    cos_contact = (2 * r - OPEN_WIDTH_MM + width_mm - 2 * CONTACT_BUFFER_MM) / (2 * r)
    cos_contact = max(-1.0, min(1.0, cos_contact))

    cos_max = cos_contact - SLIDE_MAX_MM / r
    cos_max = max(-1.0, min(1.0, cos_max))

    theta_contact = int(round(math.degrees(math.acos(cos_contact))))
    theta_max = int(round(math.degrees(math.acos(cos_max))))
    return theta_contact, theta_max


def sleep_until(deadline):
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


class PinBrakeGripper:
    def __init__(self, ser, is_open):
        self.ser = ser
        self.pressure_kpa = DEFAULT_PRESSURE_KPA
        self.is_open = is_open          # 'close' timing assumes a start from 0 deg
        self.open_done_time = 0.0       # when the last opening motion ends
        self.last_close_angle = 180
        self.save_log = False
        self.recording = None           # data of the current grasp (close ... open)
        self.print_pressure = False
        self.last_print_time = 0.0
        self.streaming = False
        self.latest_pressure = (0.0, 0.0)   # (wall time, kPa)
        self.events = queue.Queue()     # ("GS" | "PR", monotonic time)
        self.running = True
        self.reader = threading.Thread(target=self._read_serial, daemon=True)
        self.reader.start()

    def _read_serial(self):
        """Background thread: parse pressure frames and GS / PR event lines."""
        received = bytearray()
        while self.running:
            try:
                received += self.ser.read(self.ser.in_waiting or 1)
            except (serial.SerialException, OSError, TypeError) as e:
                if self.running:
                    print_from_thread(f"Serial error: {e}")
                    self.running = False
                break

            now = time.time()
            while received:
                # Pressure frame: STX, ADC (2 bytes), psi * 100 (2 bytes), ETX
                if received[0] == FRAME_START:
                    if len(received) < FRAME_LENGTH:
                        break
                    if received[FRAME_LENGTH - 1] == FRAME_END:
                        psi = int.from_bytes(received[3:5], "big") / 100.0
                        self._on_pressure(now, psi * PSI_TO_KPA)
                        del received[:FRAME_LENGTH]
                    else:
                        del received[:1]
                    continue

                line_end = received.find(b"\n")
                if line_end < 0:
                    break
                line = received[:line_end].strip().decode(errors="replace")
                del received[:line_end + 1]
                if line in ("GS", "PR"):
                    self._on_event(now, line)

    def _on_pressure(self, t, kpa):
        self.latest_pressure = (t, kpa)
        recording = self.recording
        if recording is not None:
            recording["pressure"].append((t, kpa))
        if self.print_pressure and t - self.last_print_time >= PRESSURE_PRINT_INTERVAL_S:
            self.last_print_time = t
            print_from_thread(f"  pressure: {kpa:.1f} kPa")

    def _on_event(self, t, name):
        recording = self.recording
        if recording is not None:
            recording["events"].append((t, name))
        self.events.put((name, time.monotonic()))

    def _send(self, command):
        print(f"  -> {command}")
        if self.recording is not None:
            self.recording["commands"].append((time.time(), command))
        self.ser.write((command + "\n").encode())
        self.ser.flush()

    def _clear_events(self):
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                return

    def _wait_for_grip_stop(self, deadline):
        """Return the time of the next GS, or None if the deadline passes first."""
        while self.running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                name, t_event = self.events.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            if name == "PR":
                print("  pressure reached")
            else:
                return t_event
        return None

    def _start_stream(self):
        if not self.streaming:
            self._send("s")
            self.streaming = True

    def _stop_stream(self):
        self._send("q")
        self.streaming = False

    def _deflate(self):
        """Run the deflate pump until the measured pressure is gone."""
        t_start = time.time()
        self._send("D")
        deadline = time.monotonic() + DEFLATE_TIMEOUT_S
        while self.running and time.monotonic() < deadline:
            t_sample, kpa = self.latest_pressure
            if t_sample > t_start and kpa < DEFLATE_DONE_KPA:
                print(f"  deflated to {kpa:.1f} kPa in {t_sample - t_start:.2f} s")
                return
            time.sleep(0.005)
        t_sample, kpa = self.latest_pressure
        if t_sample > t_start:
            print(f"Warning: pressure still {kpa:.1f} kPa after {DEFLATE_TIMEOUT_S:g} s of deflating.")
        else:
            print("Warning: no pressure data received while deflating.")

    def init(self):
        self.open()

    def open(self):
        """Open the gripper, shut down the pneumatics and end the current grasp."""
        self.print_pressure = False
        self._start_stream()
        self._send(f"V{SPEED_FAST_DPS}")
        self._deflate()
        self._send("OFF")
        time.sleep(0.1)
        self._send("G0")
        self._stop_stream()
        self.open_done_time = time.monotonic() + self.last_close_angle / SPEED_FAST_DPS + 0.2
        self.is_open = True

        recording, self.recording = self.recording, None
        if recording is not None and self.save_log:
            self._save_log(recording)

    def toggle_save_log(self):
        self.save_log = not self.save_log
        if self.save_log:
            print(f"Log saving ON: each grasp is saved to {LOG_DIR}/ on 'open'.")
        else:
            print("Log saving OFF")

    def _save_log(self, recording):
        t0 = recording["start_wall_time_s"]
        recording["commands"] = [
            {"t_s": round(t - t0, 4), "command": c} for t, c in recording["commands"]
        ]
        recording["events"] = [
            {"t_s": round(t - t0, 4), "event": e} for t, e in recording["events"]
        ]
        recording["pressure"] = {
            "t_s": [round(t - t0, 4) for t, _ in recording["pressure"]],
            "kpa": [round(kpa, 2) for _, kpa in recording["pressure"]],
        }

        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t0))
        path = os.path.join(LOG_DIR, f"pinBrakeGripper_{stamp}.json")
        with open(path, "w") as f:
            json.dump(recording, f)
        print(f"Saved log to {path} ({len(recording['pressure']['t_s'])} pressure samples)")

    def set_pressure(self, kpa):
        self.pressure_kpa = kpa
        if kpa > 0:
            print(f"Gripper stop pressure: {kpa:g} kPa")
        else:
            print("Pressure disabled: 'close' will only move the gripper.")

    def close(self, width_mm):
        """Close on an object: fast to predicted contact, then slow, pressurizing if enabled."""
        if not self.is_open:
            print("Error: gripper is not open. Run 'open' first.")
            return
        sleep_until(self.open_done_time)

        theta_contact, theta_max = calculate_gripper_angles(width_mm)
        t_contact = theta_contact / SPEED_FAST_DPS
        t_slow = t_contact + SLOW_DOWN_DELAY_S
        t_full_close = t_slow + max(0.0, theta_max - SPEED_FAST_DPS * t_slow) / SPEED_SLOW_DPS
        print(f"Width {width_mm:g} mm: contact at {theta_contact} deg, full close at {theta_max} deg")

        self.recording = {
            "start_wall_time_s": time.time(),
            "width_mm": width_mm,
            "stop_pressure_kpa": self.pressure_kpa,
            "theta_contact_deg": theta_contact,
            "theta_max_deg": theta_max,
            "commands": [],
            "events": [],
            "pressure": [],
        }
        self.last_print_time = time.time()
        self.print_pressure = self.pressure_kpa > 0

        self._start_stream()
        self._send("P0")
        self._send(f"V{SPEED_FAST_DPS}")
        time.sleep(0.05)
        self._clear_events()

        t_start = time.monotonic()
        self._send(f"G{theta_max}")
        self.is_open = False
        self.last_close_angle = theta_max

        sleep_until(t_start + t_slow)
        self._send(f"V{SPEED_SLOW_DPS}")

        if self.pressure_kpa > 0:
            sleep_until(t_start + t_contact + PRESSURIZE_DELAY_S)
            self._send(f"P{self.pressure_kpa:g}")

        t_stop = self._wait_for_grip_stop(t_start + t_full_close + GRIP_STOP_MARGIN_S)
        if t_stop is None:
            print("Warning: no gripper stop (GS) received.")
        elif self.pressure_kpa > 0 and t_stop - t_start < t_full_close - 0.1:
            print(f"Gripper stopped at pressure after {t_stop - t_start:.2f} s.")
        else:
            print(f"Gripper reached full close after {t_stop - t_start:.2f} s.")

    def shutdown(self):
        try:
            if self.running:
                self.open()
        except (serial.SerialException, OSError) as e:
            print(f"Could not open gripper during shutdown: {e}")
        self.running = False
        self.reader.join(timeout=0.5)
        self.ser.close()


# ------------------------------------------------------------
# Interactive command interface
# ------------------------------------------------------------

def print_usage(gripper):
    print("\n" + "=" * 60)
    print("           Pin-Brake Gripper Interactive Controller")
    print("=" * 60)
    print("Available commands:")
    print("  init                Vent, pump off, default speed, gripper fully open")
    print("  open                Deflate, shut down the pneumatics and open the gripper")
    print(f"  p <kPa>             Gripper stop pressure, 0-{MAX_PRESSURE_KPA:g} (0 = no pressure)")
    print(f"  close <width_mm>    Close on an object, width 0-{MAX_OBJECT_WIDTH_MM:g} mm")
    print("  saveLog             Toggle log saving (default off)")
    print("  help                Show this help")
    print("  quit                Open the gripper, shut down the pneumatics and exit")
    print("\nExamples:")
    print("  init")
    print("  p 5")
    print("  close 30")
    print("  open")
    print(f"\nCurrent stop pressure: {gripper.pressure_kpa:g} kPa")
    print(f"Log saving: {'ON' if gripper.save_log else 'OFF'}")
    print("=" * 60)


def parse_command(command_line):
    """Return (command, value); command is None if the input is invalid."""
    parts = command_line.split()
    command = parts[0].lower()

    if command in ("init", "open", "savelog", "help", "quit"):
        if len(parts) != 1:
            print(f"Error: '{parts[0]}' takes no arguments")
            return None, None
        return command, None

    if command in ("p", "close"):
        unit, maximum = {
            "p": ("kPa", MAX_PRESSURE_KPA),
            "close": ("mm", MAX_OBJECT_WIDTH_MM),
        }[command]
        if len(parts) != 2:
            print(f"Error: '{command}' requires 1 argument")
            return None, None
        try:
            value = float(parts[1])
        except ValueError:
            print(f"Error: argument must be a number for '{command}'")
            return None, None
        if not 0 <= value <= maximum:      # also rejects nan
            print(f"Error: '{command}' value must be between 0 and {maximum:g} {unit}")
            return None, None
        return command, value

    print(f"Error: unknown command '{parts[0]}' (type 'help')")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Interactive pin-brake gripper controller.")
    parser.add_argument("--port", help="Serial port of the gripper (default: auto-detect).")
    args = parser.parse_args()

    print("Starting pin-brake gripper controller...")
    try:
        ser, banner_seen = connect(args.port)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return 1
    if ser is None:
        return 1
    print(f"Gripper connected on {ser.port}")

    # The firmware starts with the gripper open, so a banner means it is open now.
    gripper = PinBrakeGripper(ser, is_open=banner_seen)
    print_usage(gripper)

    try:
        while gripper.running:
            try:
                command_line = read_command_line().strip()
            except EOFError:
                print("\nReceived EOF. Exiting...")
                break
            if not command_line:
                continue

            command, value = parse_command(command_line)
            if command is None:
                continue

            if command == "quit":
                print("Exiting...")
                break
            elif command == "help":
                print_usage(gripper)
            elif command == "init":
                gripper.init()
            elif command == "open":
                gripper.open()
            elif command == "p":
                gripper.set_pressure(value)
            elif command == "savelog":
                gripper.toggle_save_log()
            elif command == "close":
                gripper.close(value)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    finally:
        print("\nShutting down...")
        gripper.shutdown()
        print("Gripper controller shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
