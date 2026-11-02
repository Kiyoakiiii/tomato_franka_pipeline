#!/usr/bin/env python
"""Read-only environment check: packages, account permissions, devices, robot network.

Does not move anything. ``--robot`` additionally opens a libfranka connection to read the
state (no error recovery, no motion); skip it if another user is driving the robot.
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import grp
import importlib
import os
import pwd
import resource
import socket
import subprocess
from pathlib import Path

from _common import base_parser, config_from_args


def line(ok: bool | None, text: str) -> None:
    tag = {True: "[ OK ]", False: "[FAIL]", None: "[WARN]"}[ok]
    print(f"{tag} {text}")


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--robot", action="store_true", help="also connect to the robot and read its state")
    args = ap.parse_args()
    cfg = config_from_args(args)

    print("== Python packages")
    for mod in ["numpy", "scipy", "cv2", "skimage", "torch", "ultralytics", "franky", "serial", "open3d"]:
        try:
            m = importlib.import_module(mod)
            line(True, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            line(False if mod in {"numpy", "scipy", "cv2", "torch", "ultralytics", "franky"} else None, f"{mod}: {exc}")
    try:
        import torch

        line(torch.cuda.is_available() or None, f"CUDA available: {torch.cuda.is_available()}")
    except Exception:
        pass
    try:
        import pyzed.sl as sl

        line(True, f"pyzed (ZED SDK {sl.Camera.get_sdk_version()})")
    except Exception as exc:
        line(False, f"pyzed: {exc}  -> 需要加入 zed 组并安装 pyzed（见 docs/pipeline_quickstart_CN.md）")

    print("\n== Account permissions")
    user = pwd.getpwuid(os.getuid()).pw_name
    groups = {g.gr_name for g in grp.getgrall() if user in g.gr_mem} | {grp.getgrgid(os.getgid()).gr_name}
    for g, why in [("zed", "读取 /usr/local/zed (ZED SDK)"), ("realtime", "libfranka 实时调度"), ("dialout", "串口夹爪"), ("video", "USB 相机")]:
        line(g in groups or (None if g in {"dialout", "video"} else False), f"group {g}: {'yes' if g in groups else 'no'} ({why})")
    rt = resource.getrlimit(resource.RLIMIT_RTPRIO)
    line(rt[0] > 0 or False, f"RLIMIT_RTPRIO soft={rt[0]} hard={rt[1]} (需要 >0，否则 robot.realtime 只能设 ignore)")
    line(os.access("/usr/local/zed", os.R_OK | os.X_OK), "access /usr/local/zed")
    kernel = os.uname().release
    line("realtime" in kernel or "rt" in kernel or None, f"kernel {kernel}")

    print("\n== Devices")
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5).stdout
        zed = [l for l in out.splitlines() if "2b03" in l]
        line(bool(zed), "ZED USB device: " + (zed[0] if zed else "not found (相机未接或未上电)"))
    except Exception as exc:
        line(None, f"lsusb: {exc}")
    tty = sorted(str(p) for p in Path("/dev").glob("tty[UA]*"))
    line(bool(tty) or None, f"serial ports: {tty or 'none'}")

    print("\n== Robot network")
    host = str(cfg["robot"]["host"])
    for port, what in [(443, "Desk (HTTPS)"), (1337, "FCI command port")]:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                line(True, f"{host}:{port} reachable ({what})")
        except Exception as exc:
            line(False, f"{host}:{port} {what}: {exc}")

    print("\n== Project files")
    from seedling.config import project_path

    line(project_path(cfg["perception"]["model"]).exists(), f"model {cfg['perception']['model']}")
    line(project_path(cfg["camera"]["T_base_cam_file"]).exists() or None, f"camera extrinsics {cfg['camera']['T_base_cam_file']} (未标定前不存在属正常)")

    if args.robot:
        print("\n== Robot state (read only)")
        try:
            import franky

            rtc = franky.RealtimeConfig.Enforce if cfg["robot"]["realtime"] == "enforce" else franky.RealtimeConfig.Ignore
            r = franky.Robot(host, realtime_config=rtc)
            st = r.state
            line(True, f"connected; mode={st.robot_mode}; has_errors={r.has_errors}")
            print("   q =", [round(float(v), 3) for v in r.current_joint_positions])
            print("   EE xyz =", [round(float(v), 4) for v in r.current_pose.end_effector_pose.translation])
            print("   F_ext (base) =", [round(float(v), 2) for v in st.O_F_ext_hat_K[:3]])
        except Exception as exc:
            line(False, f"robot connection: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
