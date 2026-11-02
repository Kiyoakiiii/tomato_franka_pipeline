from __future__ import annotations

import argparse

from src.config import load_config
from src.robot_control import FrankaRobot, confirm_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Franka state and optionally run a tiny safe relative move.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--host", help="Override robot FCI IP")
    parser.add_argument("--execute", action="store_true", help="Allow optional robot movement")
    parser.add_argument("--move-z-mm", type=float, default=0.0, help="Tiny relative z move in mm; needs --execute")
    parser.add_argument("--gripper-open", action="store_true", help="Open Franka gripper; needs config use_franka_gripper=true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config["robot"]["host"] = args.host

    print("Connecting to Franka...")
    robot = FrankaRobot(config)
    try:
        print(robot.current_state_text())

        if args.move_z_mm:
            delta = [0.0, 0.0, args.move_z_mm / 1000.0]
            print(f"\nRequested relative move: {delta} m")
            if not args.execute:
                print("DRY RUN: add --execute to actually move.")
            else:
                confirm_step(True, "Tiny relative move requested. Confirm robot workspace is clear.")
                robot.move_relative(delta, relative_dynamics_factor=0.02)
                print("Tiny move finished.")

        if args.gripper_open:
            if not args.execute:
                print("DRY RUN: add --execute to open gripper.")
            else:
                confirm_step(True, "Open gripper?")
                robot.gripper_open()
                print("Gripper command finished.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
