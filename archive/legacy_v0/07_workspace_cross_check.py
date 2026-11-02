from __future__ import annotations

import argparse
import time

from src.config import load_config
from src.robot_control import FrankaRobot


def pose_xyz_text(robot: FrankaRobot) -> str:
    pose = robot.robot.current_pose.end_effector_pose
    t = pose.translation
    return f"[{float(t[0]):.6f}, {float(t[1]):.6f}, {float(t[2]):.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Workspace cross motion check at fixed pregrasp height. No gripper action.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--host", help="Override robot FCI IP")
    parser.add_argument("--execute", action="store_true", help="Actually move Franka")
    parser.add_argument("--x", type=float, default=0.4479, help="Center x in robot base meters")
    parser.add_argument("--y", type=float, default=0.0518, help="Center y in robot base meters")
    parser.add_argument("--z", type=float, default=0.3470, help="Center z in robot base meters")
    parser.add_argument("--xy-mm", type=float, default=40.0, help="Planar X/Y offset in mm")
    parser.add_argument("--down-mm", type=float, default=20.0, help="Downward Z approach in mm")
    parser.add_argument("--dynamics", type=float, default=0.08, help="Relative dynamics factor")
    parser.add_argument("--settle-s", type=float, default=0.1, help="Pause between moves")
    args = parser.parse_args()

    center = [float(args.x), float(args.y), float(args.z)]
    dxy = float(args.xy_mm) / 1000.0
    dz = float(args.down_mm) / 1000.0
    waypoints = [
        ("center", center),
        ("x_plus", [center[0] + dxy, center[1], center[2]]),
        ("center", center),
        ("x_minus", [center[0] - dxy, center[1], center[2]]),
        ("center", center),
        ("y_plus", [center[0], center[1] + dxy, center[2]]),
        ("center", center),
        ("y_minus", [center[0], center[1] - dxy, center[2]]),
        ("center", center),
        ("z_down", [center[0], center[1], center[2] - dz]),
        ("return_center", center),
    ]

    print("Workspace cross-check plan:")
    for idx, (name, xyz) in enumerate(waypoints, start=1):
        print(f"  {idx:02d}. {name}: {xyz}")
    print("  gripper: no action")
    print(f"  dynamics: {float(args.dynamics):.3f}")

    if not args.execute:
        print("\nDRY RUN: no robot motion executed.")
        return

    print("\nConnecting to Franka...")
    robot = FrankaRobot(load_config(args.config))
    if args.host:
        robot.host = args.host
    try:
        print(robot.current_state_text())
        quat = robot.robot.current_pose.end_effector_pose.quaternion.tolist()
        for idx, (name, xyz) in enumerate(waypoints, start=1):
            print(f"\nMove {idx}/{len(waypoints)}: {name} -> {xyz}")
            robot.move_to(xyz, quaternion=quat, relative_dynamics_factor=float(args.dynamics))
            time.sleep(float(args.settle_s))
            print(f"  reached: {pose_xyz_text(robot)}")
        print("Workspace cross-check finished.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
