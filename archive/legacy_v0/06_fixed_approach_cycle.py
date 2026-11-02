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
    parser = argparse.ArgumentParser(description="Fixed pregrasp approach cycle. No gripper action.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--host", help="Override robot FCI IP")
    parser.add_argument("--execute", action="store_true", help="Actually move Franka")
    parser.add_argument("--x", type=float, default=0.4479, help="Pregrasp x in robot base meters")
    parser.add_argument("--y", type=float, default=0.0518, help="Pregrasp y in robot base meters")
    parser.add_argument("--z", type=float, default=0.3470, help="Pregrasp z in robot base meters")
    parser.add_argument("--approach-down-mm", type=float, default=20.0, help="Approach depth below pregrasp")
    parser.add_argument("--dynamics", type=float, default=0.06, help="Relative dynamics factor for this cycle")
    parser.add_argument("--settle-s", type=float, default=0.2, help="Pause between moves")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config["robot"]["host"] = args.host

    pregrasp = [float(args.x), float(args.y), float(args.z)]
    approach = [float(args.x), float(args.y), float(args.z) - float(args.approach_down_mm) / 1000.0]

    print("Fixed approach cycle plan:")
    print(f"  pregrasp: {pregrasp}")
    print(f"  approach: {approach}")
    print(f"  return:   {pregrasp}")
    print("  gripper:  no action")
    print(f"  dynamics: {float(args.dynamics):.3f}")

    if not args.execute:
        print("\nDRY RUN: no robot motion executed.")
        return

    print("\nConnecting to Franka...")
    robot = FrankaRobot(config)
    try:
        print(robot.current_state_text())
        quat = robot.robot.current_pose.end_effector_pose.quaternion.tolist()

        print("\nMove 1/3: pregrasp")
        robot.move_to(pregrasp, quaternion=quat, relative_dynamics_factor=float(args.dynamics))
        time.sleep(float(args.settle_s))
        print(f"  reached: {pose_xyz_text(robot)}")

        print("Move 2/3: approach")
        robot.move_to(approach, quaternion=quat, relative_dynamics_factor=float(args.dynamics))
        time.sleep(float(args.settle_s))
        print(f"  reached: {pose_xyz_text(robot)}")

        print("Move 3/3: return pregrasp")
        robot.move_to(pregrasp, quaternion=quat, relative_dynamics_factor=float(args.dynamics))
        time.sleep(float(args.settle_s))
        print(f"  reached: {pose_xyz_text(robot)}")
        print("Fixed approach cycle finished.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
