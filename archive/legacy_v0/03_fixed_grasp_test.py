from __future__ import annotations

import argparse

from src.config import load_config
from src.localization import RobotTarget
from src.robot_control import FrankaRobot, confirm_step, run_grasp_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-position fake stem grasp test.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--execute", action="store_true", help="Actually move Franka")
    parser.add_argument("--x", type=float, help="Robot base x in meters")
    parser.add_argument("--y", type=float, help="Robot base y in meters")
    parser.add_argument("--z", type=float, help="Grasp z in meters")
    parser.add_argument("--host", help="Override robot FCI IP")
    parser.add_argument("--pregrasp-only", action="store_true", help="Only move to pregrasp pose, preserving current tool orientation")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.host:
        config["robot"]["host"] = args.host

    loc = config["localization"]
    center_x, center_y = loc["workspace_center_robot_xy_m"]
    target = RobotTarget(
        xy_m=(
            float(args.x if args.x is not None else center_x),
            float(args.y if args.y is not None else center_y),
        ),
        z_m=float(args.z if args.z is not None else loc["fixed_grasp_z_m"]),
        yaw_deg=None,
        method="fixed_cli_or_config",
    )

    dry_run = not args.execute
    robot = None
    if not dry_run:
        print("Connecting to Franka...")
        robot = FrankaRobot(config)
        print(robot.current_state_text())

    if robot is None:
        class DryRobot:
            pass
        robot = DryRobot()

    try:
        if args.pregrasp_only:
            loc = config["localization"]
            pregrasp = [
                target.xyz_m[0],
                target.xyz_m[1],
                target.xyz_m[2] + float(loc["pregrasp_z_offset_m"]),
            ]
            print("\nPregrasp-only plan:")
            print(f"  pregrasp: {pregrasp}")
            print("  orientation: preserve current end-effector orientation")
            if dry_run:
                print("\nDRY RUN: no robot motion executed.")
            else:
                current_pose = robot.robot.current_pose
                quat = current_pose.end_effector_pose.quaternion.tolist()
                confirm_step(bool(config["safety"]["require_enter_before_each_motion"]), "Move to fixed pregrasp pose?")
                robot.move_to(pregrasp, quaternion=quat, relative_dynamics_factor=0.02)
                print("Pregrasp move finished.")
        else:
            run_grasp_sequence(robot, target, config, dry_run=dry_run)
    finally:
        if hasattr(robot, "shutdown"):
            robot.shutdown()


if __name__ == "__main__":
    main()
