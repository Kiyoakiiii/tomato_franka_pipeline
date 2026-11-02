from __future__ import annotations

import math
import time

import numpy as np
from typing import Iterable

from .paths import add_lab_code_import_paths
from .localization import check_workspace, RobotTarget


def euler_xyz_deg_to_quat_xyzw(euler_deg: Iterable[float]) -> list[float]:
    rx, ry, rz = [math.radians(float(v)) for v in euler_deg]
    cx, sx = math.cos(rx / 2.0), math.sin(rx / 2.0)
    cy, sy = math.cos(ry / 2.0), math.sin(ry / 2.0)
    cz, sz = math.cos(rz / 2.0), math.sin(rz / 2.0)

    # R = Rz * Ry * Rx, equivalent to scipy Rotation.from_euler("xyz", ...)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    return [qx, qy, qz, qw]


class RobotNotConnected(RuntimeError):
    pass


class FrankaRobot:
    def __init__(self, config: dict, connect_gripper: bool | None = None) -> None:
        add_lab_code_import_paths()
        try:
            from franky import Affine, CartesianMotion, Gripper, ReferenceType, Robot
        except ImportError as exc:
            raise RuntimeError(
                "Could not import franky. Run this on the lab computer, or add lab_code/franky/franky to PYTHONPATH."
            ) from exc

        self.Affine = Affine
        self.CartesianMotion = CartesianMotion
        self.ReferenceType = ReferenceType
        self.Robot = Robot
        self.Gripper = Gripper

        self.config = config
        self.robot_cfg = config["robot"]
        self.host = self.robot_cfg["host"]
        self.robot = Robot(self.host)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = float(self.robot_cfg["relative_dynamics_factor"])

        self.serial_gripper = None
        end_effector = str(self.robot_cfg.get("end_effector", "none"))
        if connect_gripper is None:
            connect_gripper = bool(self.robot_cfg["use_franka_gripper"]) or end_effector == "franka_gripper"
        self.gripper = Gripper(self.host) if connect_gripper else None
        if end_effector == "serial_gripper":
            try:
                from serial_comm_handler_multiprocess import SerialCommHandlerMP
            except ImportError as exc:
                raise RuntimeError("serial_gripper selected, but serial_comm_handler_multiprocess.py is unavailable") from exc
            sg = self.robot_cfg["serial_gripper"]
            self.serial_gripper = SerialCommHandlerMP(
                port=str(sg["port"]),
                baudrate=int(sg["baudrate"]),
            )

        self.default_quat = euler_xyz_deg_to_quat_xyzw(self.robot_cfg["orientation_euler_xyz_deg"])

    def current_state_text(self) -> str:
        pieces = [f"Franka host: {self.host}"]
        try:
            pose_member = getattr(self.robot, "current_pose")
            pose = pose_member() if callable(pose_member) else pose_member
            pieces.append(f"current_pose: {pose}")
            if hasattr(pose, "end_effector_pose"):
                pieces.append(f"end_effector_pose: {pose.end_effector_pose}")
        except Exception as exc:
            pieces.append(f"current_pose unavailable: {exc}")

        try:
            state_member = getattr(self.robot, "state")
            state = state_member() if callable(state_member) else state_member
            if hasattr(state, "robot_mode"):
                pieces.append(f"robot_mode: {state.robot_mode}")
            if hasattr(state, "q"):
                pieces.append(f"q: {state.q}")
            if hasattr(state, "O_T_EE"):
                pieces.append(f"O_T_EE: {state.O_T_EE}")
            if not any(line.startswith(("robot_mode:", "q:", "O_T_EE:")) for line in pieces):
                pieces.append(f"state: {state}")
        except Exception as exc:
            pieces.append(f"state unavailable: {exc}")
        return "\n".join(pieces)

    def _check_target(self, position: list[float]) -> None:
        warning = check_workspace(position, self.robot_cfg["workspace_limits_m"])
        if warning:
            raise ValueError(warning)

    def move_to(
        self,
        position: list[float],
        quaternion: list[float] | None = None,
        relative_dynamics_factor: float | None = None,
        asynchronous: bool = False,
    ) -> None:
        self._check_target(position)
        quat = quaternion or self.default_quat
        factor = (
            float(relative_dynamics_factor)
            if relative_dynamics_factor is not None
            else float(self.robot_cfg["relative_dynamics_factor"])
        )
        motion = self.CartesianMotion(
            self.Affine(np.array([float(v) for v in position], dtype=np.float64), np.array([float(v) for v in quat], dtype=np.float64)),
            self.ReferenceType.Absolute,
            relative_dynamics_factor=factor,
        )
        self.robot.move(motion, asynchronous=asynchronous)

    def move_relative(self, delta_xyz: list[float], relative_dynamics_factor: float | None = None) -> None:
        factor = (
            float(relative_dynamics_factor)
            if relative_dynamics_factor is not None
            else float(self.robot_cfg["relative_dynamics_factor"])
        )
        motion = self.CartesianMotion(
            self.Affine(np.array([float(v) for v in delta_xyz], dtype=np.float64)),
            self.ReferenceType.Relative,
            relative_dynamics_factor=factor,
        )
        self.robot.move(motion)

    def go_rest(self) -> None:
        self.move_to([float(v) for v in self.robot_cfg["rest_position_m"]])

    def gripper_open(self) -> None:
        if self.serial_gripper is not None:
            sg = self.robot_cfg["serial_gripper"]
            self.serial_gripper.listen_for_commands(str(sg["open_command"]).encode())
            time.sleep(float(sg["settle_s"]))
            return
        if self.gripper is None:
            print("[robot] No Franka gripper connected in config; skipping gripper_open().")
            return
        g = self.robot_cfg["gripper"]
        self.gripper.move(float(g["open_width_m"]), float(g["speed_m_s"]))

    def gripper_close(self) -> None:
        if self.serial_gripper is not None:
            sg = self.robot_cfg["serial_gripper"]
            self.serial_gripper.listen_for_commands(str(sg["close_command"]).encode())
            time.sleep(float(sg["settle_s"]))
            return
        if self.gripper is None:
            print("[robot] No Franka gripper connected in config; skipping gripper_close().")
            return
        g = self.robot_cfg["gripper"]
        # grasp() is safer than hard move for real stems because it allows a force limit.
        self.gripper.grasp(
            float(g["close_width_m"]),
            float(g["speed_m_s"]),
            float(g["force_n"]),
            0.004,
            0.004,
        )

    def shutdown(self) -> None:
        if self.serial_gripper is not None:
            try:
                self.serial_gripper.listen_for_commands(b"t")
                time.sleep(0.2)
            except Exception:
                pass


def confirm_step(enabled: bool, message: str) -> None:
    if not enabled:
        return
    answer = input(f"{message}\nPress Enter to continue, or type q to abort: ").strip().lower()
    if answer in {"q", "quit", "abort", "stop"}:
        raise KeyboardInterrupt("Aborted by user")


def run_grasp_sequence(
    robot: FrankaRobot,
    target: RobotTarget,
    config: dict,
    dry_run: bool,
) -> None:
    loc = config["localization"]
    require_confirm = bool(config["safety"]["require_enter_before_each_motion"])
    grasp = target.xyz_m
    pregrasp = [grasp[0], grasp[1], grasp[2] + float(loc["pregrasp_z_offset_m"])]
    lift = [grasp[0], grasp[1], grasp[2] + float(loc["lift_z_offset_m"])]

    print("\nPlanned grasp sequence:")
    print(f"  pregrasp: {pregrasp}")
    print(f"  grasp:    {grasp}")
    print(f"  lift:     {lift}")
    print(f"  yaw:      {'fixed tool orientation' if target.yaw_deg is None else target.yaw_deg}")
    if target.warning:
        print(f"  WARNING: {target.warning}")

    if dry_run:
        print("\nDRY RUN: no robot motion executed.")
        return

    confirm_step(require_confirm, "Move to pregrasp pose?")
    robot.move_to(pregrasp)
    confirm_step(require_confirm, "Open gripper?")
    robot.gripper_open()
    confirm_step(require_confirm, "Descend to grasp pose?")
    robot.move_to(grasp)
    confirm_step(require_confirm, "Close gripper?")
    robot.gripper_close()
    confirm_step(require_confirm, "Lift object/stem?")
    robot.move_to(lift)
    print("Grasp sequence finished.")
