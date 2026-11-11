"""Robot arm interface: franky (real FR3) and a kinematic simulator with identical API.

Every motion goes through ``RobotBase._validate`` (finite, rigid, contact point
inside the workspace box, IK reachable) before anything is sent. Error recovery
is never automatic; callers must call ``recover()`` explicitly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import kinematics
from .config import ee_to_contact, flange_to_ee
from .transforms import apply_T, check_rigid, make_T


class UnsafeTarget(ValueError):
    pass


@dataclass
class MotionResult:
    ok: bool
    reason: str = ""
    reaction_triggered: bool = False
    peak_force_n: float | None = None  # peak |F_ext - F_baseline| during the motion (arm-level estimate)
    duration_s: float = 0.0
    extra: dict = field(default_factory=dict)


class RobotBase:
    def __init__(self, cfg: dict, step_mode: bool = False, confirm_fn=None):
        self.cfg = cfg
        self.rcfg = cfg["robot"]
        self.step_mode = step_mode
        self.confirm_fn = confirm_fn or _console_confirm
        self.F_T_EE = flange_to_ee(cfg)
        self.T_EG = ee_to_contact(cfg)

    # ---- state -----------------------------------------------------------------------------
    def T_BE(self) -> np.ndarray:
        raise NotImplementedError

    def q(self) -> np.ndarray:
        raise NotImplementedError

    def external_force(self) -> np.ndarray:
        """Estimated external force at the EE in the base frame (3,), N. Not a stem force measurement."""
        raise NotImplementedError

    def T_BG(self) -> np.ndarray:
        return self.T_BE() @ self.T_EG

    # ---- safety ----------------------------------------------------------------------------
    def _validate(self, T_BE: np.ndarray, check_ik: bool = True) -> None:
        T_BE = np.asarray(T_BE, dtype=float)
        try:
            check_rigid(T_BE)
        except ValueError as exc:
            raise UnsafeTarget(str(exc)) from exc
        p = apply_T(T_BE @ self.T_EG, np.zeros(3))
        ws = self.rcfg["workspace_m"]
        for i, ax in enumerate("xyz"):
            lo, hi = ws[ax]
            if not lo <= p[i] <= hi:
                raise UnsafeTarget(f"contact point {ax}={p[i]:.3f} m outside workspace [{lo}, {hi}]")
        if check_ik and kinematics.solve_ik(T_BE, self.q(), self.F_T_EE, float(self.rcfg["ik_joint_margin_rad"])) is None:
            raise UnsafeTarget("no IK solution within joint limits (margin applied)")

    def confirm(self, description: str) -> None:
        if self.step_mode:
            self.confirm_fn(description)

    # ---- motions ---------------------------------------------------------------------------
    def move_to_pose(self, T_BE: np.ndarray, dynamics=None, force_limit_n: float | None = None,
                     description: str = "move") -> MotionResult:
        raise NotImplementedError

    def move_contact_to(self, T_BG: np.ndarray, **kw) -> MotionResult:
        """Move so that the gripper contact frame G reaches T_BG."""
        return self.move_to_pose(T_BG @ np.linalg.inv(self.T_EG), **kw)

    def move_joints(self, q: np.ndarray, dynamics=None, description: str = "joint move") -> MotionResult:
        raise NotImplementedError

    def has_errors(self) -> bool:
        return False

    def recover(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _console_confirm(description: str) -> None:
    ans = input(f"\n[STEP] {description}\n  Enter = 执行, q = 中止: ").strip().lower()
    if ans in {"q", "quit", "abort"}:
        raise KeyboardInterrupt("aborted by operator")


SIM_ENV = "SEEDLING_SIM_Q"


class PlanningOnlyRobot(RobotBase):
    """What FrankyRobot gives when the environment variable SEEDLING_SIM_Q names a JSON file with "q" (the layout
    simulation, scripts/74): it reports those joints, never imports franky nor connects to the robot, and refuses
    every motion. (A sitecustomize stand-in used before 2026-11-04 was sometimes overwritten during the import, and
    the planning scripts then connected to the real robot: they only read its joints.)"""

    def __init__(self, cfg: dict, step_mode: bool = False, confirm_fn=None):
        import json
        import os

        super().__init__(cfg, step_mode, confirm_fn)
        self._q = np.array(json.load(open(os.environ[SIM_ENV]))["q"], float)

    def q(self):
        return self._q.copy()

    def T_BE(self):
        return kinematics.fk(self._q, self.F_T_EE)

    def external_force(self):
        return np.zeros(3)

    def has_errors(self) -> bool:
        return False

    def _refuse(self, *a, **k):
        raise SystemExit("planning-only robot (SEEDLING_SIM_Q is set): no motion")

    move_to_pose = move_contact_to = move_joints = recover = _refuse

    def close(self) -> None:
        pass


STATE_ENV = "SEEDLING_SIM_STATE"


class StateSimRobot(RobotBase):
    """What FrankyRobot gives when SEEDLING_SIM_STATE names a JSON file (the whole-run simulation, scripts/80): the
    joints live in that file, so the processes of a run (77 -> 69 -> 65 / 71 / 68) see one arm. Every joint move is
    checked against the world model of the current tray and disc files (every 2 deg) and logged with its smallest
    clearance; a move that goes more than 5 mm into something fails (as the real reflex would stop it)."""

    def __init__(self, cfg: dict, step_mode: bool = False, confirm_fn=None):
        import os

        super().__init__(cfg, step_mode, confirm_fn)
        self.path = os.environ[STATE_ENV]
        self.cfg_ = cfg

    def _load(self) -> dict:
        import json

        return json.load(open(self.path))

    def q(self):
        return np.array(self._load()["q"], float)

    def T_BE(self):
        return kinematics.fk(self.q(), self.F_T_EE)

    def external_force(self):
        return np.zeros(3)

    def has_errors(self) -> bool:
        return False

    def recover(self) -> bool:
        return True

    def move_joints(self, q, dynamics=None, description="joint move") -> MotionResult:
        import json
        import os

        from .calibration import load_T_FC
        from .config import PROJECT_DIR
        from .world import World

        q = np.asarray(q, float).reshape(7)
        st = self._load()
        qa = np.array(st["q"], float)
        if os.environ.get("SEEDLING_SIM_NOCHECK"):  # profiling the program's own time: the move taken as it is
            st["q"] = q.tolist()
            json.dump(st, open(self.path, "w"))
            return MotionResult(True, "", False, 0.0)
        grid = json.loads((PROJECT_DIR / self.cfg_["tray"]["grid_file"]).read_text())
        df = PROJECT_DIR / "calibration" / "disc.json"
        disc = json.loads(df.read_text()) if df.exists() else None
        T_FC, _ = load_T_FC(self.cfg_)
        # at the plant the fingers and the plants are meant to touch: only rigid things and the arm itself are checked
        m, where = World(self.cfg_, T_FC, grid, disc).worst(qa, q, fingers=False, plants=False, plug=False)
        rec = {"description": description, "from": qa.tolist(), "to": q.tolist(), "margin_mm": round(m * 1000, 1), "where": where,
               "max_joint_deg": round(float(np.degrees(np.abs(q - qa)).max()), 1),
               "dynamics": "normal" if dynamics is None else ("fine" if isinstance(dynamics, str) else dynamics)}
        st.setdefault("log", []).append(rec)
        if m < -0.005:
            st["log"][-1]["failed"] = True
            json.dump(st, open(self.path, "w"))
            return MotionResult(False, f"simulated collision: {where} {m * 1000:+.0f} mm (more than 5 mm into it)", True, 0.0)
        st["q"] = q.tolist()
        tmp = self.path + ".tmp"
        json.dump(st, open(tmp, "w"))
        os.replace(tmp, self.path)
        return MotionResult(True)

    def move_to_pose(self, T_BE, dynamics=None, force_limit_n=None, description="move") -> MotionResult:
        qn = kinematics.solve_ik_near(np.asarray(T_BE), self.q(), self.F_T_EE)
        if qn is None:
            return MotionResult(False, "no IK", False, 0.0)
        return self.move_joints(qn, dynamics, description)

    def move_contact_to(self, T_BG, **kw) -> MotionResult:
        return self.move_to_pose(np.asarray(T_BG) @ np.linalg.inv(self.T_EG), **kw)

    def close(self) -> None:
        pass


SHARED_ENV = "SEEDLING_SHARED_ROBOT"  # set by a run that executes its steps in one process (77 / 69 --inproc)
_SHARED: dict = {}


class FrankyRobot(RobotBase):
    """With SEEDLING_SHARED_ROBOT set, one connection for the whole process: the steps of a run executed in one process
    (seedling.inproc) use the same franky.Robot instead of connecting again for every step (2026-11-10)."""

    def __new__(cls, *a, **k):
        import os

        if os.environ.get(STATE_ENV):
            return StateSimRobot(*a, **k)
        if os.environ.get(SIM_ENV):
            return PlanningOnlyRobot(*a, **k)
        if os.environ.get(SHARED_ENV) and "robot" in _SHARED:
            return _SHARED["robot"]
        return super().__new__(cls)

    def __init__(self, cfg: dict, step_mode: bool = True, confirm_fn=None):
        import os

        if getattr(self, "_connected", False):  # the shared connection: only the caller's settings
            self.step_mode = step_mode
            self.confirm_fn = confirm_fn or _console_confirm
            return
        super().__init__(cfg, step_mode, confirm_fn)
        import franky

        self.fr = franky
        rt = franky.RealtimeConfig.Enforce if self.rcfg.get("realtime", "enforce") == "enforce" else franky.RealtimeConfig.Ignore
        self.robot = franky.Robot(str(self.rcfg["host"]), realtime_config=rt)
        # franky multiplies this global factor with each motion's own factor; keep it at 1 so the per-motion
        # value (config robot.dynamics / fine_dynamics, or the caller's) is the real one
        self.robot.relative_dynamics_factor = franky.RelativeDynamicsFactor(1.0, 1.0, 1.0)
        if self.rcfg.get("collision_force_n"):
            f = float(self.rcfg["collision_force_n"])
            self.robot.set_collision_behavior([20.0] * 7, [f] * 6)
        self._check_ee_frame()
        self._connected = True
        if os.environ.get(SHARED_ENV):
            _SHARED["robot"] = self

    MAX_DYNAMICS = 0.2  # hard cap on any motion's relative speed/acceleration/jerk

    def _dyn(self, dynamics):
        """None: robot.dynamics, "fine": robot.fine_dynamics, else the caller's value. A value is one factor for
        velocity, acceleration and jerk, or [velocity, acceleration, jerk]: long moves are limited by the velocity,
        so that can go up while the jerk stays low (0.06 for all three gave one acceleration-discontinuity reflex)."""
        d = self.rcfg["dynamics"] if dynamics is None else self.rcfg["fine_dynamics"] if isinstance(dynamics, str) else dynamics
        v, a, j = (min(float(x), self.MAX_DYNAMICS) for x in ((d, d, d) if np.isscalar(d) else d))
        return self.fr.RelativeDynamicsFactor(v, a, j)

    def _check_ee_frame(self) -> None:
        """Warn when Desk's EE frame disagrees with the configured flange_to_ee (FK vs reported pose)."""
        T_fk = kinematics.fk_flange(self.q(), np.zeros(7)) @ self.F_T_EE  # nominal model, as Desk/franky use
        T_rb = self.T_BE()
        err = np.linalg.norm(T_fk[:3, 3] - T_rb[:3, 3])
        if err > 0.005:
            print(f"[robot] WARNING: FK with configured flange_to_ee differs from robot EE pose by {err*1000:.1f} mm; "
                  "check robot.flange_to_ee against the end-effector configured in Desk.")

    def T_BE(self) -> np.ndarray:
        return np.array(self.robot.current_pose.end_effector_pose.matrix, dtype=float)

    def q(self) -> np.ndarray:
        return np.array(self.robot.current_joint_positions, dtype=float).reshape(7)

    def external_force(self) -> np.ndarray:
        return np.array(self.robot.state.O_F_ext_hat_K, dtype=float).reshape(6)[:3]

    def has_errors(self) -> bool:
        return bool(self.robot.has_errors)

    def recover(self) -> bool:
        return bool(self.robot.recover_from_errors())

    def _run(self, motion, force_limit_n: float | None, f0: np.ndarray) -> MotionResult:
        fr = self.fr
        triggered = {"v": False}
        if force_limit_n:
            lim = float(force_limit_n)
            ms = [fr.Measure.FORCE_X, fr.Measure.FORCE_Y, fr.Measure.FORCE_Z]
            cond = None
            for m, b in zip(ms, f0):
                c = ((m - float(b)) > lim) | ((m - float(b)) < -lim)
                cond = c if cond is None else (cond | c)
            reaction = fr.Reaction(cond, fr.CartesianStopMotion())
            reaction.register_callback(lambda *_: triggered.__setitem__("v", True))
            motion.add_reaction(reaction)
        t0 = time.time()
        peak = 0.0
        self.trace = [self.q()]  # joints during the motion (~10 ms apart), to compare with the planned path
        try:
            self.robot.move(motion, asynchronous=True)
            while not self.robot.join_motion(timeout=0.01):
                peak = max(peak, float(np.linalg.norm(self.external_force() - f0)))
                self.trace.append(self.q())
        except fr.ControlException as exc:
            return MotionResult(False, f"control_exception: {exc}", triggered["v"], peak, time.time() - t0)
        except fr.CommandException as exc:
            return MotionResult(False, f"command_exception: {exc}", triggered["v"], peak, time.time() - t0)
        peak = max(peak, float(np.linalg.norm(self.external_force() - f0)))
        if triggered["v"]:
            return MotionResult(False, "force_limit_reaction", True, peak, time.time() - t0)
        return MotionResult(True, "", False, peak, time.time() - t0)

    def move_to_pose(self, T_BE, dynamics=None, force_limit_n=None, description="move") -> MotionResult:
        self._validate(T_BE)
        self.confirm(f"{description}: EE -> {np.round(np.asarray(T_BE)[:3, 3], 4).tolist()} m")
        fr = self.fr
        d = self._dyn(dynamics)
        motion = fr.CartesianMotion(fr.Affine(np.asarray(T_BE, dtype=float)), fr.ReferenceType.Absolute, relative_dynamics_factor=d)
        return self._run(motion, force_limit_n, self.external_force())

    def move_joints(self, q, dynamics=None, description="joint move") -> MotionResult:
        q = np.asarray(q, dtype=float).reshape(7)
        if not np.all(np.isfinite(q)) or np.any(q < kinematics.Q_MIN) or np.any(q > kinematics.Q_MAX):
            raise UnsafeTarget("joint target outside FR3 limits")
        self._validate(kinematics.fk(q, self.F_T_EE), check_ik=False)
        self.confirm(f"{description}: q -> {np.round(q, 3).tolist()}")
        d = self._dyn(dynamics)
        q0 = self.q()
        motion = self.fr.JointMotion(q, relative_dynamics_factor=d)
        r = self._run(motion, None, self.external_force())
        # the checks before every move assume the straight joint path q0 -> q (franky's time-synchronised profiles
        # stay within ~0.6 deg of it in simulation): how far the real motion left it
        dq = q - q0
        if len(self.trace) > 2 and float(dq @ dq) > 1e-8:
            Q = np.array(self.trace) - q0
            s = np.clip(Q @ dq / float(dq @ dq), 0.0, 1.0)
            r.extra["off_straight_deg"] = float(np.degrees(np.abs(Q - np.outer(s, dq)).max()))
            if r.extra["off_straight_deg"] > 2.0:
                print(f"[robot] WARNING: {description}: the joints left the planned straight path by {r.extra['off_straight_deg']:.1f} deg")
        return r

    def close(self) -> None:
        try:
            self.robot.stop()
        except Exception:
            pass


class SimRobot(RobotBase):
    """Kinematic stand-in: motions complete instantly; optional world hooks inject forces."""

    def __init__(self, cfg: dict, step_mode: bool = False, confirm_fn=None, world=None, q0=None):
        super().__init__(cfg, step_mode, confirm_fn)
        self._q = np.array(q0 if q0 is not None else cfg["robot"]["home_joints_rad"], dtype=float)
        self._T = kinematics.fk(self._q, self.F_T_EE)
        self.world = world
        self.log: list[dict] = []

    def T_BE(self):
        return self._T.copy()

    def q(self):
        return self._q.copy()

    def external_force(self):
        return np.zeros(3)

    def move_to_pose(self, T_BE, dynamics=None, force_limit_n=None, description="move") -> MotionResult:
        self._validate(T_BE)
        self.confirm(f"{description}: EE -> {np.round(np.asarray(T_BE)[:3, 3], 4).tolist()} m")
        # a Cartesian motion keeps the current IK branch
        q = kinematics.solve_ik_near(T_BE, self._q, self.F_T_EE)
        if q is None:
            q = kinematics.solve_ik(T_BE, self._q, self.F_T_EE)
        T_from = self._T
        self.log.append({"type": "cartesian", "description": description, "target": np.asarray(T_BE).tolist()})
        force = self.world.motion_force(T_from @ self.T_EG, T_BE @ self.T_EG) if self.world else 0.0
        if force_limit_n and force > force_limit_n:
            # stop part-way, as the real reaction would
            T_stop = T_from.copy()
            T_stop[:3, 3] = T_from[:3, 3] + 0.3 * (np.asarray(T_BE)[:3, 3] - T_from[:3, 3])
            self._T = T_stop
            return MotionResult(False, "force_limit_reaction", True, force)
        self._T = np.asarray(T_BE, dtype=float).copy()
        self._q = q if q is not None else self._q
        if self.world:
            self.world.on_moved(self._T @ self.T_EG)
        return MotionResult(True, "", False, force)

    def move_joints(self, q, dynamics=None, description="joint move") -> MotionResult:
        q = np.asarray(q, dtype=float).reshape(7)
        T = kinematics.fk(q, self.F_T_EE)
        self._validate(T, check_ik=False)
        self.confirm(f"{description}: q -> {np.round(q, 3).tolist()}")
        self._q, self._T = q, T
        self.log.append({"type": "joint", "description": description, "target": q.tolist()})
        if self.world:
            self.world.on_moved(self._T @ self.T_EG)
        return MotionResult(True)


def make_robot(cfg: dict, sim: bool, step_mode: bool, world=None) -> RobotBase:
    if sim:
        return SimRobot(cfg, step_mode=step_mode, world=world)
    return FrankyRobot(cfg, step_mode=step_mode)


def identity() -> np.ndarray:
    return make_T()
