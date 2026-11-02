"""Transplanting state machine.

OBSERVE -> VALIDATE -> PREGRASP -> [REOBSERVE] -> ACQUIRE -> EXTRACT -> TRANSPORT -> PLACE -> VERIFY_RELEASE -> DONE
any failure -> RECOVERY -> (OBSERVE if a retry is allowed, else FAILED); no target -> REJECTED

Gripper state and arm-level force estimates are evidence, not proof: a successful
grasp command does not establish that the stem is held, and the external wrench
estimate is not the force on the stem.
"""
from __future__ import annotations

import copy
import time
from enum import Enum

import numpy as np

from . import kinematics
from .grasp import GraspCandidate, plan_poses
from .perception import Observation, observe, save_observation
from .robot import MotionResult, RobotBase, SimRobot, UnsafeTarget
from .trial_log import Session, TrialLog
from .transforms import inv_T


class S(str, Enum):
    OBSERVE = "OBSERVE"
    VALIDATE = "VALIDATE"
    PREGRASP = "PREGRASP"
    REOBSERVE = "REOBSERVE"
    ACQUIRE = "ACQUIRE"
    EXTRACT = "EXTRACT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    VERIFY_RELEASE = "VERIFY_RELEASE"
    RECOVERY = "RECOVERY"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class TrialAbort(RuntimeError):
    pass


class TransplantTask:
    def __init__(self, cfg: dict, camera, segmenter, robot: RobotBase, gripper, T_BC: np.ndarray, session: Session,
                 ask_operator: bool = False, ids: dict | None = None, targeter=None, T_FC: np.ndarray | None = None):
        """T_BC: fixed camera in the base frame. T_FC: wrist camera on the flange (then T_BC is set at each
        stationary capture from the current flange pose)."""
        self.cfg, self.cam, self.seg, self.robot, self.gripper, self.T_BC, self.session = cfg, camera, segmenter, robot, gripper, T_BC, session
        self.T_FC = T_FC
        self.t = cfg["task"]
        self.ask_operator = ask_operator
        self.ids = ids or {}
        self.targeter = targeter or SceneTargeter()

    # ------------------------------------------------------------------ helpers ---------------------
    def _move(self, log: TrialLog, state: str, T_BG: np.ndarray, desc: str, fine: bool = False, force_limit: float | None = None) -> MotionResult:
        dyn = float(self.cfg["robot"]["fine_dynamics" if fine else "dynamics"])
        try:
            r = self.robot.move_contact_to(T_BG, dynamics=dyn, force_limit_n=force_limit, description=f"{state}: {desc}")
        except UnsafeTarget as exc:
            r = MotionResult(False, f"unsafe_target: {exc}")
        log.event(state, "motion", desc=desc, ok=r.ok, reason=r.reason, peak_force_n=None if r.peak_force_n is None else round(r.peak_force_n, 2),
                  p_B=np.round(self.robot.T_BG()[:3, 3], 4))
        return r

    def _observe(self, log: TrialLog, state: str, prefix: str, cfg: dict | None = None, with_candidates: bool = True) -> Observation:
        obs = observe(self.cam, self.seg, cfg or self.cfg, self.T_BC, q_seed=self.robot.q(), with_candidates=with_candidates)
        save_observation(obs, log.dir, prefix, self.cfg, self.T_BC)
        best = obs.best()
        log.event(state, "observe", prefix=prefix, n_instances=len(obs.instances),
                  n_ok_segments=sum(s.ok for g in obs.geoms for s in g.segments),
                  n_feasible=sum(c.feasible for c in obs.candidates), timings=obs.timings,
                  best=best.summary() if best else None, reject=obs.rejection_summary())
        return obs

    def go_home(self, log: TrialLog | None = None) -> MotionResult:
        q = np.array(self.cfg["robot"]["home_joints_rad"], dtype=float)
        r = self.robot.move_joints(q, dynamics=float(self.cfg["robot"]["dynamics"]), description="go home (observation pose)")
        if log:
            log.event("HOME", "motion", ok=r.ok, reason=r.reason)
        return r

    def _ask(self, question: str, default: str = "") -> str:
        if not self.ask_operator:
            return default
        ans = input(f"  [操作员] {question} ").strip()
        return ans or default

    # ------------------------------------------------------------------ main ------------------------
    def run_trial(self, trial_id: str, plant_id: str = "", notes: str = "") -> dict:
        log = TrialLog(self.session, trial_id)
        row = {
            "trial_id": trial_id, "session_id": self.session.dir.name, "plant_id": plant_id,
            "condition_id": self.ids.get("condition_id", ""), "model_id": self.ids.get("model_id", ""),
            "calibration_id": self.ids.get("calibration_id", ""), "setting_id": self.ids.get("setting_id", ""),
            "eligible": 1, "rejected": 0, "acquisition_attempted": 0, "extraction_attempted": 0,
            "retries": 0, "human_intervention": 0, "attempts_log_path": str(log.events.relative_to(self.session.dir)), "notes": notes,
        }
        max_attempts = int(self.t["max_attempts"])
        attempt = 0
        state = S.OBSERVE
        target: GraspCandidate | None = None
        recovery = {"reason": "", "retry": False, "stage": ""}
        peak_extraction_force = None

        log.event("START", "info", trial_id=trial_id, plant_id=plant_id)
        if not self.go_home(log).ok:
            raise TrialAbort("cannot reach home pose")
        self.gripper.open()
        t_start = time.time()

        def fail(stage: S, reason: str, retry: bool) -> S:
            recovery.update(reason=reason, retry=retry, stage=stage.value)
            return S.RECOVERY

        while True:
            if state == S.OBSERVE:
                attempt += 1
                if attempt > max_attempts:
                    row.update(failure_stage=recovery["stage"], failure_reason=f"max_attempts: {recovery['reason']}")
                    state = S.FAILED
                    continue
                if attempt > 1:
                    row["retries"] = attempt - 1
                target, why = self.targeter.acquire(self, log, attempt)
                if target is None:
                    if attempt == 1:
                        row.update(rejected=1, rejection_reason=why)
                        state = S.REJECTED
                    else:
                        row.update(failure_stage="OBSERVE", failure_reason=f"target lost on retry ({why})")
                        state = S.FAILED
                    continue
                state = S.VALIDATE

            elif state == S.VALIDATE:
                need = target.diameter_m + 2 * 0.004 + float(self.cfg["gripper"]["empty_close_width_m"])
                if float(self.cfg["gripper"]["open_width_m"]) < need:
                    row.update(rejected=1, rejection_reason=f"open width < stem diameter + margin ({need*1000:.1f} mm)")
                    state = S.REJECTED
                    continue
                log.event(state.value, "target", **target.summary())
                state = S.PREGRASP

            elif state == S.PREGRASP:
                self.gripper.open()
                if self.t.get("pregrasp_joint_motion", True):
                    # large re-orientation from the home pose: go in joint space to an IK solution seeded
                    # from the current configuration, rather than a long Cartesian line
                    T_pre_E = target.T_BG_pre @ inv_T(self.robot.T_EG)
                    margin = float(self.cfg["robot"]["ik_joint_margin_rad"])
                    q_now = self.robot.q()  # prefer the solution on the current branch (least joint change)
                    q_pre = kinematics.solve_ik_near(T_pre_E, q_now, self.robot.F_T_EE, margin)
                    if q_pre is None:
                        q_pre = kinematics.solve_ik(T_pre_E, q_now, self.robot.F_T_EE, margin)
                    if q_pre is None:
                        state = fail(S.PREGRASP, "no IK for pre-grasp from current configuration", retry=True)
                        continue
                    if self.T_FC is not None:  # wrist camera: the housing hangs beside the flange
                        from .wrist import path_clearance

                        z, part = path_clearance(q_now, q_pre, self.T_FC, self.cfg)
                        if z < float(self.cfg["grasp"]["table_z_m"]):
                            state = fail(S.PREGRASP, f"joint path to pre-grasp: {part} below the table (z={z:.3f})", retry=True)
                            continue
                    try:
                        r = self.robot.move_joints(q_pre, dynamics=float(self.cfg["robot"]["dynamics"]), description=f"{state.value}: joint move to pre-grasp")
                    except UnsafeTarget as exc:
                        r = MotionResult(False, f"unsafe_target: {exc}")
                    log.event(state.value, "motion", desc="pre-grasp (joint)", ok=r.ok, reason=r.reason, p_B=np.round(self.robot.T_BG()[:3, 3], 4))
                    if r.ok:  # settle exactly on the Cartesian pre-grasp pose
                        r = self._move(log, state.value, target.T_BG_pre, "pre-grasp (cartesian settle)", fine=True)
                else:
                    r = self._move(log, state.value, target.T_BG_pre, "pre-grasp", force_limit=float(self.t["guard_force_limit_n"]))
                if not r.ok:
                    state = fail(S.PREGRASP, r.reason, retry=True)
                    continue
                state = S.REOBSERVE if (self.t.get("reobserve_before_acquire", True) and self.targeter.can_refine(self)) else S.ACQUIRE

            elif state == S.REOBSERVE:
                _, match, dist = self.targeter.refine(self, log, attempt, target)
                if match is None:
                    log.event(state.value, "info", result="inconclusive (target not re-found; keeping first estimate)")
                elif dist > float(self.t["reobserve_max_shift_m"]):
                    log.event(state.value, "info", result="target moved", shift_mm=round(dist * 1000, 1))
                    state = fail(S.REOBSERVE, f"target shifted {dist*1000:.1f} mm", retry=True)
                    continue
                else:
                    log.event(state.value, "info", result="refined", shift_mm=round(dist * 1000, 2))
                    target = match
                    r = self._move(log, state.value, target.T_BG_pre, "refined pre-grasp", fine=True, force_limit=float(self.t["guard_force_limit_n"]))
                    if not r.ok:
                        state = fail(S.REOBSERVE, r.reason, retry=True)
                        continue
                state = S.ACQUIRE

            elif state == S.ACQUIRE:
                row["acquisition_attempted"] = 1
                r = self._move(log, state.value, target.T_BG, "approach to stem", fine=True, force_limit=float(self.t["guard_force_limit_n"]))
                if not r.ok:
                    state = fail(S.ACQUIRE, f"approach: {r.reason}", retry=True)
                    continue
                g = self.gripper.grasp(target.grasp_width_m)
                log.event(state.value, "grasp", command_ok=g.command_ok, width_mm=None if g.width_m is None else round(g.width_m * 1000, 2),
                          holding=g.holding, reason=g.reason)
                if g.holding is False:
                    row["acquisition_success"] = 0
                    state = fail(S.ACQUIRE, "grasp missed (gripper closed to empty width)", retry=True)
                    continue
                row["acquisition_success"] = 1 if g.holding else None  # None: gripper cannot tell
                state = S.EXTRACT

            elif state == S.EXTRACT:
                row["extraction_attempted"] = 1
                n = int(self.t["extraction_steps"])
                dz = float(self.t["extraction_height_m"]) / n
                T0 = self.robot.T_BG()
                peak = 0.0
                ok = True
                for k in range(1, n + 1):
                    T = T0.copy()
                    T[2, 3] += dz * k
                    r = self._move(log, state.value, T, f"lift {k}/{n}", fine=True, force_limit=float(self.t["extraction_force_limit_n"]))
                    peak = max(peak, r.peak_force_n or 0.0)
                    if not r.ok:
                        ok = False
                        reason = "extraction resistance above limit" if r.reaction_triggered else r.reason
                        break
                peak_extraction_force = peak
                if not ok:
                    row["extraction_success"] = 0
                    # do not regrasp a plant that resisted extraction (possible damage)
                    state = fail(S.EXTRACT, reason, retry=False)
                    continue
                w = self.gripper.width()
                holding = self.gripper.holding_from_width(w)
                log.event(state.value, "check", width_mm=None if w is None else round(w * 1000, 2), holding=holding, peak_force_n=round(peak, 2))
                if holding is False:
                    row["extraction_success"] = 0
                    state = fail(S.EXTRACT, "slipped during extraction", retry=False)
                    continue
                if self.t.get("verify_extraction_visual", True):
                    still = self.targeter.still_at_origin(self, log, attempt, target)
                    log.event(state.value, "visual", stem_still_at_origin=still)
                    if still:
                        row["extraction_success"] = 0
                        state = fail(S.EXTRACT, "stem still visible at original position", retry=False)
                        continue
                row["extraction_success"] = 1
                state = S.TRANSPORT

            elif state == S.TRANSPORT:
                plan = dict(plan_poses(target.T_BG, self.cfg))
                names = ["transport_lift"] + [k for k in plan if k.startswith("waypoint_")] + ["above_place"]
                guard = float(self.t["guard_force_limit_n"])
                r = MotionResult(True)
                for name in names:
                    r = self._move(log, state.value, plan[name], name, force_limit=guard)
                    if not r.ok:
                        break
                if not r.ok:
                    state = fail(S.TRANSPORT, r.reason, retry=False)
                    continue
                state = S.PLACE

            elif state == S.PLACE:
                plan = dict(plan_poses(target.T_BG, self.cfg))
                r = self._move(log, state.value, plan["place"], "descend into receiver", fine=True, force_limit=float(self.t["guard_force_limit_n"]))
                if not r.ok and not r.reaction_triggered:
                    state = fail(S.PLACE, r.reason, retry=False)
                    continue
                if r.reaction_triggered:
                    log.event(state.value, "info", result="contact during descent; releasing here")
                self.gripper.open()
                time.sleep(float(self.t["release_dwell_s"]))
                T_back = self.robot.T_BG()
                T_back[:3, 3] -= T_back[:3, 2] * float(self.cfg["grasp"]["pregrasp_distance_m"])
                self._move(log, state.value, T_back, "retreat along -approach", fine=True)
                T_up = self.robot.T_BG()
                T_up[2, 3] = max(T_up[2, 3], plan["place_up"][2, 3])
                self._move(log, state.value, T_up, "retreat up")
                state = S.VERIFY_RELEASE

            elif state == S.VERIFY_RELEASE:
                w = self.gripper.width()
                released = None if w is None else w > target.grasp_width_m + 0.005
                log.event(state.value, "check", width_mm=None if w is None else round(w * 1000, 2), released=released)
                ans = self._ask("放置是否成功（苗在接收位且直立）? [y/n，回车=按传感结果]", "")
                if ans.lower().startswith("y"):
                    row["placement_success"] = 1
                elif ans.lower().startswith("n"):
                    row["placement_success"] = 0
                else:
                    row["placement_success"] = None if released is None else int(released)
                state = S.DONE

            elif state == S.RECOVERY:
                log.event(state.value, "enter", **recovery)
                self._recover(log)
                if recovery["retry"] and attempt < max_attempts:
                    state = S.OBSERVE
                else:
                    row.update(failure_stage=recovery["stage"], failure_reason=recovery["reason"])
                    state = S.FAILED

            elif state in (S.DONE, S.REJECTED, S.FAILED):
                break

        row["cycle_time_s"] = round(time.time() - t_start, 2)
        row["end_to_end_success"] = int(state == S.DONE and row.get("acquisition_success") != 0 and row.get("extraction_success") == 1
                                        and row.get("placement_success") == 1)
        if state == S.DONE:
            self.go_home(log)
        if self.ask_operator and state != S.REJECTED:
            row["damage_assessable"] = 1
            row["visible_stem_damage"] = int(self._ask("茎部可见新增损伤? [y/N]", "n").lower().startswith("y"))
            row["visible_root_damage"] = int(self._ask("根部可见新增损伤? [y/N]", "n").lower().startswith("y"))
            row["human_intervention"] = int(self._ask("是否有人工介入? [y/N]", "n").lower().startswith("y"))
            extra = self._ask("备注（可空）:", "")
            if extra:
                row["notes"] = (row["notes"] + " | " + extra).strip(" |")
        if peak_extraction_force is not None:
            row["notes"] = (row["notes"] + f" | peak_extraction_force_est_n={peak_extraction_force:.1f}").strip(" |")
        log.event("END", "result", final_state=state.value, **{k: row.get(k) for k in ("acquisition_success", "extraction_success", "placement_success", "end_to_end_success", "failure_reason", "rejection_reason", "cycle_time_s")})
        self.session.append_csv("end_to_end_trials.csv", row)
        return row

    def _recover(self, log: TrialLog) -> None:
        if self.robot.has_errors():
            # never clear errors on the real robot without an operator decision
            if isinstance(self.robot, SimRobot):
                ans = "r"
            else:
                ans = input("  [操作员] 机器人处于错误状态（检查原因/周围环境）。输入 r 清错并退回，其他键中止: ").strip()
            if ans.lower() != "r":
                raise TrialAbort("robot error; operator chose to stop")
            ok = self.robot.recover()
            log.event("RECOVERY", "recover_from_errors", ok=ok)
        self.gripper.open()
        T = self.robot.T_BG()
        T_back = T.copy()
        T_back[:3, 3] -= T[:3, 2] * float(self.cfg["grasp"]["pregrasp_distance_m"])
        r = self._move(log, "RECOVERY", T_back, "retreat along -approach", fine=True)
        if r.ok:
            T_up = T_back.copy()
            T_up[2, 3] = max(T_up[2, 3] + 0.03, float(self.t["transport_height_m"]))
            self._move(log, "RECOVERY", T_up, "retreat up")
        self.go_home(log)


class SceneTargeter:
    """Default targeting: segment the whole image and pick the best-ranked grasp candidate."""

    name = "scene"

    def can_refine(self, task: TransplantTask) -> bool:
        return True

    def acquire(self, task: TransplantTask, log: TrialLog, attempt: int):
        obs = task._observe(log, S.OBSERVE.value, f"a{attempt}_observe")
        best = obs.best()
        return best, ("" if best else obs.rejection_summary())

    def refine(self, task: TransplantTask, log: TrialLog, attempt: int, target: GraspCandidate):
        """Returns (status, candidate, shift_m), status "found" or "inconclusive"; the task judges the shift."""
        cfg2 = copy.deepcopy(task.cfg)  # only re-evaluate the approach direction already chosen
        cfg2["grasp"]["approach_yaw_offsets_deg"] = [target.yaw_offset_deg]
        cfg2["grasp"]["approach_pitch_deg"] = [target.pitch_deg]
        obs2 = task._observe(log, S.REOBSERVE.value, f"a{attempt}_reobserve", cfg=cfg2)
        best, dist = None, np.inf
        for c in obs2.candidates:
            d = float(np.linalg.norm(c.p_B - target.p_B))
            if c.feasible and c.yaw_offset_deg == target.yaw_offset_deg and c.pitch_deg == target.pitch_deg and d < dist:
                best, dist = c, d
        return ("inconclusive" if best is None else "found"), best, dist

    def still_at_origin(self, task: TransplantTask, log: TrialLog, attempt: int, target: GraspCandidate) -> bool | None:
        obs = task._observe(log, S.EXTRACT.value, f"a{attempt}_post_extract", with_candidates=False)
        T_BC = task.T_BC
        for g in obs.geoms:
            for sg in g.segments:
                if sg.p_C is None:
                    continue
                p = T_BC[:3, :3] @ np.array(sg.p_C) + T_BC[:3, 3]
                if np.linalg.norm(p[:2] - target.p_B[:2]) < 0.015 and p[2] < target.p_B[2] + 0.01:
                    return True
        return False
