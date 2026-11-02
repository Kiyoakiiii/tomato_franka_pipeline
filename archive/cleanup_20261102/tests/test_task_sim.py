import csv

import numpy as np
import pytest

from seedling.gripper import SimGripper
from seedling.robot import SimRobot
from seedling.synthetic import GTSegmenter, SyntheticCamera, default_sim_camera, default_world
from seedling.task import TransplantTask
from seedling.trial_log import Session
from seedling.transforms import make_T


def _task(cfg, tmp_path, world=None, T_BC_used=None):
    world = world or default_world(0, 1)
    world.open_width = cfg["gripper"]["open_width_m"]
    T_BC, intr = default_sim_camera()
    cam = SyntheticCamera(world, T_BC, intr)
    session = Session(tmp_path, "test", cfg)
    t = TransplantTask(cfg, cam, GTSegmenter(), SimRobot(cfg, world=world), SimGripper(cfg, world=world),
                       T_BC if T_BC_used is None else T_BC_used, session)
    return t, world, session


def _rows(session):
    with (session.dir / "end_to_end_trials.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_full_transplant_success(cfg, tmp_path):
    task, world, session = _task(cfg, tmp_path)
    row = task.run_trial("t1")
    assert row["end_to_end_success"] == 1
    assert world.stems[0].extracted and not world.stems[0].attached
    place = np.array(cfg["task"]["place_position_m"])
    assert np.linalg.norm(world.stems[0].base_B[:2] - place[:2]) < 0.03
    assert _rows(session)[0]["end_to_end_success"] == "1"


def test_extraction_resistance_stops_without_retry(cfg, tmp_path):
    world = default_world(0, 1)
    world.stems[0].extraction_force_n = 100.0
    task, world, session = _task(cfg, tmp_path, world)
    row = task.run_trial("t1")
    assert row["end_to_end_success"] == 0 and row["extraction_success"] == 0
    assert row["failure_stage"] == "EXTRACT" and row["retries"] == 0
    assert not world.stems[0].attached  # released during recovery


def test_calibration_error_causes_missed_grasp_and_bounded_retries(cfg, tmp_path):
    T_BC, _ = default_sim_camera()
    T_bad = make_T(None, [0.0, 0.0, 0.0]) @ T_BC
    T_bad[:3, 3] += [0.0, 0.03, 0.0]  # 30 mm extrinsic error, larger than the pad opening
    task, world, session = _task(cfg, tmp_path, T_BC_used=T_bad)
    row = task.run_trial("t1")
    assert row["acquisition_success"] == 0
    assert row["end_to_end_success"] == 0
    assert row["retries"] == int(cfg["task"]["max_attempts"]) - 1


def test_no_target_is_rejected(cfg, tmp_path):
    world = default_world(0, 1)
    world.stems = []
    task, world, session = _task(cfg, tmp_path, world)
    row = task.run_trial("t1")
    assert row["rejected"] == 1 and row["rejection_reason"] == "no_stem_detected"


def test_unknown_csv_column_raises(cfg, tmp_path):
    s = Session(tmp_path, "x", cfg)
    with pytest.raises(KeyError):
        s.append_csv("end_to_end_trials.csv", {"not_a_column": 1})
