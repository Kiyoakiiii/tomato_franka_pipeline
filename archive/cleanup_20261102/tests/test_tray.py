import numpy as np
import pytest

from seedling.gripper import SimGripper
from seedling.robot import SimRobot
from seedling.synthetic import SyntheticCamera, render, tray_sim_camera, tray_world
from seedling.task import TransplantTask
from seedling.trial_log import Session
from seedling.tray import Tray, TrayTargeter, estimate_cell_stem


def test_grid_from_rotated_corners(tmp_path):
    R = np.array([[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]])
    A = np.array([0.45, -0.1, 0.08])
    B = A + R @ [0, 0.15, 0]  # 4 columns, 50 mm
    C = A + R @ [0.10, 0, 0]  # 3 rows, 50 mm
    t = Tray("t", 3, 4, A, B, C, 0.03)
    assert np.allclose(t.grasp_point("C4"), A + R @ [0.10, 0.15, 0])
    assert t.pitch_mm() == pytest.approx((50.0, 50.0))
    assert t.squareness_deg() == pytest.approx(90.0)
    assert t.soil_z("B2") == pytest.approx(0.05)
    t2 = Tray.load(t.save(tmp_path / "tray.yaml"))
    assert np.allclose(t2.grasp_point("B3"), t.grasp_point("B3"), atol=2e-5)  # saved to 0.01 mm
    with pytest.raises(KeyError):
        t.grasp_point("D1")


def test_cell_vision_recovers_tilted_stem(cfg):
    world, tray = tray_world(tilt_deg=20, seed=2)
    T_BC, intr = tray_sim_camera()
    f = render(world, T_BC, intr, seed=2)
    errs, ok = [], 0
    for k, cid in enumerate(tray.cells()):
        est = estimate_cell_stem(f, T_BC, tray.grasp_point(cid), tray.soil_z(cid), tray.grasp_height_m, cfg)
        if not est.ok:
            continue
        ok += 1
        s, gp = world.stems[k], tray.grasp_point(cid)
        gt = s.base_B + s.dir_B * (gp[2] - s.base_B[2]) / s.dir_B[2]
        errs.append(np.linalg.norm(est.p_B[:2] - gt[:2]))
    assert ok >= 10  # a few back-row cells may be occluded
    assert np.median(errs) < 0.002  # blind error at 20 deg is ~13 mm


def test_cell_vision_empty_cell_fails(cfg):
    world, tray = tray_world(tilt_deg=0, seed=0)
    world.stems.pop(0)  # A1 empty
    T_BC, intr = tray_sim_camera()
    f = render(world, T_BC, intr)
    est = estimate_cell_stem(f, T_BC, tray.grasp_point("A1"), tray.soil_z("A1"), tray.grasp_height_m, cfg)
    assert not est.ok


@pytest.mark.parametrize("mode,tilt,expect", [("blind", 0, 1), ("vision", 0, 1), ("vision", 30, 1)])
def test_tray_trial_end_to_end(cfg, tmp_path, mode, tilt, expect):
    world, tray = tray_world(tilt_deg=tilt, seed=1)
    cfg["grasp"]["table_z_m"] = world.tray_top_z - 0.005
    cfg["task"]["reobserve_before_acquire"] = False
    world.open_width = cfg["gripper"]["open_width_m"]
    T_BC, intr = tray_sim_camera()
    task = TransplantTask(cfg, SyntheticCamera(world, T_BC, intr), None, SimRobot(cfg, world=world), SimGripper(cfg, world=world),
                          T_BC, Session(tmp_path, "tray", cfg))
    cell = "A1"
    task.targeter = TrayTargeter(tray, cell, mode)
    row = task.run_trial("t1", plant_id=cell)
    assert row["end_to_end_success"] == expect
    assert world.stems[0].extracted


def test_blind_misses_strongly_tilted_stem(cfg, tmp_path):
    world, tray = tray_world(tilt_deg=40, seed=3)
    cfg["grasp"]["table_z_m"] = world.tray_top_z - 0.005
    cfg["task"]["reobserve_before_acquire"] = False
    world.open_width = cfg["gripper"]["open_width_m"]
    T_BC, intr = tray_sim_camera()
    task = TransplantTask(cfg, SyntheticCamera(world, T_BC, intr), None, SimRobot(cfg, world=world), SimGripper(cfg, world=world),
                          T_BC, Session(tmp_path, "tray", cfg))
    results = []
    for cid in ["A1", "A2", "A3", "A4"]:
        task.targeter = TrayTargeter(tray, cid, "blind")
        results.append(task.run_trial(f"t_{cid}", plant_id=cid)["end_to_end_success"])
    assert sum(results) < 4  # 40 deg at 35 mm grasp height is ~29 mm off: some blind grasps must miss
