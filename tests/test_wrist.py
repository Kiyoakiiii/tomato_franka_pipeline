import numpy as np

from seedling.config import load_config
from seedling.synthetic import sim_wrist_T_FC
from seedling.transforms import apply_T, inv_T
from seedling.wrist import lowest_body_z, path_clearance, plan_observation


def test_observation_pose_looks_at_target_and_stays_high():
    cfg = load_config(None, [])
    T_FC = sim_wrist_T_FC()
    q0 = np.array(cfg["robot"]["home_joints_rad"])
    target = np.array([0.55, 0.05, 0.06])
    floor = 0.03 + cfg["tray"]["observe"]["min_clearance_m"]
    plan = plan_observation(target, q0, T_FC, floor, cfg)
    assert plan.ok, plan.tried
    from seedling import kinematics
    from seedling.calibration import T_base_flange

    T_BC = T_base_flange(kinematics.fk(plan.q), cfg) @ T_FC
    p_C = apply_T(inv_T(T_BC), target)
    assert abs(np.linalg.norm(p_C) - cfg["tray"]["observe"]["distance_m"]) < 0.005
    assert np.linalg.norm(p_C[:2]) / p_C[2] < 0.02  # target on the optical axis
    assert path_clearance(q0, plan.q, T_FC, cfg)[0] >= floor
    assert lowest_body_z(plan.q, T_FC, cfg)[0] >= floor
