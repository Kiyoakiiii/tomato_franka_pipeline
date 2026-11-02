import numpy as np
from scipy.spatial.transform import Rotation

from seedling.calibration import solve_eye_to_hand
from seedling.synthetic import look_at
from seedling.transforms import apply_T, inv_T, make_T


def test_eye_to_hand_recovers_extrinsics():
    rng = np.random.default_rng(0)
    T_BC = look_at(np.array([0.55, -0.55, 0.45]), np.array([0.5, 0.0, 0.15]))
    T_EM = make_T(Rotation.from_euler("xyz", [90, 0, 30], degrees=True).as_matrix(), [0.0, 0.05, -0.02])
    BE, CM = [], []
    for _ in range(15):
        T_BE = make_T(Rotation.from_euler("xyz", [180 + rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-40, 40)], degrees=True).as_matrix(),
                      [0.5 + rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 0.25 + rng.uniform(-0.05, 0.05)])
        BE.append(T_BE)
        CM.append(inv_T(T_BC) @ T_BE @ T_EM)
    res = solve_eye_to_hand(BE, CM)
    assert np.allclose(res["T_BC"], T_BC, atol=1e-5)
    assert res["fit_rms_mm"] < 0.01
    p = apply_T(inv_T(T_BC), [0.5, 0, 0.1])
    assert np.linalg.norm(apply_T(res["T_BC"], p) - [0.5, 0, 0.1]) < 1e-5


def _wrist_setup(seed=0):
    from seedling.synthetic import look_at

    rng = np.random.default_rng(seed)
    # camera on the wrist: ~8 cm off the flange axis, looking roughly along the flange z axis
    T_FC = make_T(Rotation.from_euler("xyz", [8, -5, 92], degrees=True).as_matrix(), [0.06, -0.05, 0.03])
    T_BM = make_T(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [0.55, 0.05, 0.02])  # marker flat on the table
    # start pose: camera 0.45 m above-and-behind the marker, looking at it
    T_BC0 = look_at(np.array([0.30, 0.0, 0.40]), T_BM[:3, 3])
    return T_FC, T_BM, T_BC0 @ inv_T(T_FC), rng


def test_wrist_poses_keep_marker_in_view():
    from seedling.calibration import orbit_poses, perturbation_poses

    T_FC, T_BM, T_BF0, _ = _wrist_setup()
    m = T_BM[:3, 3]
    poses = perturbation_poses(T_BF0, 6) + orbit_poses(T_BF0, T_FC, m, 20)
    assert len(poses) == 26
    for T_BF in poses:
        p_C = apply_T(inv_T(T_BF @ T_FC), m)
        assert p_C[2] > 0.2  # in front of the camera
        u, v = 1064 * p_C[0] / p_C[2], 1064 * p_C[1] / p_C[2]
        assert abs(u) < 900 and abs(v) < 500  # inside a 1920x1080 image (centred coords)


def test_eye_in_hand_recovers_flange_camera_with_noise():
    from seedling.calibration import orbit_poses, perturbation_poses, solve_eye_in_hand

    T_FC, T_BM, T_BF0, rng = _wrist_setup(3)
    poses = perturbation_poses(T_BF0, 6) + orbit_poses(T_BF0, T_FC, T_BM[:3, 3], 18)
    BF, CM = [], []
    for T_BF in poses:
        T_CM = inv_T(T_BF @ T_FC) @ T_BM
        T_CM[:3, 3] += rng.normal(0, 0.001, 3)  # ~1 mm PnP noise
        T_CM[:3, :3] = T_CM[:3, :3] @ Rotation.from_rotvec(rng.normal(0, np.radians(0.2), 3)).as_matrix()
        BF.append(T_BF)
        CM.append(T_CM)
    res = solve_eye_in_hand(BF, CM)
    # what matters is the mapped position of a stem 0.45 m in front of the camera
    p_C = np.array([0.02, -0.03, 0.45])
    T_BF = BF[0]
    err = np.linalg.norm(apply_T(T_BF @ res["T_FC"], p_C) - apply_T(T_BF @ T_FC, p_C))
    assert err < 0.002
    assert np.linalg.norm(res["T_BM"][:3, 3] - T_BM[:3, 3]) < 0.002


def test_eye_in_hand_multi_marker_reprojection():
    from seedling.calibration import solve_eye_in_hand_markers, synthetic_wrist_samples

    samples, T_FC, T_BM = synthetic_wrist_samples(1)
    res = solve_eye_in_hand_markers([np.array(s["T_BF"]) for s in samples],
                                    [{int(k): np.array(v["corners_px"]) for k, v in s["markers"].items()} for s in samples],
                                    [np.array(s["K"]) for s in samples], 0.04)
    p_C = np.array([0.02, -0.03, 0.45])
    T_BF = np.array(samples[0]["T_BF"])
    assert np.linalg.norm(apply_T(T_BF @ res["T_FC"], p_C) - apply_T(T_BF @ T_FC, p_C)) < 0.001
    assert all(np.linalg.norm(res["T_BM"][k][:3, 3] - T_BM[k][:3, 3]) < 0.001 for k in T_BM)


def test_ik_near_stays_on_branch():
    from seedling import kinematics

    q0 = np.array([0.243, -0.183, -0.042, -2.421, -0.09, 2.255, 1.068])  # wrist camera looking down (lab pose)
    T0 = kinematics.fk(q0)
    T1 = T0 @ make_T(Rotation.from_euler("x", 12, degrees=True).as_matrix(), [0.02, 0.0, 0.0])
    q1 = kinematics.solve_ik_near(T1, q0)
    assert q1 is not None
    assert np.allclose(kinematics.fk(q1)[:3, 3], T1[:3, 3], atol=1e-3)
    assert np.degrees(np.abs(q1 - q0)).max() < 25


def test_leave_one_out_detects_wrong_calibration():

    from seedling.calibration import leave_one_out_errors, synthetic_wrist_samples
    from seedling.transforms import make_T

    samples, T_FC, _ = synthetic_wrist_samples(2)
    S = [s for s in samples if s["markers"]]
    args = ([np.array(s["T_BF"]) for s in S], [{int(k): np.array(v["corners_px"]) for k, v in s["markers"].items()} for s in S],
            [np.array(s["K"]) for s in S], 0.04)
    good = np.concatenate([e["px"] for e in leave_one_out_errors(T_FC, *args)])
    bad = np.concatenate([e["px"] for e in leave_one_out_errors(T_FC @ make_T(Rotation.from_euler("x", 1, degrees=True).as_matrix()), *args)])
    assert np.median(good) < 0.8
    assert np.median(bad) > 3 * np.median(good)
