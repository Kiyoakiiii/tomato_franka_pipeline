"""Hand-eye calibration of the ZED, for both mounting options.

eye-to-hand (camera fixed in the room, marker on the gripper):
    T_BE_i * T_EM = T_BC * T_CM_i          unknowns T_BC, T_EM
eye-in-hand (camera on the wrist, marker fixed on the table):
    T_BF_i * T_FC * T_CM_i = T_BM          unknowns T_FC, T_BM

F is the robot flange. The wrist result is stored relative to the flange, not the
Desk end-effector frame, so changing the gripper (and its Desk EE setting) does not
invalidate the camera calibration. Fit and held-out samples are kept separate so
the reported validation error is not the fit residual.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import Frame
from .config import project_path
from .transforms import T_from_list, T_to_list, apply_T, inv_T, make_T


# ------------------------------------------------------------------ persistence --------------------
def save_T_BC(path: str | Path, T_BC: np.ndarray, meta: dict) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"T_base_cam": T_to_list(T_BC), "convention": "p_base = T_base_cam @ p_cam; camera frame IMAGE (x right, y down, z fwd), metres"}
    data.update(meta)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_T_BC(cfg: dict, path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    path = project_path(path or cfg["camera"]["T_base_cam_file"])
    if not path.exists():
        raise FileNotFoundError(f"camera extrinsics not found: {path}. (fixed-camera mount; the 30/31 hand-eye scripts are in archive/cleanup_20261102/scripts).")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return T_from_list(data["T_base_cam"]), data


# ------------------------------------------------------------------ marker detection ---------------
@dataclass
class MarkerObservation:
    marker_id: int
    T_CM: np.ndarray
    reproj_px: float
    corners_px: np.ndarray
    depth_center_C: np.ndarray | None  # marker centre from the stereo point cloud (diagnostic)


def detect_markers(frame: Frame, marker_length_m: float, dictionary: str = "DICT_4X4_50",
                   ids: list[int] | None = None) -> list[MarkerObservation]:
    """All markers of the dictionary (optionally only ``ids``), each with its own square PnP pose."""
    aruco = cv2.aruco
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), aruco.DetectorParameters())
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    corners, found, _ = det.detectMarkers(gray)
    if found is None:
        return []
    obj = marker_corners_M(marker_length_m)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-9)
    K = frame.intr.K()
    out = []
    for c, mid in sorted(zip(corners, found.reshape(-1)), key=lambda t: int(t[1])):
        if (ids is not None and int(mid) not in ids) or any(o.marker_id == int(mid) for o in out):
            continue  # unwanted id, or the same id seen twice (ambiguous)
        pts32 = c.reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray, pts32, (5, 5), (-1, -1), term)
        img_pts = pts32.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img_pts) ** 2, 1))))
        T = make_T(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
        # stereo depth at the marker centre region (inner 50% of the quad)
        poly = img_pts.mean(0) + 0.5 * (img_pts - img_pts.mean(0))
        m = np.zeros(frame.rgb.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, poly.astype(np.int32), 1)
        pts = frame.xyz[m > 0]
        pts = pts[np.isfinite(pts).all(1)]
        depth_c = np.median(pts, 0) if len(pts) > 20 else None
        out.append(MarkerObservation(int(mid), T, err, img_pts, depth_c))
    return out


def detect_marker(frame: Frame, marker_length_m: float, dictionary: str = "DICT_4X4_50", marker_id: int | None = None) -> MarkerObservation | None:
    obs = detect_markers(frame, marker_length_m, dictionary, None if marker_id is None else [marker_id])
    return obs[0] if obs else None


def marker_corners_M(marker_length_m: float) -> np.ndarray:
    """Corner coordinates in the marker frame, in OpenCV's detection order (TL, TR, BR, BL)."""
    h = marker_length_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


# ------------------------------------------------------------------ solver -------------------------
def _pack(T_BC, T_EM):
    return np.concatenate([Rotation.from_matrix(T_BC[:3, :3]).as_rotvec(), T_BC[:3, 3], Rotation.from_matrix(T_EM[:3, :3]).as_rotvec(), T_EM[:3, 3]])


def _unpack(x):
    return make_T(Rotation.from_rotvec(x[0:3]).as_matrix(), x[3:6]), make_T(Rotation.from_rotvec(x[6:9]).as_matrix(), x[9:12])


def _probe_points(L: float) -> np.ndarray:
    return np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [-L, -L, 0]], dtype=float)


def residuals_per_sample(T_BC, T_EM, T_BE_list, T_CM_list) -> np.ndarray:
    """Marker-origin position disagreement (m) between the robot chain and the camera chain."""
    o = np.zeros(3)
    return np.array([np.linalg.norm(apply_T(T_BE @ T_EM, o) - apply_T(T_BC @ T_CM, o)) for T_BE, T_CM in zip(T_BE_list, T_CM_list)])


def _park(As: list[np.ndarray], Bs: list[np.ndarray]) -> np.ndarray:
    """Closed-form X for A_k X = X B_k (Park & Martin 1994). Pairs with tiny rotations are skipped."""
    keep = [(A, B) for A, B in zip(As, Bs) if np.linalg.norm(Rotation.from_matrix(A[:3, :3]).as_rotvec()) > np.radians(3)]
    if len(keep) < 2:
        raise ValueError("calibration poses need more rotation variety")
    M = np.zeros((3, 3))
    for A, B in keep:
        M += np.outer(Rotation.from_matrix(B[:3, :3]).as_rotvec(), Rotation.from_matrix(A[:3, :3]).as_rotvec())
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in keep])
    d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in keep])
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    return make_T(R, t)


def _pairs(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def park_ax_xb(T_BE_list: list[np.ndarray], T_CM_list: list[np.ndarray]) -> np.ndarray:
    """Eye-to-hand initial guess: A = T_BE_j inv(T_BE_i), B = T_CM_j inv(T_CM_i), X = T_BC."""
    P = _pairs(len(T_BE_list))
    return _park([T_BE_list[j] @ inv_T(T_BE_list[i]) for i, j in P], [T_CM_list[j] @ inv_T(T_CM_list[i]) for i, j in P])


def solve_eye_to_hand(T_BE_list: list[np.ndarray], T_CM_list: list[np.ndarray], probe_len_m: float = 0.03) -> dict:
    if len(T_BE_list) < 5:
        raise ValueError("need at least 5 poses (15+ recommended, with varied orientations)")
    T_BC0 = park_ax_xb(T_BE_list, T_CM_list)
    T_EMs = [inv_T(TBE) @ T_BC0 @ TCM for TBE, TCM in zip(T_BE_list, T_CM_list)]
    t_EM0 = np.median([T[:3, 3] for T in T_EMs], 0)
    R_EM0 = Rotation.from_matrix(np.array([T[:3, :3] for T in T_EMs])).mean().as_matrix()
    T_EM0 = make_T(R_EM0, t_EM0)

    probes = _probe_points(probe_len_m)

    def fun(x):
        T_BC, T_EM = _unpack(x)
        res = []
        for T_BE, T_CM in zip(T_BE_list, T_CM_list):
            res.append((apply_T(T_BE @ T_EM, probes) - apply_T(T_BC @ T_CM, probes)).ravel())
        return np.concatenate(res)

    sol = least_squares(fun, _pack(T_BC0, T_EM0), loss="soft_l1", f_scale=0.002)
    T_BC, T_EM = _unpack(sol.x)
    r = residuals_per_sample(T_BC, T_EM, T_BE_list, T_CM_list)
    return {
        "T_BC": T_BC,
        "T_EM": T_EM,
        "fit_residual_mm": (r * 1000).tolist(),
        "fit_rms_mm": float(np.sqrt(np.mean(r**2)) * 1000),
        "init_T_BC": T_BC0,
    }


# ------------------------------------------------------------------ eye-in-hand (wrist camera) -----
def residuals_eye_in_hand(T_FC, T_BM, T_BF_list, T_CM_list) -> np.ndarray:
    """Distance (m) between the fixed marker origin and where each pose's camera chain puts it."""
    o = np.zeros(3)
    return np.array([np.linalg.norm(apply_T(T_BF @ T_FC @ T_CM, o) - apply_T(T_BM, o)) for T_BF, T_CM in zip(T_BF_list, T_CM_list)])


def solve_eye_in_hand(T_BF_list: list[np.ndarray], T_CM_list: list[np.ndarray], probe_len_m: float = 0.03) -> dict:
    if len(T_BF_list) < 5:
        raise ValueError("need at least 5 poses (15+ recommended, with varied orientations)")
    P = _pairs(len(T_BF_list))
    # A X = X B with A = inv(T_BF_j) T_BF_i, B = T_CM_j inv(T_CM_i), X = T_FC
    T_FC0 = _park([inv_T(T_BF_list[j]) @ T_BF_list[i] for i, j in P], [T_CM_list[j] @ inv_T(T_CM_list[i]) for i, j in P])
    T_BMs = [T_BF @ T_FC0 @ T_CM for T_BF, T_CM in zip(T_BF_list, T_CM_list)]
    T_BM0 = make_T(Rotation.from_matrix(np.array([T[:3, :3] for T in T_BMs])).mean().as_matrix(), np.median([T[:3, 3] for T in T_BMs], 0))
    probes = _probe_points(probe_len_m)

    def fun(x):
        T_FC, T_BM = _unpack(x)
        target = apply_T(T_BM, probes)
        return np.concatenate([(apply_T(T_BF @ T_FC @ T_CM, probes) - target).ravel() for T_BF, T_CM in zip(T_BF_list, T_CM_list)])

    sol = least_squares(fun, _pack(T_FC0, T_BM0), loss="soft_l1", f_scale=0.002)
    T_FC, T_BM = _unpack(sol.x)
    r = residuals_eye_in_hand(T_FC, T_BM, T_BF_list, T_CM_list)
    return {"T_FC": T_FC, "T_BM": T_BM, "fit_residual_mm": (r * 1000).tolist(), "fit_rms_mm": float(np.sqrt(np.mean(r**2)) * 1000), "init_T_FC": T_FC0}


def _project(K: np.ndarray, p_C: np.ndarray) -> np.ndarray:
    return (p_C[:, :2] / p_C[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def solve_eye_in_hand_markers(T_BF_list: list[np.ndarray], obs_list: list[dict[int, np.ndarray]], K_list: list[np.ndarray],
                              marker_length_m: float, T_CM_ref: list[np.ndarray] | None = None, ref_id: int | None = None) -> dict:
    """Eye-in-hand with one or more fixed markers, refined on corner reprojection error.

    obs_list[i] maps marker id -> 4x2 detected corners (px) in sample i. Unknowns: T_FC and one base pose per
    marker id, so the layout of the markers on the sheet need not be known. The initial guess comes from the
    pose-based solver on the reference marker (the id seen most often)."""
    counts: dict[int, int] = {}
    for o in obs_list:
        for k in o:
            counts[k] = counts.get(k, 0) + 1
    ids = sorted(counts)
    if ref_id is None:
        ref_id = max(ids, key=lambda k: (counts[k], -k))
    obj = marker_corners_M(marker_length_m)
    if T_CM_ref is None:  # per-sample PnP of the reference marker
        T_CM_ref = []
        for o, K in zip(obs_list, K_list):
            if ref_id not in o:
                T_CM_ref.append(None)
                continue
            _, rv, tv = cv2.solvePnP(obj, o[ref_id], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            T_CM_ref.append(make_T(cv2.Rodrigues(rv)[0], tv.reshape(3)))
    use = [i for i, T in enumerate(T_CM_ref) if T is not None]
    T_FC0 = solve_eye_in_hand([T_BF_list[i] for i in use], [T_CM_ref[i] for i in use])["T_FC"]
    # initial base pose of every marker: average over its PnP observations through the initial T_FC
    T_BM0 = {}
    for k in ids:
        Ts = []
        for T_BF, o, K in zip(T_BF_list, obs_list, K_list):
            if k in o:
                _, rv, tv = cv2.solvePnP(obj, o[k], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                Ts.append(T_BF @ T_FC0 @ make_T(cv2.Rodrigues(rv)[0], tv.reshape(3)))
        T_BM0[k] = make_T(Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean().as_matrix(), np.median([T[:3, 3] for T in Ts], 0))

    def unpack(x):
        T_FC = make_T(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])
        return T_FC, {k: make_T(Rotation.from_rotvec(x[6 + 6 * j:9 + 6 * j]).as_matrix(), x[9 + 6 * j:12 + 6 * j]) for j, k in enumerate(ids)}

    def residual_px(T_FC, T_BM, i):
        T_CB = inv_T(T_BF_list[i] @ T_FC)
        return {k: _project(K_list[i], apply_T(T_CB @ T_BM[k], obj)) - c for k, c in obs_list[i].items()}

    def fun(x):
        T_FC, T_BM = unpack(x)
        return np.concatenate([r.ravel() for i in range(len(obs_list)) for r in residual_px(T_FC, T_BM, i).values()])

    x0 = np.concatenate([Rotation.from_matrix(T_FC0[:3, :3]).as_rotvec(), T_FC0[:3, 3]]
                        + [np.concatenate([Rotation.from_matrix(T_BM0[k][:3, :3]).as_rotvec(), T_BM0[k][:3, 3]]) for k in ids])
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=1.0, x_scale="jac")
    T_FC, T_BM = unpack(sol.x)
    per_sample = [float(np.sqrt(np.mean([np.sum(r**2, 1).mean() for r in residual_px(T_FC, T_BM, i).values()])))
                  for i in range(len(obs_list))]
    return {"T_FC": T_FC, "T_BM": T_BM, "ref_id": ref_id, "ids": ids, "per_sample_rms_px": per_sample,
            "fit_rms_px": float(np.sqrt(np.mean(np.square(per_sample)))), "init_T_FC": T_FC0}


def predict_marker_errors(T_FC: np.ndarray, T_BM: dict[int, np.ndarray], T_BF: np.ndarray, obs: dict[int, np.ndarray],
                          K: np.ndarray, marker_length_m: float) -> dict:
    """How well a calibration predicts one (held-out) sample: corner reprojection error in px, converted to the
    lateral error in mm at the marker distance, and the 3-D offset of the marker centre measured by PnP."""
    obj = marker_corners_M(marker_length_m)
    T_CB = inv_T(T_BF @ T_FC)
    px, lat_mm, pos_mm = [], [], []
    for k, c in obs.items():
        if k not in T_BM:
            continue
        p_C = apply_T(T_CB @ T_BM[k], obj)
        e = np.linalg.norm(_project(K, p_C) - c, axis=1)
        px.append(float(e.mean()))
        lat_mm.append(float(e.mean() * p_C[:, 2].mean() / K[0, 0] * 1000))
        _, rv, tv = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        pos_mm.append(float(np.linalg.norm(apply_T(T_BF @ T_FC, tv.reshape(3)) - T_BM[k][:3, 3]) * 1000))
    return {"px": px, "lateral_mm": lat_mm, "pnp_pos_mm": pos_mm}


def solve_eye_in_hand_joint_offsets(q_list: list[np.ndarray], obs_list: list[dict[int, np.ndarray]], K_list: list[np.ndarray],
                                    marker_length_m: float, joints: tuple[int, ...] = (4,)) -> dict:
    """Like solve_eye_in_hand_markers, plus constant offsets of the given joints (0-based) on the reported
    angles. Joint 1 (same as rotating every marker about the base z) and joint 7 (same as rotating the camera
    about the flange axis) cannot be identified this way. Returns also "joint_offsets_rad" (7,)."""
    from . import kinematics

    zero = np.zeros(7)
    T_BF0 = [kinematics.fk_flange(q, zero) for q in q_list]
    res = solve_eye_in_hand_markers(T_BF0, obs_list, K_list, marker_length_m)
    if not joints:
        return {**res, "joint_offsets_rad": zero}
    ids = res["ids"]
    obj = marker_corners_M(marker_length_m)
    J = list(joints)

    def unpack(x):
        dq = np.zeros(7)
        dq[J] = x[:len(J)]
        y = x[len(J):]
        T_FC = make_T(Rotation.from_rotvec(y[:3]).as_matrix(), y[3:6])
        return dq, T_FC, {k: make_T(Rotation.from_rotvec(y[6 + 6 * j:9 + 6 * j]).as_matrix(), y[9 + 6 * j:12 + 6 * j]) for j, k in enumerate(ids)}

    def residual_px(dq, T_FC, T_BM, i):
        T_CB = inv_T(kinematics.fk_flange(q_list[i], dq) @ T_FC)
        return {k: _project(K_list[i], apply_T(T_CB @ T_BM[k], obj)) - c for k, c in obs_list[i].items()}

    def fun(x):
        dq, T_FC, T_BM = unpack(x)
        return np.concatenate([r.ravel() for i in range(len(obs_list)) for r in residual_px(dq, T_FC, T_BM, i).values()])

    x0 = np.concatenate([np.zeros(len(J)), Rotation.from_matrix(res["T_FC"][:3, :3]).as_rotvec(), res["T_FC"][:3, 3]]
                        + [np.concatenate([Rotation.from_matrix(res["T_BM"][k][:3, :3]).as_rotvec(), res["T_BM"][k][:3, 3]]) for k in ids])
    sol = least_squares(fun, x0, loss="soft_l1", f_scale=1.0, x_scale="jac")
    dq, T_FC, T_BM = unpack(sol.x)
    per_sample = [float(np.sqrt(np.mean([np.sum(r**2, 1).mean() for r in residual_px(dq, T_FC, T_BM, i).values()])))
                  for i in range(len(obs_list))]
    return {"T_FC": T_FC, "T_BM": T_BM, "ref_id": res["ref_id"], "ids": ids, "per_sample_rms_px": per_sample,
            "fit_rms_px": float(np.sqrt(np.mean(np.square(per_sample)))), "init_T_FC": res["T_FC"], "joint_offsets_rad": dq}


def fit_marker_poses(T_FC: np.ndarray, T_BF_list: list[np.ndarray], obs_list: list[dict[int, np.ndarray]],
                     K_list: list[np.ndarray], marker_length_m: float) -> dict[int, np.ndarray]:
    """Base pose of every marker with the camera calibration held fixed (corner reprojection over all samples).

    Used to test a calibration on data it was not fitted to: the markers may lie anywhere (their positions are
    estimated here), but a wrong T_FC cannot explain views from different directions with one marker pose."""
    obj = marker_corners_M(marker_length_m)
    out = {}
    for k in sorted({k for o in obs_list for k in o}):
        use = [i for i, o in enumerate(obs_list) if k in o]
        Ts = []
        for i in use:
            _, rv, tv = cv2.solvePnP(obj, obs_list[i][k], K_list[i], None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            Ts.append(T_BF_list[i] @ T_FC @ make_T(cv2.Rodrigues(rv)[0], tv.reshape(3)))
        x0 = np.concatenate([Rotation.from_matrix(Ts[0][:3, :3]).as_rotvec(), np.mean([T[:3, 3] for T in Ts], 0)])
        T_CB = [inv_T(T_BF_list[i] @ T_FC) for i in use]

        def fun(x):
            T_BM = make_T(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])
            return np.concatenate([(_project(K_list[i], apply_T(T @ T_BM, obj)) - obs_list[i][k]).ravel() for i, T in zip(use, T_CB)])

        x = least_squares(fun, x0, loss="soft_l1", f_scale=1.0, x_scale="jac").x
        out[k] = make_T(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:])
    return out


def leave_one_out_errors(T_FC: np.ndarray, T_BF_list: list[np.ndarray], obs_list: list[dict[int, np.ndarray]],
                         K_list: list[np.ndarray], marker_length_m: float) -> list[dict]:
    """For every sample: marker poses fitted (T_FC fixed) on all OTHER samples, then that sample predicted.
    Returns predict_marker_errors per sample (empty for samples whose markers are seen nowhere else)."""
    res = []
    for i in range(len(obs_list)):
        rest = [j for j in range(len(obs_list)) if j != i]
        seen = {k for j in rest for k in obs_list[j]}
        obs = {k: c for k, c in obs_list[i].items() if k in seen}
        if not obs:
            res.append({"px": [], "lateral_mm": [], "pnp_pos_mm": []})
            continue
        T_BM = fit_marker_poses(T_FC, [T_BF_list[j] for j in rest], [obs_list[j] for j in rest], [K_list[j] for j in rest], marker_length_m)
        res.append(predict_marker_errors(T_FC, T_BM, T_BF_list[i], obs, K_list[i], marker_length_m))
    return res


def view_elevation_deg(T_BC: np.ndarray) -> float:
    """Angle of the optical axis below the horizontal (90 = looking straight down)."""
    return float(np.degrees(np.arcsin(np.clip(-T_BC[2, 2], -1, 1))))


def save_T_FC(path: str | Path, T_FC: np.ndarray, meta: dict) -> Path:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"T_flange_cam": T_to_list(T_FC),
            "convention": "p_flange = T_flange_cam @ p_cam; F = FR3 flange (independent of gripper / Desk EE); camera IMAGE frame, metres"}
    data.update(meta)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_T_FC(cfg: dict, path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    path = project_path(path or cfg["camera"]["T_flange_cam_file"])
    if not path.exists():
        raise FileNotFoundError(f"wrist camera calibration not found: {path}. Run scripts/32_wrist_calib_collect.py and 33_wrist_calib_solve.py.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    from . import kinematics

    kinematics.set_joint_offsets(data.get("joint_offsets_rad"))  # estimated together with T_FC; both or neither
    return T_from_list(data["T_flange_cam"]), data


def flange_pose(robot) -> np.ndarray:
    """Flange pose from the measured joint angles through the corrected model (kinematics.JOINT_OFFSETS).
    Use this, not the robot's reported EE pose (nominal kinematics), wherever a camera pose is needed."""
    from . import kinematics

    return kinematics.fk_flange(robot.q())


def T_base_flange(T_BE: np.ndarray, cfg: dict) -> np.ndarray:
    """Flange pose from the reported EE pose and the configured Desk EE offset (robot.flange_to_ee)."""
    from .config import flange_to_ee

    return T_BE @ inv_T(flange_to_ee(cfg))


def load_samples(session_dir: str | Path) -> list[dict]:
    p = Path(session_dir) / "samples.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------------ wrist calibration poses ---------
def perturbation_poses(T_BF0: np.ndarray, n: int, trans_m: float = 0.03, rot_deg: float = 10.0, seed: int = 0,
                       up_only: bool = False) -> list[np.ndarray]:
    """Small moves around the start pose (the marker stays in view) for a first, rough solution.
    Rotations are about the flange's own axes, alternating so every axis is excited. With ``up_only`` the
    flange never goes below its start height (rotating about the flange only lifts things hanging below it)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        rv = rng.uniform(-0.4, 0.4, 3) * np.radians(rot_deg)
        rv[k % 3] = np.radians(rot_deg) * (1 if (k // 3) % 2 == 0 else -1)
        T = T_BF0 @ make_T(Rotation.from_rotvec(rv).as_matrix())
        d = rng.uniform(-trans_m, trans_m, 3)
        if up_only:
            d[2] = abs(d[2])
        T[:3, 3] = T_BF0[:3, 3] + d
        out.append(T)
    return out


def orbit_poses(T_BF0: np.ndarray, T_FC: np.ndarray, marker_B: np.ndarray, n: int, max_angle_deg: float = 25.0,
                roll_deg: float = 15.0, dist_scale: tuple[float, float] = (0.85, 1.15), min_height_m: float = 0.20,
                seed: int = 1) -> list[np.ndarray]:
    """Flange poses whose camera orbits the marker: the start camera pose is rotated rigidly about the marker
    (so the marker stays near the image centre), moved along the line of sight, and rolled about the optical
    axis. Cameras lower than ``min_height_m`` above the marker are rejected."""
    rng = np.random.default_rng(seed)
    T_BC0 = T_BF0 @ T_FC
    v0 = T_BC0[:3, 3] - marker_B
    out, tries = [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rng.uniform(0.3, 1.0) * max_angle_deg)
        R = Rotation.from_rotvec(axis * ang).as_matrix()
        c = marker_B + rng.uniform(*dist_scale) * (R @ v0)
        if c[2] - marker_B[2] < min_height_m:
            continue
        T_BC = make_T(R @ T_BC0[:3, :3], c) @ make_T(Rotation.from_euler("z", rng.uniform(-roll_deg, roll_deg), degrees=True).as_matrix())
        out.append(T_BC @ inv_T(T_FC))
    return out


def order_poses(T_start: np.ndarray, poses: list[np.ndarray], rot_weight_m_per_rad: float = 0.3) -> list[np.ndarray]:
    """Greedy nearest-neighbour order, so consecutive moves are small."""
    def dist(a, b):
        return np.linalg.norm(a[:3, 3] - b[:3, 3]) + rot_weight_m_per_rad * Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
    left, out, cur = list(poses), [], T_start
    while left:
        j = int(np.argmin([dist(cur, T) for T in left]))
        cur = left.pop(j)
        out.append(cur)
    return out


def synthetic_wrist_samples(seed: int = 0):
    """Test data for the wrist solver: four 40 mm markers on a sheet, wrist camera ~0.35 m above, 0.3 px corner noise."""
    from .synthetic import look_at

    rng = np.random.default_rng(seed)
    L = 0.04
    T_FC = make_T(Rotation.from_euler("xyz", [8, -5, 92], degrees=True).as_matrix(), [0.06, -0.05, 0.03])
    T_sheet = make_T(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [0.50, 0.08, -0.02])
    T_BM = {k: T_sheet @ make_T(np.eye(3), [sx * 0.075, sy * 0.06, 0]) for k, (sx, sy) in enumerate([(-1, 1), (1, 1), (-1, -1), (1, -1)])}
    K = np.array([[1064.0, 0, 946], [0, 1064, 536], [0, 0, 1]])
    T_BF0 = look_at(np.array([0.48, 0.10, 0.33]), T_sheet[:3, 3]) @ inv_T(T_FC)
    out = []
    poses = perturbation_poses(T_BF0, 6, up_only=True) + orbit_poses(T_BF0, T_FC, T_sheet[:3, 3], 18, 20, dist_scale=(1.05, 1.3),
                                                                      min_height_m=0.35)
    for k, T_BF in enumerate(poses):
        markers = {}
        for mid, T in T_BM.items():
            p_C = apply_T(inv_T(T_BF @ T_FC) @ T, marker_corners_M(L))
            uv = _project(K, p_C) + rng.normal(0, 0.3, (4, 2))
            if p_C[:, 2].min() > 0.1 and (uv >= 0).all() and (uv[:, 0] < 1920).all() and (uv[:, 1] < 1080).all():
                markers[str(mid)] = {"corners_px": uv.tolist(), "reproj_px": 0.3}
        out.append({"tag": f"s{k}", "T_BF": T_BF.tolist(), "K": K.tolist(), "marker_length_m": L, "markers": markers})
    return out, T_FC, T_BM


def camera_pose_in_base(cfg: dict, robot) -> np.ndarray:
    """T_BC for the current robot pose: wrist camera -> T_BF(now) @ T_FC; fixed camera -> stored T_BC.
    Only meaningful while the robot is stationary (capture and pose must refer to the same instant)."""
    if cfg["camera"].get("mount", "fixed") == "wrist":
        T_FC, _ = load_T_FC(cfg)
        return flange_pose(robot) @ T_FC
    return load_T_BC(cfg)[0]
