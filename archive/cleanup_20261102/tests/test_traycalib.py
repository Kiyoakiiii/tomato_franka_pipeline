"""Mounting correction from tray views: a known small change of the camera mounting is recovered."""
import numpy as np
from scipy.spatial.transform import Rotation

from seedling import kinematics
from seedling.traycalib import cell_pixels, fit_mounting, tray_points
from seedling.transforms import apply_T, inv_T, make_T
from seedling.wrist import look_from


def test_mounting_change_recovered():
    rng = np.random.default_rng(0)
    pitch, sign = 0.0448, -1
    names = [f"{chr(ord('A') + r)}{c + 1}" for r in range(6) for c in range(12)]
    tray = np.array([0.66, 0.21, -np.pi / 2, 0.125, 1.0])  # A1, columns along -y, rows towards +x
    P = tray_points(tray, names, pitch, sign)
    T_FC0 = make_T(Rotation.from_euler("xyz", [0, -1, 46], degrees=True).as_matrix(), [0.035, -0.12, 0.022])
    true = T_FC0 @ make_T(Rotation.from_rotvec(np.radians([0.2, -0.1, 1.3])).as_matrix(), [-0.008, 0.004, -0.010])
    K = np.array([[1065.0, 0, 960], [0, 1065.0, 540], [0, 0, 1]])
    F = np.eye(4)
    centre = P.mean(0)
    views = []
    q = np.array([0.22, -0.07, -0.3, -1.66, -0.04, 1.75, 0.67])
    for el, az, d, roll in [(60, 15, 0.72, -60), (65, 20, 0.64, 0), (70, -10, 0.61, 0), (78, -25, 0.6, 0), (73, 12, 0.72, 60),
                            (66, -20, 0.42, -30), (77, -12, 0.43, 0), (79, 14, 0.45, 60)]:
        T_BC = look_from(centre + rng.normal(0, 0.03, 3) * [1, 1, 0], np.radians(az), el, d, roll)
        qn = kinematics.solve_ik_near(T_BC @ inv_T(true) @ F, q, F, 0.05)
        assert qn is not None
        q = qn
        T_BC = kinematics.fk_flange(qn) @ true
        cells = {n: p for n, p in zip(names, P)}
        nm, uv = cell_pixels(cells, T_BC, K)
        uv = uv + rng.normal(0, 0.7, uv.shape)
        keep = (uv[:, 0] > 30) & (uv[:, 1] > 30) & (uv[:, 0] < 1890) & (uv[:, 1] < 1050)
        views.append({"q": qn, "K": K, "cells": cells, "names": [n for n, k in zip(nm, keep) if k], "uv": uv[keep]})
    fit = fit_mounting(views, T_FC0, pitch)
    err = inv_T(true) @ fit["T_FC"]
    assert np.degrees(Rotation.from_matrix(err[:3, :3]).magnitude()) < 0.1
    assert np.linalg.norm(err[:3, 3]) < 0.002
    # the tray comes out where it is
    assert np.linalg.norm(fit["tray"][:2] - tray[:2]) < 0.002
