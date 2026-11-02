import numpy as np

from seedling.camera import Intrinsics
from seedling.config import load_config
from seedling.stemline import estimate_stem_line
from seedling.synthetic import SimStem, SimWorld, render_stereo
from seedling.wrist import look_from

INTR = Intrinsics(1064.0, 1064.0, 960.0, 540.0, 1920, 1080)


def _world(tilt_deg, az_deg):
    t, a = np.radians(tilt_deg), np.radians(az_deg)
    d = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
    base = np.array([0.55, 0.02, 0.10])
    w = SimWorld([SimStem(base, d, length=0.09, radius=0.0012)], tray_top_z=0.105, tray_xy=(0.45, 0.65, -0.08, 0.12))
    return w, base, d


def _estimate(tilt, az, cfg):
    world, base, d = _world(tilt, az)
    target = base + d * 0.03
    T_BC = look_from(target, np.arctan2(target[1], target[0]), 50.0, 0.36, 0.0)
    return estimate_stem_line(render_stereo(world, T_BC, INTR), T_BC, cfg), base, d


def test_triangulated_stem_line_matches_truth():
    cfg = load_config(None, [])
    for tilt, az in ((0, 0), (25, 60), (45, 90), (55, 30)):
        est, base, d = _estimate(tilt, az, cfg)
        assert est.ok, (tilt, az, est.status)
        # root on the substrate (tray_top - 0.005 = 0.10 = stem base height in this world)
        assert np.linalg.norm(est.root_B[:2] - base[:2]) < 0.003, (tilt, az, est.summary())
        assert np.degrees(np.arccos(np.clip(est.dir_B @ d, -1, 1))) < 4.0, (tilt, az, est.summary())
        assert abs(est.diameter_m - 0.0024) < 0.0015


def test_stem_along_the_baseline_is_rejected_not_wrong():
    cfg = load_config(None, [])
    est, base, d = _estimate(35, -120, cfg)  # appears ~30 deg from horizontal in the image: ambiguous disparity
    if est.ok:
        assert np.linalg.norm(est.root_B[:2] - base[:2]) < 0.003
