import numpy as np

from seedling.grasp import generate_candidates, scene_points
from seedling.stem3d import analyse_frame
from seedling.synthetic import GTSegmenter, default_sim_camera, default_world, render
from seedling.transforms import apply_T


def test_synthetic_stem_axis_localisation(cfg):
    T_BC, intr = default_sim_camera()
    errs = []
    for seed in range(4):
        world = default_world(seed, 1)
        f = render(world, T_BC, intr, seed=seed)
        geos = analyse_frame(f, GTSegmenter().predict_frame(f), cfg["perception"])
        s = world.stems[0]
        ok = [sg for g in geos for sg in g.segments if sg.ok]
        assert ok, "expected at least one valid segment on a clean synthetic stem"
        for sg in ok:
            r = apply_T(T_BC, np.array(sg.p_C)) - s.base_B
            errs.append(np.linalg.norm(r - (r @ s.dir_B) * s.dir_B))
    assert np.median(errs) < 0.001


def test_rgb_only_frame_is_rejected_not_guessed(cfg):
    T_BC, intr = default_sim_camera()
    f = render(default_world(0, 1), T_BC, intr)
    f.xyz[:] = np.nan
    geos = analyse_frame(f, GTSegmenter().predict_frame(f), cfg["perception"])
    assert not any(sg.ok for g in geos for sg in g.segments)
    assert generate_candidates(geos, T_BC, cfg, scene_points(f)) == []


def test_candidates_respect_workspace(cfg):
    T_BC, intr = default_sim_camera()
    f = render(default_world(0, 1), T_BC, intr)
    geos = analyse_frame(f, GTSegmenter().predict_frame(f), cfg["perception"])
    cfg["robot"]["workspace_m"]["x"] = [0.6, 0.75]  # stem at x~0.5 is now outside
    cands = generate_candidates(geos, T_BC, cfg, scene_points(f))
    assert cands and not any(c.feasible for c in cands)
    assert all("workspace" in c.reject_reason for c in cands)
