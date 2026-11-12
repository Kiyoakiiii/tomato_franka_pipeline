"""Close-up re-location of the cell to grasp (seedling/closeup.py).

- synthetic: a tray rendered as the ZED's depth would see it (wall tops, cell bottoms, table), at any yaw; the
  overview cells given a few mm and a degree off; the close-up must bring the cell back to the truth (the only test
  with the true position known);
- recorded close-ups (experiments/grasp/closeups/, from the lab's Google Drive; skipped when not there): the same
  answer from different starting guesses within the window, the same cell seen again from another pose agreeing,
  and a picture without depth refused;
- grasp_xy: when the refined centre is used (config grasp.closeup_xy) and when not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seedling.camera import Frame, Intrinsics
from seedling.closeup import grasp_xy, refine_cell
from seedling.config import load_config
from seedling.synthetic import look_at

ROOT = Path(__file__).resolve().parents[1]
CU = ROOT / "experiments" / "grasp" / "closeups"
TRAY = ROOT / "experiments" / "tray"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tray_cells(origin, yaw_deg, z_top, p=0.0448, rows=6, cols=12) -> dict:
    a = np.radians(yaw_deg)
    ec = np.array([np.cos(a), np.sin(a)])      # along the columns (1 -> 12)
    er = np.array([np.sin(a), -np.cos(a)])     # along the rows (A -> F)
    o = np.asarray(origin, float)              # the inner corner before A1
    return {f"{chr(65 + r)}{c + 1}": [*(o + (r + 0.5) * p * er + (c + 0.5) * p * ec), z_top] for r in range(rows) for c in range(cols)}


def render_tray(cells: dict, T_BC, z_top, p=0.0448, wall=0.008, rim=0.006, depth=0.045, table=0.06, w=960, h=540, f=530.0, seed=0):
    """depth (camera frame) of a tray seen from T_BC: wall tops and rim at z_top, cell bottoms 45 mm lower, table 60 mm lower"""
    intr = Intrinsics(f, f, w / 2, h / 2, w, h)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack([(u - intr.cx) / f, (v - intr.cy) / f, np.ones_like(u, float)], -1)
    R, o = T_BC[:3, :3], T_BC[:3, 3]
    db = dc @ R.T
    A1, A2, B1 = (np.array(cells[k][:2]) for k in ("A1", "A2", "B1"))
    ec, er = (A2 - A1) / p, (B1 - A1) / p
    corner = A1 - 0.5 * p * (ec + er)

    def hit(z):
        t = (z - o[2]) / db[..., 2]
        return t, o[:2] + t[..., None] * db[..., :2]

    t_top, q = hit(z_top)
    d = q - corner
    fr, fc = d @ er / p, d @ ec / p
    on_tray = (fr > -rim / p) & (fr < 6 + rim / p) & (fc > -rim / p) & (fc < 12 + rim / p)
    inner = (fr > 0) & (fr < 6) & (fc > 0) & (fc < 12)
    on_wall = on_tray & (~inner | (np.abs(fr - np.round(fr)) * p < wall / 2) | (np.abs(fc - np.round(fc)) * p < wall / 2))
    t_bot, _ = hit(z_top - depth)
    t_tab, _ = hit(z_top - table)
    t = np.where(on_wall, t_top, np.where(on_tray, t_bot, t_tab))
    xyz = (dc * t[..., None]).astype(np.float32)
    xyz += np.random.default_rng(seed).normal(0, 0.0005, xyz.shape).astype(np.float32)
    rgb = np.full((h, w, 3), 110, np.uint8)  # plain grey: neither plant nor peat
    return Frame(rgb, xyz, None, intr)


@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 90.0, -120.0])
@pytest.mark.parametrize("cell", ["D4", "A1", "F12", "C9"])
def test_synthetic_tray_any_yaw(cfg, yaw_deg, cell):
    truth = tray_cells([0.55, 0.20], yaw_deg, 0.12)
    c = np.array(truth[cell])
    T_BC = look_at(c + [-0.20, 0.08, 0.33], c)  # ~0.4 m away, looking down ~55 deg
    frame = render_tray(truth, T_BC, 0.12)
    # the overview's fit: 7 mm and 1 deg off (about the tray centre), its top 8 mm low
    a = np.radians(1.0)
    Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pivot = np.mean([v[:2] for v in truth.values()], axis=0)
    guess = {k: [*(pivot + Rz @ (np.array(v[:2]) - pivot) + [0.005, -0.005]), 0.112] for k, v in truth.items()}
    off0 = np.hypot(*(np.array(guess[cell][:2]) - c[:2])) * 1000
    r = refine_cell(frame, T_BC, guess, cell, cfg)
    assert r["ok"], r["status"]
    err = np.hypot(*(np.array(r["xy"]) - c[:2])) * 1000
    assert err < 1.0, f"{err:.2f} mm from the true centre (the guess was {off0:.1f} mm off)"
    assert abs(r["z_top"] - 0.12) < 0.002


@pytest.mark.parametrize("cells_off", [1.0, 0.7])
def test_synthetic_never_moves_a_whole_cell(cfg, cells_off):
    """The close-up moves the cell by less than half a cell, whatever the guess. The cells look alike, so an overview
    a whole cell off (1.0) is NOT found here: it is matched onto the cell it names and stays a cell off (catching
    that is the overview's job: the outline cross-check). A guess beyond the window (0.7 cell) is not pulled onto
    the neighbour either."""
    truth = tray_cells([0.55, 0.20], 15.0, 0.12)
    c = np.array(truth["D4"])
    T_BC = look_at(c + [-0.20, 0.08, 0.33], c)
    frame = render_tray(truth, T_BC, 0.12)
    ec = (np.array(truth["A2"][:2]) - np.array(truth["A1"][:2]))
    guess = {k: [v[0] + ec[0] * cells_off, v[1] + ec[1] * cells_off, v[2]] for k, v in truth.items()}
    r = refine_cell(frame, T_BC, guess, "D4", cfg)
    moved = np.hypot(*(np.array(r["xy"]) - np.array(guess["D4"][:2])))
    assert moved < 0.5 * 0.0448, f"moved {moved * 1000:.0f} mm from the guess"
    if r["ok"]:
        # whatever it matched, it is on a cell centre of the true grid (to 1 mm): never between two cells
        d = min(np.hypot(*(np.array(r["xy"]) - np.array(v[:2]))) for v in truth.values())
        assert d < 0.001 or cells_off != 1.0


# ---------------------------------------------------------------- recorded close-ups (lab data) -------------------

def _recorded(cid: str):
    """the close-up, its camera pose (the calibration of the overview fit used then) and that fit's cells"""
    from seedling import kinematics
    from seedling.calibration import load_T_FC
    from seedling.transforms import inv_T

    f = CU / f"{cid}.npz"
    if not f.exists():
        pytest.skip("close-up not here (experiments/ comes from the lab's Google Drive)")
    t = cid.split("_observe_")[1]
    ovs = [d for d in sorted(TRAY.iterdir()) if d.name < t and d.name[:8] == t[:8] and (d / "tray_grid.json").exists()]
    ovs = [d for d in ovs if json.loads((d / "tray_grid.json").read_text()).get("ok")]
    s = json.loads((ovs[-1] / "tray_grid.json").read_text())
    keep = kinematics.JOINT_OFFSETS.copy()
    dq = np.asarray(load_T_FC(load_config())[1].get("joint_offsets_rad") or np.zeros(7), float)  # unchanged since 2026-10-20
    kinematics.set_joint_offsets(keep)  # the module-wide offsets left as they were (other tests use the nominal model)
    fr = Frame.load(f)
    T_FC = inv_T(kinematics.fk_flange(np.array(s["q"]), offsets=dq)) @ np.array(s["T_BC"])
    return fr, kinematics.fk_flange(np.array(fr.meta["q"]), offsets=dq) @ T_FC, s["cells"]


@pytest.mark.parametrize("cid", ["D4_observe_20261102_113424", "C9_observe_20261105_174657", "C8_observe_20261106_174352",
                                 "C9_observe_20261106_180615"])
def test_recorded_same_answer_from_other_guesses(cfg, cid):
    fr, T_BC, cells = _recorded(cid)
    cell = cid.split("_")[0]
    r0 = refine_cell(fr, T_BC, cells, cell, cfg)
    assert r0["ok"], r0["status"]
    for dx, dy in ((0.006, -0.004), (-0.008, 0.003), (0.0, 0.009)):
        moved = {k: [v[0] + dx, v[1] + dy, v[2]] for k, v in cells.items()}
        r = refine_cell(fr, T_BC, moved, cell, cfg)
        assert r["ok"], r["status"]
        d = np.hypot(*(np.array(r["xy"]) - np.array(r0["xy"]))) * 1000
        assert d < 1.5, f"started {dx * 1000:+.0f},{dy * 1000:+.0f} mm off: {d:.1f} mm from the first answer"


@pytest.mark.parametrize("ids", [["D4_observe_20261102_113424", "D4_observe_20261102_114049", "D4_observe_20261102_114332"],
                                 ["C9_observe_20261105_161557", "C9_observe_20261105_162044", "C9_observe_20261105_163028",
                                  "C9_observe_20261105_164038"]])
def test_recorded_repeatable_across_poses(cfg, ids):
    """the same cell (tray not moved) seen from different observation poses: the refined centres agree"""
    xy = []
    for cid in ids:
        fr, T_BC, cells = _recorded(cid)
        r = refine_cell(fr, T_BC, cells, cid.split("_")[0], cfg)
        assert r["ok"], f"{cid}: {r['status']}"
        xy.append(r["xy"])
    xy = np.array(xy) * 1000
    spread = np.hypot(*(xy - xy.mean(0)).T).max()
    assert spread < 3.0, f"refined centres up to {spread:.1f} mm from their mean"


def test_recorded_no_depth_refused(cfg):
    fr, T_BC, cells = _recorded("C9_observe_20261106_152552")  # the camera gave no depth that time
    r = refine_cell(fr, T_BC, cells, "C9", cfg)
    assert not r["ok"] and "depth" in r["status"]


# ---------------------------------------------------------------- grasp_xy ------------------------------------------

def _obs(ok=True, shift=(3.0, -4.0), cell=(0.70, 0.10, 0.18)):
    return {"overview_cell": list(cell), "closeup_xy": {"ok": ok, "status": "ok" if ok else "no clear match", "shift_mm": list(shift),
                                                        "xy": [cell[0] + shift[0] / 1000, cell[1] + shift[1] / 1000]}}


def test_grasp_xy_modes():
    cell = [0.70, 0.10, 0.18]
    log = {"grasp": {"closeup_xy": "log"}}
    app = {"grasp": {"closeup_xy": "apply", "closeup_max_shift_mm": 15.0}}
    assert grasp_xy(_obs(), log, cell)[0] is None  # recorded only
    xy, _ = grasp_xy(_obs(), app, cell)
    assert np.allclose(xy, [0.703, 0.096])
    assert grasp_xy(_obs(ok=False), app, cell)[0] is None  # not sound
    assert grasp_xy(_obs(shift=(12.0, 12.0)), app, cell)[0] is None  # too far
    assert grasp_xy(_obs(), app, [0.71, 0.10, 0.18])[0] is None  # made for another tray fit
    assert grasp_xy(None, app, cell)[0] is None and grasp_xy({}, app, cell)[0] is None
