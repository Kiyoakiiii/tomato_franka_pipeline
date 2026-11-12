"""Regression tests on recorded overview pictures (experiments/tray/<id>/frame.npz, not in git: from the lab's Google
Drive). Each labelled picture is located again with today's code (seedling.overview, what locate.py runs)
from the camera pose saved with it, and checked against tests/data/overview_labels.json:

- tray: the labelled (corner) cells within tol_mm (median), or the locate must fail when the label says so
  (tray not fully in view, rotated); "either": may fail, but if located it must be right;
- known_bug: the current code gets it wrong; an expected failure (strict: it reports when a fix makes it pass);
- disc: centre within tol_mm and the free holes as labelled, where the label has them.

Skipped when the pictures are not there. Slow (~5-40 s a picture): run them with
    python -m pytest -q tests/test_overview_replay.py          (or -k <id>)
and see tools/replay_overview.py for the whole set with a summary.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAY = ROOT / "experiments" / "tray"
LABELS = json.loads((Path(__file__).parent / "data" / "overview_labels.json").read_text(encoding="utf-8"))["pictures"]


def _have(fid: str) -> bool:
    return (TRAY / fid / "frame.npz").exists() and (TRAY / fid / "tray_grid.json").exists()


@pytest.fixture(scope="module")
def located():
    """picture id -> locate_overview result (each picture located once for all its tests)"""
    from seedling.camera import Frame
    from seedling.config import load_config
    from seedling.overview import locate_overview

    cfg = load_config()
    cache = {}

    def get(fid):
        if fid not in cache:
            saved = json.loads((TRAY / fid / "tray_grid.json").read_text())
            hint = LABELS[fid].get("z_hint", saved.get("z_top") if saved.get("ok") else None)
            with redirect_stdout(io.StringIO()):
                cache[fid] = locate_overview(Frame.load(TRAY / fid / "frame.npz"), np.array(saved["T_BC"]), cfg, z_hint=hint)
        return cache[fid]

    return get


def _median_mm(a: dict, b: dict) -> float:
    return float(np.median([np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in b])) * 1000


def _param(fid, part):
    bug = LABELS[fid][part].get("known_bug")
    return pytest.param(fid, marks=pytest.mark.xfail(reason=f"known bug: {bug}", strict=True)) if bug else fid


@pytest.mark.parametrize("fid", [_param(k, "tray") for k in sorted(LABELS)])
def test_tray(fid, located):
    if not _have(fid):
        pytest.skip("picture not here (experiments/ comes from the lab's Google Drive)")
    lab = LABELS[fid]["tray"]
    g = located(fid)["grid"]
    if lab["expect"] == "fail":
        assert not g.ok, f"located although the label says it cannot be ({lab.get('why', '')}): {g.status}"
        return
    if lab["expect"] == "either" and not g.ok:
        return
    assert g.ok, f"not located: {g.status}"
    d = _median_mm(g.cells, lab["cells"])
    assert d <= lab.get("tol_mm", 10.0), f"cells {d:.1f} mm from the labelled ones ({g.status})"


@pytest.mark.parametrize("fid", [_param(k, "disc") for k, v in sorted(LABELS.items()) if v.get("disc")])
def test_disc(fid, located):
    if not _have(fid):
        pytest.skip("picture not here")
    lab = LABELS[fid]["disc"]
    d = located(fid)["disc"]
    if lab["expect"] == "fail":
        assert not d.get("ok"), f"disc found although the label says it cannot be: {d.get('status')}"
        return
    assert d.get("ok"), f"disc not found: {d.get('status')}"
    e = float(np.hypot(*(np.array(d["centre"]) - np.array(lab["centre"])))) * 1000
    assert e <= lab.get("tol_mm", 10.0), f"disc centre {e:.1f} mm from the label"
    if "free_holes" in lab:
        assert sorted(d["free_holes"]) == sorted(lab["free_holes"])
