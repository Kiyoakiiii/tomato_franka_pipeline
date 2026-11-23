"""Which cells hold a plug, from the overview picture (seedling.traygrid.occupied_cells), on pictures checked by eye.

2026-11-19: the old rule took empty cells for plugs when the camera, looking down at ~45 deg, saw the next cell's plug
through the empty cell (A7 behind B7 in two runs: the arm went for an empty cell first). Plugs on these pictures,
checked on the overlays: B6-B8, C6-C8 (the middle group) and E1-E3, F2-F4 (the left group)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from seedling.camera import load_frame_any
from seedling.config import PROJECT_DIR, load_config
from seedling.traygrid import occupied_cells

PLUGS = {"B6", "B7", "B8", "C6", "C7", "C8", "E1", "E2", "E3", "F2", "F3", "F4"}
# picture: the plugs that must be found (the others there may be missed: under leaves or too dark), the plugs there
EXACT = {"20261119_182415": PLUGS, "20261119_182113": PLUGS, "20261117_175955": PLUGS}
NO_FALSE = {"20261118_170238": PLUGS, "20261118_170500": PLUGS, "20261119_181102": PLUGS, "20261117_175925": PLUGS,
            "20261117_175904": PLUGS, "20261117_184718": PLUGS - {"B8"}}


def _run(pic):
    d = PROJECT_DIR / "experiments" / "tray" / pic
    if not (d / "frame.npz").exists():
        pytest.skip(f"picture {pic} not here")
    g = json.loads((d / "tray_grid.json").read_text())
    fr = load_frame_any(d / "frame.npz")
    return set(occupied_cells(fr, np.array(g["T_BC"]), g["cells"], g["z_top"], load_config()))


@pytest.mark.parametrize("pic", sorted(EXACT))
def test_plugs_found_exactly(pic):
    assert _run(pic) == EXACT[pic]


@pytest.mark.parametrize("pic", sorted(NO_FALSE))
def test_no_empty_cell_taken_for_a_plug(pic):
    assert _run(pic) <= NO_FALSE[pic]
