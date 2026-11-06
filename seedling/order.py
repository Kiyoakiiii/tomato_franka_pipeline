"""The order the plants are taken in (user's rule, 2026-11-04): the row nearest the robot first (its cells' mean
distance from the base), within a row from right to left as seen from the robot (smallest base-frame y first).
Cells whose only soil lies at the cell bottom (deeper than scrap_depth_m below the tray top) are soil scraps."""
from __future__ import annotations

import numpy as np


def plant_order(grid: dict, cells_with_plug, scrap_depth_m: float = 0.030) -> list[str]:
    top = float(grid["z_top"])
    soil = grid.get("soil_z_overview") or {}
    cells = grid["cells"]
    rows = {}
    for k in cells:
        rows.setdefault(k[0], []).append(float(np.hypot(*cells[k][:2])))
    row_dist = {r: float(np.mean(v)) for r, v in rows.items()}
    out = []
    for k in cells_with_plug:
        z = soil.get(k)
        if z is not None and z < top - scrap_depth_m:
            continue
        out.append((row_dist[k[0]], float(cells[k][1]), k))
    return [k for *_, k in sorted(out)]
