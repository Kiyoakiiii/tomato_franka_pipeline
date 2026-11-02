"""Session / trial logging: JSONL events, saved frames, CSV rows matching experiments/templates."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import PROJECT_DIR, snapshot

TEMPLATES = PROJECT_DIR / "experiments" / "templates"


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


class Session:
    def __init__(self, root: str | Path, name: str, cfg: dict, extra: dict | None = None):
        self.dir = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot(cfg), sort_keys=False, allow_unicode=True))
        if extra:
            (self.dir / "session.json").write_text(json.dumps(extra, indent=2, default=_jsonable, ensure_ascii=False))

    def append_csv(self, template: str, row: dict) -> Path:
        cols = (TEMPLATES / template).read_text().strip().split(",")
        path = self.dir / template
        new = not path.exists()
        unknown = set(row) - set(cols)
        if unknown:
            raise KeyError(f"columns not in template {template}: {sorted(unknown)}")
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
        return path


class TrialLog:
    def __init__(self, session: Session, trial_id: str):
        self.trial_id = trial_id
        self.dir = session.dir / trial_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.t0 = time.time()

    def event(self, state: str, kind: str, **data) -> None:
        rec = {"t": round(time.time() - self.t0, 3), "wall": time.strftime("%H:%M:%S"), "state": state, "kind": kind, **data}
        with self.events.open("a") as fh:
            fh.write(json.dumps(rec, default=_jsonable, ensure_ascii=False) + "\n")
        print(f"  [{rec['t']:7.2f}s] {state:<15} {kind:<10} " + ", ".join(f"{k}={v}" for k, v in data.items() if k not in {"candidates"})[:200])

    def path(self, name: str) -> Path:
        return self.dir / name
