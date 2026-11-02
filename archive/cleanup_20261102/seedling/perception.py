"""One observation = grab -> segment -> centreline/3D fit -> grasp candidates."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import Frame
from .grasp import GraspCandidate, generate_candidates, scene_points
from .stem3d import InstanceGeometry, analyse_frame
from .viz import draw_perception, save_rgb


@dataclass
class Observation:
    frame: Frame
    instances: list
    geoms: list[InstanceGeometry]
    candidates: list[GraspCandidate]
    timings: dict = field(default_factory=dict)

    def best(self) -> GraspCandidate | None:
        return next((c for c in self.candidates if c.feasible), None)

    def rejection_summary(self) -> str:
        if not self.instances:
            return "no_stem_detected"
        if not any(s.ok for g in self.geoms for s in g.segments):
            reasons = [s.reject_reason.split(" (")[0] for g in self.geoms for s in g.segments] or [g.reject_reason for g in self.geoms]
            return "no_valid_3d_segment: " + ", ".join(sorted(set(reasons)))[:200]
        if not self.candidates:
            return "no_grasp_candidates_evaluated (no camera extrinsics or skipped)"
        if not any(c.feasible for c in self.candidates):
            reasons = sorted({c.reject_reason.split(" (")[0] for c in self.candidates})
            return "no_feasible_grasp: " + ", ".join(reasons)[:200]
        return ""

    def to_dict(self) -> dict:
        return {
            "frame_meta": self.frame.meta,
            "instances": [{"index": i.index, "conf": i.conf, "area": i.area, "box": i.box_xyxy} for i in self.instances],
            "segments": [s.to_dict() for g in self.geoms for s in g.segments],
            "instance_rejects": {g.instance: g.reject_reason for g in self.geoms if g.reject_reason},
            "candidates": [c.summary() for c in self.candidates[:30]],
            "timings_s": self.timings,
        }


def observe(camera, segmenter, cfg: dict, T_BC: np.ndarray | None, q_seed=None, n_average: int | None = None, frame: Frame | None = None,
            with_candidates: bool = True) -> Observation:
    t = {}
    t0 = time.time()
    if frame is None:
        frame = camera.grab(n_average=int(n_average or cfg["camera"].get("average_frames", 1)))
    t["grab"] = time.time() - t0
    t0 = time.time()
    instances = segmenter.predict_frame(frame)
    t["segment"] = time.time() - t0
    t0 = time.time()
    geoms = analyse_frame(frame, instances, cfg["perception"])
    t["geometry"] = time.time() - t0
    t0 = time.time()
    cands = generate_candidates(geoms, T_BC, cfg, scene_points(frame), q_seed=q_seed) if (T_BC is not None and with_candidates) else []
    t["grasp"] = time.time() - t0
    return Observation(frame, instances, geoms, cands, {k: round(v, 3) for k, v in t.items()})


def save_observation(obs: Observation, out_dir: Path, prefix: str, cfg: dict, T_BC=None, save_frame: bool = True, lines=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_frame:
        obs.frame.save(out_dir / f"{prefix}_frame", save_png=False)
    best = obs.best()
    info = [f"instances={len(obs.instances)} ok_segments={sum(s.ok for g in obs.geoms for s in g.segments)} "
            f"feasible={sum(c.feasible for c in obs.candidates)}"]
    if best is not None:
        info.append(f"best: p_B={np.round(best.p_B, 3).tolist()} yaw={best.yaw_offset_deg:+.0f} pitch={best.pitch_deg:.0f} d={best.diameter_m*1000:.1f}mm")
    elif obs.rejection_summary():
        info.append(obs.rejection_summary()[:90])
    img = draw_perception(obs.frame, obs.instances, obs.geoms, obs.candidates, T_BC, cfg, (lines or []) + info)
    save_rgb(out_dir / f"{prefix}_overlay.jpg", img)
    (out_dir / f"{prefix}_perception.json").write_text(json.dumps(obs.to_dict(), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
