#!/usr/bin/env python
"""ZED live view + RGB-D capture (no robot). Optional live stem perception overlay.

Keys:  s = save frame (npz with XYZ/confidence + png)   p = toggle perception overlay   q = quit
Saved frames go to data/captures/<session>/ and a row is appended to vision_frames.csv.

    .venvs/pipeline/bin/python scripts/10_capture.py --plant-id P01 --perception
"""
from __future__ import annotations

import _common  # noqa: F401  (first: caps math threads, sets project paths)

import csv
import time

import cv2
import numpy as np

from _common import base_parser, config_from_args

from seedling.config import PROJECT_DIR


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--plant-id", default="")
    ap.add_argument("--role", default="unassigned", help="train/val/test/calibration/... (vision_frames.assigned_role)")
    ap.add_argument("--perception", action="store_true", help="run YOLO + 3D fitting on the live view")
    ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--average", type=int, default=1, help="frames averaged per saved capture")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from seedling.camera import ZedCamera
    from seedling.viz import draw_perception

    cam = ZedCamera(cfg)
    out = PROJECT_DIR / "data" / "captures" / args.session
    out.mkdir(parents=True, exist_ok=True)
    seg = T_BC = None
    show_perc = args.perception
    if args.perception:
        from seedling.segmentation import StemSegmenter

        seg = StemSegmenter(cfg)
    try:
        from seedling.calibration import load_T_BC

        T_BC, _ = load_T_BC(cfg)
    except FileNotFoundError:
        pass
    csv_path = out / "vision_frames.csv"
    cols = (PROJECT_DIR / "experiments" / "templates" / "vision_frames.csv").read_text().strip().split(",")
    n = 0
    print(f"saving to {out}")
    try:
        while True:
            f = cam.grab(1)
            valid = np.isfinite(f.depth).mean()
            lines = [f"plant={args.plant_id or '-'} saved={n} valid_depth={valid*100:.0f}%  [s]ave [p]erception [q]uit"]
            img = f.rgb
            if show_perc and seg is not None:
                from seedling.perception import observe

                obs = observe(None, seg, cfg, T_BC, frame=f, with_candidates=T_BC is not None)
                img = draw_perception(f, obs.instances, obs.geoms, obs.candidates, T_BC, cfg, lines + [f"ok segments={sum(s.ok for g in obs.geoms for s in g.segments)}"])
            else:
                img = draw_perception(f, [], [], lines=lines)
            cv2.imshow("zed", cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1280, int(1280 * img.shape[0] / img.shape[1]))))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):
                show_perc = not show_perc and seg is not None
            if k == ord("s"):
                fs = cam.grab(args.average) if args.average > 1 else f
                fid = time.strftime("%Y%m%d_%H%M%S") + f"_{n:03d}"
                fs.meta.update(plant_id=args.plant_id, role=args.role)
                path = fs.save(out / fid)
                new = not csv_path.exists()
                with csv_path.open("a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    if new:
                        w.writeheader()
                    w.writerow({"frame_id": fid, "plant_id": args.plant_id, "session_id": args.session, "assigned_role": args.role,
                                "rgb_path": str(path.with_suffix(".png").relative_to(PROJECT_DIR)),
                                "depth_path": str(path.relative_to(PROJECT_DIR)), "camera_serial": fs.meta.get("serial", "")})
                n += 1
                print(f"saved {path.name}  valid depth {np.isfinite(fs.depth).mean()*100:.1f}%")
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
