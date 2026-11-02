#!/usr/bin/env python3
"""Diagnose missing candidates versus ranking errors. Annotation-based oracle is NOT deployable."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_runtime import configure, digest, load_libraries, local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    weights, output = local(args.weights), local(args.output)
    output.mkdir(parents=True, exist_ok=False)
    YOLO = load_libraries()
    import numpy as np
    from PIL import Image, ImageDraw
    model = YOLO(str(weights))
    samples = [r for r in csv.DictReader(local('data/curated/seedlings_dev_v3/manifest.csv').open()) if r['split'] == 'val']
    rows = []
    for sample in samples:
        image, label = local(sample['derived_image']), local(sample['derived_label'])
        if digest(image) != sample['image_file_sha256'] or digest(label) != sample['derived_label_sha256']:
            raise ValueError('Changed development data')
        with Image.open(image) as source:
            w, h = source.size
        target = Image.new('1', (w, h))
        draw = ImageDraw.Draw(target)
        for line in label.read_text().splitlines():
            xy = list(map(float, line.split()[1:]))
            draw.polygon([(x*w,y*h) for x,y in zip(xy[::2],xy[1::2])], fill=1)
        gt = np.asarray(target, dtype=bool)
        result = model.predict(str(image), imgsz=960, conf=0.25, max_det=10, retina_masks=True,
                               device='cpu', verbose=False, save=False)[0]
        candidates = []
        if result.masks is not None:
            for mask, box in zip(result.masks.data.cpu().numpy(), result.boxes):
                pred = mask > 0.5
                if pred.shape != gt.shape:
                    raise ValueError('Native mask shape mismatch')
                union = int((gt | pred).sum())
                candidates.append({'confidence': float(box.conf), 'iou_with_annotation': int((gt & pred).sum())/union,
                                   'box_xyxy': box.xyxy[0].tolist()})
        selected = max(candidates, key=lambda c:c['confidence']) if candidates else None
        oracle = max(candidates, key=lambda c:c['iou_with_annotation']) if candidates else None
        rows.append({'sample_id':sample['sample_id'], 'capture_group':sample['capture_group'],
                     'candidate_count':len(candidates), 'selected_iou': selected['iou_with_annotation'] if selected else 0,
                     'oracle_iou_for_diagnosis_only': oracle['iou_with_annotation'] if oracle else 0,
                     'candidates':candidates})
    summary = {'weights':str(weights), 'weights_sha256':digest(weights), 'imgsz':960, 'conf':0.25,
               'frames':len(rows), 'selected_mean_iou':float(np.mean([r['selected_iou'] for r in rows])),
               'oracle_mean_iou_diagnostic_only':float(np.mean([r['oracle_iou_for_diagnosis_only'] for r in rows])),
               'frames_with_any_candidate_iou_ge_0_5':sum(r['oracle_iou_for_diagnosis_only'] >= 0.5 for r in rows),
               'selected_iou_lt_0_5_but_correct_candidate_available':sum(r['selected_iou'] < 0.5 <= r['oracle_iou_for_diagnosis_only'] for r in rows),
               'frames_without_candidate_iou_ge_0_5':sum(r['oracle_iou_for_diagnosis_only'] < 0.5 for r in rows),
               'warning':'Oracle uses ground-truth annotations solely to diagnose candidate coverage. It is not executable selection, independent test accuracy, or grasp success.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'per_image_candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
