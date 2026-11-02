#!/usr/bin/env python3
"""Compare completed reviews on identical development frames; generate an offline HTML report."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_runtime import digest, local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="append", required=True, help="Display name=project-local review directory")
    parser.add_argument("--candidate", action="append", required=True, help="Names eligible for fresh-model selection")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = local(args.output)
    models = []
    reference_ids = None
    for spec in args.review:
        name, value = spec.split("=", 1)
        path = local(value)
        summary = json.loads((path / "summary.json").read_text())
        rows = list(csv.DictReader((path / "per_image.csv").open()))
        ids = {r["sample_id"] for r in rows}
        if len(ids) != len(rows) or (reference_ids is not None and ids != reference_ids):
            raise ValueError("Reviews do not contain identical unique images")
        reference_ids = ids
        if summary["input_size"] != 960 or summary["confidence_threshold"] != 0.25 or not summary.get("retina_masks", False):
            raise ValueError("All primary reviews must use 960, conf 0.25 and native masks")
        if digest(local(summary["weights"])) != summary["weights_sha256"]:
            raise ValueError("Reviewed model changed")
        for r in rows:
            r["selected_mask_iou"] = float(r["selected_mask_iou"])
            r["predicted_instances"] = int(r["predicted_instances"])
        if abs(sum(r["selected_mask_iou"] for r in rows)/len(rows)-summary["mean_selected_mask_iou"]) > 1e-10:
            raise ValueError("Summary does not match per-image scores")
        training_path = path.parent / "summary.json" if path.name == "prediction_review" else None
        training = json.loads(training_path.read_text()) if training_path and training_path.is_file() else None
        models.append({"name": name, "review": str(path.relative_to(local('.'))),
                       "summary": summary, "training": training,
                       "selection_eligible": name in args.candidate, "rows": rows})
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("Duplicate model names")
    candidates = [m for m in models if m["selection_eligible"]]
    if len(candidates) != len(args.candidate):
        raise ValueError("Each candidate must match one review")
    ranked = sorted(candidates, key=lambda m: (-m["summary"]["mean_selected_mask_iou"], m["summary"]["no_detection_images"]))
    output.mkdir(parents=True, exist_ok=False)
    result = {"primary_confidence": 0.25, "imgsz": 960, "images": len(reference_ids),
              "selection_rule": "Highest mean selected-mask IoU, then fewer no-output frames; development selection only",
              "selected_fresh_candidate": ranked[0]["name"], "models": models,
              "independent_test": False,
              "limitations": ["Time groups are not verified plant identities", "Legacy model training overlap is unknown",
                              "Validation was used repeatedly for development selection", "IoU is not grasp success"]}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    table_rows = []
    galleries = []
    flat_rows = []
    for model in models:
        s = model["summary"]
        mean = s["mean_selected_mask_iou"]
        iou50 = sum(r["selected_mask_iou"] >= 0.5 for r in model["rows"])
        map95 = (model["training"] or {}).get("development_metrics", {}).get("metrics/mAP50-95(M)")
        view = os.path.relpath(local(model["review"]) / "review.html", output)
        table_rows.append(f'<tr><td><a href="{html.escape(view)}">{html.escape(model["name"])}</a></td>'
                          f'<td>{mean:.3f}</td><td>{s["no_detection_images"]}/{len(reference_ids)}</td>'
                          f'<td>{iou50}/{len(reference_ids)}</td><td>{map95:.3f}</td></tr>' if map95 is not None else
                          f'<tr><td><a href="{html.escape(view)}">{html.escape(model["name"])}</a></td>'
                          f'<td>{mean:.3f}</td><td>{s["no_detection_images"]}/{len(reference_ids)}</td>'
                          f'<td>{iou50}/{len(reference_ids)}</td><td>未测</td></tr>')
        flat_rows.append({"model": model["name"], "mean_selected_mask_iou": mean,
                          "no_detection_images": s["no_detection_images"], "iou_at_least_0_5_images": iou50,
                          "mask_map50_95": map95, "selection_eligible": model["selection_eligible"]})
        groups = ''.join(f'<tr><td>{g["capture_group"]}</td><td>{g["images"]}</td>'
                         f'<td>{g["mean_selected_mask_iou"]:.3f}</td><td>{g["no_detection"]}</td></tr>'
                         for g in s["capture_groups"])
        thresholds = ''.join(f'<tr><td>{t["threshold"]}</td><td>{t["mean_selected_mask_iou"]:.3f}</td>'
                             f'<td>{t["no_detection_images"]}</td></tr>' for t in s.get("threshold_diagnostics", []))
        galleries.append(f'<details><summary>{html.escape(model["name"])}：拍摄组与阈值诊断</summary>'
                         '<p>各阈值用于开发诊断；主比较仍固定为 0.25。拍摄组不是已确认的植株编号。</p>'
                         f'<table><tr><th>拍摄组</th><th>帧数</th><th>平均 IoU</th><th>无输出</th></tr>{groups}</table>'
                         f'<table><tr><th>阈值</th><th>平均 IoU</th><th>无输出</th></tr>{thresholds}</table></details>')
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    winner = ranked[0]
    legacy_models = [m for m in models if "legacy" in m["review"]]
    if legacy_models and winner["summary"]["mean_selected_mask_iou"] < legacy_models[0]["summary"]["mean_selected_mask_iou"]:
        recommendation = ('旧权重在这批图片上的平均 IoU 仍更高，本轮没有证明替换收益。'
                          '保留旧权重作基线，用未见过的新植株比较后再决定替换；新训练模型保留用于对照。')
    else:
        recommendation = '新候选的优势仅限这批开发图片；仍需未见过的新植株和真实负样本验证后决定替换。'
    cards = []
    for row in sorted(winner["rows"], key=lambda r: r["selected_mask_iou"]):
        sid = row["sample_id"]
        panels = []
        for model in models:
            mr = next(r for r in model["rows"] if r["sample_id"] == sid)
            src = os.path.relpath(local(model["review"]) / f"{sid}.jpg", output)
            panels.append(f'<figure><figcaption>{html.escape(model["name"])} · IoU {mr["selected_mask_iou"]:.3f}</figcaption>'
                          f'<a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}"></a></figure>')
        cards.append(f'<details class="sample" data-miss="{int(row["predicted_instances"] == 0)}">'
                     f'<summary>{sid} · {row["capture_group"]} · 当前候选 IoU {row["selected_mask_iou"]:.3f}</summary>'
                     + ''.join(panels) + '</details>')
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>番茄目标茎段模型对照</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1320px;margin:32px auto;padding:0 20px;color:#203028}'
                'table{border-collapse:collapse;margin:20px 0;width:100%}td,th{text-align:left;border-bottom:1px solid #d9e2dd;padding:10px}'
                'details{border:1px solid #d9e2dd;border-radius:8px;margin:12px 0;padding:12px}summary{cursor:pointer}img{width:100%}figure{margin:16px 0}'
                '.note{background:#fff5de;padding:16px;border-radius:8px}a{color:#155c42}</style>'
                '<h1>番茄目标茎段模型对照</h1><p class="note">这是开发验证，不是独立新植株测试。旧模型可能训练过这些照片；'
                '时间分组尚未确认植株身份。所有模型使用相同 49 帧、960 输入和 0.25 阈值。无输出帧按 IoU=0 计入。</p>'
                f'<p>本轮重新训练候选中，按预定平均 IoU 规则选择：<strong>{html.escape(winner["name"])}</strong>。尚未验证机器人抓取。</p>'
                f'<p class="note">{recommendation}</p>'
                '<table><tr><th>模型（点击看全部预测）</th><th>平均 mask IoU ↑</th><th>无输出 ↓</th><th>IoU ≥ 0.5 ↑</th><th>mask mAP50–95 ↑</th></tr>'
                + ''.join(table_rows) + '</table><p>“有输出”不等于找对目标；IoU 与 mAP 是不同指标，不是抓取成功率。</p>'
                + ''.join(galleries) + '<h2>逐图对照</h2><p>按本轮候选 IoU 从低到高排列；每幅图左为标注、右为预测。点击图片查看大图。</p>'
                '<label><input type="checkbox" id="misses">只显示本轮候选无输出的帧</label>' + ''.join(cards)
                + '<script>document.getElementById("misses").onchange=function(){document.querySelectorAll(".sample").forEach(e=>{e.hidden=this.checked&&e.dataset.miss!=="1"})}</script></html>')
    (output / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(output / "report.html"), "selected_fresh_candidate": winner["name"],
                      "mean_selected_mask_iou": winner["summary"]["mean_selected_mask_iou"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
