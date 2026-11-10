# 视觉模型

| 文件 | 用途与状态 |
| --- | --- |
| `stem_yolo11n_v3_clean_one_stem.pt` | 原有 YOLO11n 分割模型，保留原文件；旧训练数据划分未知 |
| `stem_yolo11n_seedlings_dev_v3_20261001.pt` | 160 轮重训的 YOLO11n；960 / 0.25 下开发平均 IoU 0.641、2/49 无输出 |
| `pretrained/` | 官方通用预训练权重及下载来源、哈希；不等于番茄训练结果 |

新候选在 186 张训练图上训练，使用 49 张按拍摄时间组留出的开发验证图。植株身份尚未确认，没有正式测试集。10 月 1 日三组训练和验证均已结束，见 [最终比较页](../experiments/vision/20261001_comparison_final/report.html)。新训练中 YOLO11n 最好，但其平均 IoU 0.641 仍低于旧模型在同批图上的 0.709，故当前保留旧权重作基线。旧模型的训练重叠未知，仍需新植株验证；没有模型已被验证可以执行机器人抓取。

本轮原始运行和中间权重保留在 `experiments/vision/`，新文件与验证过的最佳 checkpoint 校验一致。旧控制配置没有改为自动加载新模型。

详见 [训练与数据记录](../docs/archive/vision_training_20260930_CN.md) 和 [全部预测对照图](../experiments/vision/20260930_yolo26n_gpu_960_stable/prediction_review/review.html)。

2026-10-05 整理：删除了效果较差的 YOLO26n、YOLO11s 候选权重和多余的预训练权重（比较结果保留在 `experiments/vision/` 和 `docs/archive/vision_training_20260930_CN.md`）。保留当前在用的基线 `stem_yolo11n_v3_clean_one_stem.pt`、最好的重训候选 `stem_yolo11n_seedlings_dev_v3_20261001.pt` 及其预训练起点 `pretrained/yolo11n-seg.pt`。各训练 run 目录中的重复权重也已删除。
