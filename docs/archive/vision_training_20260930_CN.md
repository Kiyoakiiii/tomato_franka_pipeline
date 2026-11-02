> 归档文档（2026-09-30 / 10-01 的阶段记录）。其中提到的旧脚本、src/、config.json 已移至 `archive/legacy_v0/`，部分工具和候选权重已删除。

# 数据整理与 YOLO26 分割训练

> 2026-10-01 后续三组训练、验证和导出均已完成。见 [最终比较页](../experiments/vision/20261001_comparison_final/report.html)、[可导出的比较图](../experiments/vision/20261001_comparison_final/figures/development_comparison.png) 和 [选型记录](../experiments/vision/20261001_comparison_final/selection.json)。当前保留旧权重作基线；本轮尚未证明替换收益。

## 最终开发比较

所有下表主指标使用相同 49 张开发图、完整原图视野、960 输入、0.25 阈值、最高置信度候选和原图尺寸 mask。无输出计入 IoU=0；“有输出”不等于找到正确目标。

| 模型 | 本轮训练 | 平均 mask IoU | 无输出 | IoU≥0.5 帧数 | mask mAP50 / mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| 原有 YOLO11n | 旧训练历史不完整 | 0.7092 | 1/49 | 43/49 | 未测 |
| 9 月 30 日 YOLO26n | 42 完整轮 + 部分第 43 轮 | 0.4370 | 15/49 | 27/49 | 0.6506 / 0.2709 |
| 重训 YOLO11n | 160 完整轮 | 0.6406 | 2/49 | 41/49 | 0.8477 / 0.3640 |
| 重训 YOLO26n | 23 完整轮后恢复 137 轮，总计 160 | 0.4469 | 14/49 | 26/49 | 0.6828 / 0.3184 |
| 追加 YOLO11s | 67 完整轮，40 轮未改善后早停 | 0.5930 | 0/49 | 34/49 | 0.6715 / 0.2964 |

新训练中选 YOLO11n 作后续新植株对照候选，但当前不替换原有基线。旧权重可能训练过这些图片；时间组也不等于确定的植株身份，所以不能据此断言某种架构的泛化能力更好。本轮只用一个种子，YOLO26 发生过断点恢复，容量追加又改变了物理 batch；这些均是开发试验，不是架构的严格统计排名。

预定的 1280 输入诊断分别得到平均 IoU：旧模型 0.7088、上轮 YOLO26n 0.4545、重训 YOLO11n 0.5852、重训 YOLO26n 0.4283、YOLO11s 0.4777。没有新候选达到预定的分辨率采用条件，保持 960 输入。降低 YOLO26 的阈值到 0.05 后，其平均 IoU 为 0.5395、仍有 7 帧无输出，依然没有超过重训 YOLO11n；这些阈值结果仅作开发诊断。

候选检查使用真实标签来计算“如果能选对候选”的诊断上界，不能用于实际选择或冒充可部署成绩。YOLO11n 为 0.6608，只有 1 张所选 IoU<0.5 而另有候选达到 0.5；YOLO26n 为 0.4625，同类帧 1 张；YOLO11s 为 0.6547，同类帧 8 张。YOLO11s 虽然每张都有输出，仍存在目标排序和轮廓覆盖问题。错误、漏检与不理想实验均保留。

本轮 GPU 训练串行运行，PyTorch 峰值预留显存：YOLO11n 6728 MiB、YOLO26n 7616 MiB、YOLO11s 4864 MiB。YOLO11n 耗时约 1799 秒；YOLO26 恢复段约 1647 秒，前段另有约 304 秒已完成轮次记录；YOLO11s 约 875 秒。未触发显存/主机内存保护退出。中断、恢复、预训练来源、环境、参数和工具快照均保存在各 run 中。所有训练进程现已结束。

现有标签已做重复实例修正，但**没有按新标注语义全面重标**，也没有新植株独立测试；规范和采集记录模板已准备。下一步应在实际工作距离和背景下补采未见过的植株，记录真实 plant_id，统一主茎端点规则，并纳入无目标负样本，再决定是否替换旧模型。分割结果不证明三维定位、抓取或损伤指标已经达标。

## 10 月 1 日：扩大资源后的对照方法

用户明确允许当前空闲时段使用最多 80% CPU/GPU。新增 `shared80` 配置：本进程及子线程限定在 20 个逻辑 CPU 中的 16 个，计算库线程数为 4；GPU 分配器上限为总显存的 70%，保留上下文余量，并在批次间暂停来近似控制 80% 活跃时间。它不是硬件利用率配额，也不会为了占满额度而额外制造计算。全卡空闲显存或系统可用内存低于总量的 20% 时，只让本训练在批次边界保存退出。没有更改既有进程、驱动、优先级或系统设置。

对照从官方通用 COCO 权重分别重训 YOLO11n-seg 和 YOLO26n-seg，不接着旧番茄权重训练。参数在首轮结果产生前保存：同一 186/49 划分、960 输入、batch 8、nbs 16、AdamW lr0=0.001、lrf=0.05、cosine schedule、3 轮预热、bias warmup=0、seed 42、mask_ratio=2，最多 160 轮/每组 60 分钟、patience 40。保留同样的轻度增强，关闭 mosaic/mixup/copy-paste。各模型的结构、内置损失和官方预训练不同；这里只是相同训练配方的一次开发对照，不是充分调参后的架构结论。

主评估固定原图尺寸 mask、置信度 0.25、最高置信度候选；无输出计 IoU=0。用平均所选 mask IoU 在新训练候选间排序，同时报告无输出数、IoU≥0.5 帧数、拍摄组结果和 mask mAP。额外记录 0.05/0.1/0.4/0.5 阈值诊断，但不以降低阈值后的“有输出”冒充正确识别。旧模型的新评估工具在 0.25 下逐张复现之前的 49 帧 IoU、像素精确率/召回率和预测数。

另有预先记录的 [1280 输入诊断](../configs/vision_resolution_diagnostic_20261001.json)，保持原始全图，不按真实标签裁剪。它属于开发调参；主表仍使用 960 输入。只有所选新模型在 1280 下平均 IoU 至少增加 0.02、且无输出数不增加，才建议采用更高输入尺寸。

训练器读取无损 PNG 的开销较明显，因此在派生 images 中添加逐像素核对的原始分辨率 BGR `.npy` 缓存，约 1.36 GiB；这不是深度数据。Ultralytics 在 cache=False 时也会读取已存在的同名 NPY。第一个训练中途开始使用相同像素的缓存，有单独的 `pixel_cache_event.json` 记录；后续训练在启动时校验缓存清单及文件哈希。未改变原图、标签、随机增强或数据划分。

队列 `tools/run_vision_comparison.py` 串行完成训练、逐图检查、校验导出和比较页，过程写入项目内事件日志，界面连接中断后先读状态，避免重复启动。原有旧权重继续保留。当前只有一个随机种子，且时间组不是确定的植株编号；任何结果仍需新植株验证。

新建的 `tools/predict_stem_offline.py` 可对一张本地 RGB 文件输出预测 mask、叠图和 JSON。默认 CPU、960 输入、0.25 阈值，不导入旧机器人代码、不访问相机。它的输出只是二维候选，不包含三维定位或可执行抓取姿态。

```bash
.venvs/vision-cpu/bin/python -I -B tools/predict_stem_offline.py \
  --weights models/stem_yolo11n_v3_clean_one_stem.pt \
  --image data/curated/seedlings_dev_v3/images/val/20260423_153449_073.png \
  --output experiments/vision/my_new_offline_check
```

本轮标注和新植株记录规则见 [数据说明](../data/README.md)。现有标签没有按模型预测重写；如果改为完整可见主茎，应另建数据版本，不能把新的标注语义混入本轮分数。

第一组 YOLO11n 已完整训练 160 轮，耗时约 1799 秒，最佳权重的开发 mask mAP50=0.8477、mAP50–95=0.3640；960 输入、阈值 0.25 下原图 mask 平均 IoU=0.6406，2/49 无输出，41/49 达到 IoU≥0.5。额外的 1280 输入使平均 IoU 降至 0.5852，虽然无输出减少至 1/49，仍不建议采用。候选诊断发现，8 张所选 IoU<0.5 的图中只有 1 张存在另一候选达到 0.5；使用真实标签选最优候选的诊断上界为 0.6608，**不是可部署结果**。

第二组 `20261001_yolo26n_matched160` 在完整记录第 23 轮后、第 24 轮早期意外终止；命令会话返回 143，日志未见 OOM traceback，来源未确定。未执行停止其他进程的命令。原 run、日志、best/last 权重保留，终止情况另记 `interruption.json`。

在新目录 `20261001_yolo26n_matched160_resumed` 从完整第 23 轮 last.pt 继续，总目标仍为 160 轮，保持数据和优化设置不变。复制 checkpoint 后仅调整输出位置，恢复 optimizer、EMA、学习率进度和早停最佳记录；来源 checkpoint 哈希和完成轮数写入 `resume_info`。Ultralytics 不保存完整随机数状态，因此不声称逐位复现不中断的训练轨迹。新目录中的 results.csv 先保留前 23 轮，再追加恢复轮次；其 time 列在恢复阶段重新计时，画学习曲线使用 epoch 列。恢复队列用独立后台会话运行，日志为 `.tmp/20261001_detached_resume.log`。

## 9 月 30 日历史结果

这次按用户确认的“单株工作区域中的目标茎段”继续实施。已完成数据导出、CPU/GPU 独立环境、两轮 GPU 开发训练和逐图验证，训练进程均已结束。当前交付的是待新植株验证的候选模型，尚不能确认优于旧权重。所有新增环境、缓存、模型和日志位于本项目内；未连接相机、机械臂或夹爪。

候选权重：[stem_yolo26n_seedlings_dev_v3_20260930.pt](../models/stem_yolo26n_seedlings_dev_v3_20260930.pt)，附 [完整来源与验证记录](../models/stem_yolo26n_seedlings_dev_v3_20260930.metadata.json)。查看 [49 张预测对照图](../experiments/vision/20260930_yolo26n_gpu_960_stable/prediction_review/review.html)。权重副本与已验证 checkpoint 的 SHA256 一致；原模型和旧控制配置保留。

| 实验 | 输入 / 训练情况 | 开发 mask mAP50 | 开发 mask mAP50–95 | 固定阈值 0.25 的平均所选 mask IoU | 无输出帧 |
| --- | --- | --- | --- | --- | --- |
| YOLO26s CPU 初训 | 640 / 5 完整轮 + 第 6 部分轮 | 0.1821 | 0.0496 | 0.0000 | 49/49 |
| YOLO26s GPU 首轮 | 960 / 29 完整轮 + 第 30 部分轮 | 0.2845 | 0.1008 | 0.1958 | 27/49 |
| YOLO26n GPU 修正配置 | 960 / 42 完整轮 + 第 43 部分轮 | 0.6506 | 0.2709 | 0.4370 | 15/49 |
| 旧 YOLO11n 权重诊断 | 960 / 旧训练数据与实际轮数未知 | 未测 | 未测 | 0.7092 | 1/49 |

这些数值来自同一组 49 帧开发图片，植株身份尚未确认；旧模型还可能训练过这些图片。模型容量、训练时长、批次及优化设置不同，因此这张表记录当前产物，**不是公平的架构排名或独立测试**。mAP 与原图尺寸 IoU 是不同统计量，也都不是抓取成功率。

## 数据版本与适用范围

当前训练入口为 [seedlings_dev_v3/data.yaml](../data/curated/seedlings_dev_v3/data.yaml)，说明见 [数据版本 README](../data/curated/seedlings_dev_v3/README.md)。235 张有标签的四月图片分为 186 张训练、49 张开发验证。验证组在训练前固定，按连续拍摄组划分。6 张五月图片与 3 张误拍排除；另 3 张含植物但缺标注的图片待补标。

三个双实例标签经原图复核后，确认是同一目标茎的重复轮廓。派生标签保留原始第一个轮廓，记录来源和原因；没有改动原始标签，也没有把这一步说成边界精修。原标注、复制图像与修正记录已通过 [检查工具](../tools/check_seedlings_export.py) 验证。

时间组不是植株身份；同株跨组再次出现仍可能造成数据泄漏。49 帧也不是 49 株独立植物，因此本轮结果只用于调试和比较开发版本，不能用于论文的独立泛化成绩。正式测试应采新的、有明确植株编号的实验室数据。

保留完整 1920×1080 原图，训练器按输入尺寸缩放。没有按真实 mask 裁剪验证图片。标签表示前景目标茎段，不能用本轮结果声称已识别画面中所有茎或已选择可抓位置。深度没有输入分割模型；原 RGB-D 数据保留，但低质量深度仍是后续三维定位的独立问题。

## 模型与环境

采用官方 COCO 预训练 **YOLO26s-seg** 进行微调；没有从旧项目权重接着训练。YOLO26 已有分割实现，但较新架构不保证在这些幼苗上比旧模型更准确。[官方模型文档](https://docs.ultralytics.com/models/yolo26/)

原权重 `models/stem_yolo11n_v3_clean_one_stem.pt` 保留不变，其元数据实际是 YOLO11n-seg。旧训练划分未知，不能拿它对当前验证集跑一次就宣称公平比较。后续应从通用预训练权重，用相同数据划分重训 YOLO11 基线。

旧权重在本次 49 张照片上的诊断复测：960 输入、阈值 0.25，48 张有输出，平均所选 mask IoU 为 0.7092、中位数 0.7572，20 张有多个候选。来源训练数据未知，图片可能已被它见过，不能与新划分重训结果作独立泛化排名。[旧模型逐图检查](../experiments/vision/20260930_legacy_yolo11_diagnostic/review.html)

读取旧 checkpoint 中保留的参数发现：请求的训练上限为 90 epochs、batch 8、imgsz 960、optimizer auto、nbs 64、warmup 3 epochs、warmup_bias_lr 0、mask_ratio 4。这些是保存的设置，不等于已证明完成 90 轮；没有按其中的旧数据路径访问外部文件。[旧训练参数记录](../experiments/vision/20260930_legacy_training_settings/checkpoint_training_settings.json)

| 项目 | 已使用版本或位置 |
| --- | --- |
| Python | 3.12.3，项目内 `.venvs/vision-cpu` |
| PyTorch / torchvision | 2.9.1+cpu / 0.24.1+cpu |
| Ultralytics | 8.4.46 |
| NumPy | 2.2.6 |
| 完整环境版本 | [requirements-vision-cpu.lock.txt](../requirements-vision-cpu.lock.txt)，每个 run 也保留 freeze 文件 |
| 新预训练权重 | `models/pretrained/yolo26s-seg.pt` |
| 权重来源与本地 SHA256 | `models/pretrained/yolo26s-seg.source.json` |
| 配置和缓存 | 项目内 `.config/`、`.cache/`、`.tmp/` |

CPU 版 PyTorch 按官方版本配对安装到独立环境，没有更改旧机器人环境。[PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/)

训练入口在导入计算库前设置 CPU affinity，仅允许本进程使用 2 个逻辑 CPU，并限制 PyTorch 线程数为 2、OpenCV 为 1。使用 batch 2、workers 0、无全数据缓存、CPU 专用依赖；关闭在线同步和第三方实验跟踪。首次运行仅下载官方预训练文件与所需依赖、字体，不上传本地训练图片。

## 已执行的实验

冒烟检查：8 张训练、2 张验证，320 输入、1 epoch；训练与最终验证共约 7.2 秒，进程峰值内存约 1.5 GB。此检查确认读取、训练、保存与验证路径可运行，零检测分数不用于模型能力结论。

全量初训：186 张训练、49 张开发验证，640 输入、最多 12 epochs，同时设 15 分钟训练预算。预算在本任务的批次边界触发正常退出并保存权重，不向系统其他进程发送信号；最后一轮可能只完成部分批次，保存和最终验证另需时间。

采用 AdamW、初始学习率 0.001、batch 2、`nbs=2`、`mask_ratio=2`、seed 42。禁用 mosaic/mixup/copy-paste，使用轻微色彩、平移、缩放、旋转和水平翻转，避免拼图增强破坏当前单株目标语义。完整配置在各 run 的 `run_metadata.json` 和 `train/args.yaml` 中；不是已证明最优的参数。

排查时发现首轮显式 AdamW 配置沿用了默认 `warmup_bias_lr=0.1`。本地 8.4.46 的 `BaseTrainer.build_optimizer` 在自动选择 AdamW 时将其设为 0，并提示 Adam 不应使用高于 0.01 的偏置预热学习率。这可能影响稳定性，但没有受控实验就不能将全部误差归因于它。后续工具默认值已修正为 0；首轮 run 的原始设置、权重和结果保留，下面的复现命令显式传入 0.1 以重现历史配置。

CPU 初训已完成 5 个完整 epoch 和第 6 个部分 epoch，触发 15 分钟训练预算后保存并验证，总耗时约 945.8 秒，峰值内存 2554 MiB。最佳 checkpoint 的最终开发 mask mAP50 为 0.1821、mAP50-95 为 0.0496；这个短程结果尚不足以使用。证据位于 [CPU summary.json](../experiments/vision/20260930_yolo26s_cpu_pilot/summary.json)。

另以固定置信度阈值 0.25、640 输入检查全部 49 张 CPU 初训验证图，49 张均无输出，平均目标 mask IoU 为 0。这与 mAP 非零不矛盾：mAP 汇总了更低阈值的候选，不能把它当成默认使用阈值下的可用检测率。完整叠图与逐图记录见 [CPU 预测检查](../experiments/vision/20260930_yolo26s_cpu_pilot/prediction_review/review.html)。该结果已保留，没有为了提高数字删除漏检帧或临时降低阈值。

GPU 960 输入冒烟检查已成功，AMP 检查通过，训练/验证/保存总耗时约 31.6 秒，PyTorch 峰值预留显存 2242 MiB。随后完成 YOLO26s 全量初训：960 输入、batch 2，29 个完整 epoch 加第 30 个部分 epoch，达到 30 分钟预算后保存退出，总耗时 1815.2 秒，峰值预留显存 2968 MiB。最佳权重最终开发 mask mAP50 为 0.2845，mAP50-95 为 0.1008；固定阈值 0.25 下漏检 27/49 张，平均所选 mask IoU 为 0.1958，尚不能替换旧模型。全部记录位于 [YOLO26s 初训](../experiments/vision/20260930_yolo26s_gpu_960/)，逐图见 [预测检查页](../experiments/vision/20260930_yolo26s_gpu_960/prediction_review/review.html)。CPU 与 GPU 初训的输入分辨率、精度和轮数不同，不能作为受控性能比较。

根据首轮问题，追加同代轻量 **YOLO26n-seg** 开发实验，保留相同划分和 960 输入。使用 batch 4、`nbs=64`（预热后目标累计 16 个微批次）、optimizer auto、warmup 3 epochs、warmup_bias_lr 0，仍限制 2 个 CPU 和 30% PyTorch 分配器显存；请求上限为 80 epochs / 30 分钟。检查已通过，8 张训练和 8 张验证的运行峰值预留显存 3786 MiB。

全量运行达到时间预算后正常退出，共完成 42 个完整 epoch 和第 43 个部分 epoch，总耗时约 1813.1 秒，峰值 PyTorch 预留显存 3964 MiB，进程峰值主机内存约 4049 MiB。记录与权重位于 [YOLO26n 修正配置](../experiments/vision/20260930_yolo26n_gpu_960_stable/)。**没有完成请求上限的 80 轮，也没有证明已收敛。** 最佳 checkpoint 的最终指标见上表；工作副本导出到 `models/`，不改旧配置的默认权重。

自动优化器实际选为 AdamW，初始学习率为 0.002；请求参数 `lr0=0.001` 在 auto 模式下被忽略，实际参数记录在 `resolved_training_settings.json` 中。本次同时改变了模型容量、批次和优化设置，不能据此独立证明哪一个因素导致效果变化。

## 逐图检查结论

YOLO26n 在固定阈值下有 34/49 帧输出预测，15 帧无输出；平均所选 mask IoU 为 0.4370，中位数为 0.5485，27/49 帧达到 IoU ≥0.5。这是帧级诊断，不能将 49 帧当成 49 株独立植物。逐图和分拍摄组数据保留在预测检查目录的 CSV/JSON 中，没有删掉难例。

已查看漏检汇总图和有输出的高、低重叠例图。`20260423_153449_073` 的预测与标注贴合较好（IoU 0.901）；`20260423_152048_341` 找到的是同一根前景主茎，但预测覆盖的长度比标注更长（IoU 0.411）。这说明部分低 IoU 来自茎段起止差异，不能全解释成定位到错误物体；同时，真实漏检仍然存在，不能因此忽略它们。

当前建议保留新模型作开发候选，先统一目标茎段的标注起止与遮挡规则，再用带真实植株编号、符合实验室部署距离的新苗数据比较新旧模型。旧模型原训练数据未知，不能凭本次旧图成绩宣布它的泛化能力更好；也不能因 YOLO26 更新就宣布新模型更好。

完整性检查确认：247 张原图、235 份原标注以及原 YOLO11n 权重均保持原哈希；3 处重复实例只在派生标签中修正，训练与验证没有共用时间组或完全相同图像。原始目录中没有训练缓存。

## 使用方式

在项目根目录执行。新 run 名称必须唯一，脚本拒绝覆盖已有 run。CPU 入口不会自动切换到 GPU；GPU 入口必须在协调好的使用时段显式运行。

```bash
cd /home/kevin/projects/tomato_franka_pipeline
.venvs/vision-cpu/bin/python -I -B tools/check_seedlings_export.py

# 只读查看运行状态；连接中断后先查这个，避免重复启动。
python3 -I -S -B tools/vision_status.py

# 修正配置的 YOLO26n 实验，同样需要已协调的 GPU 时段。
.venvs/vision-gpu/bin/python -I -B tools/train_stem_gpu.py \
  --name yolo26n_gpu_repeat --stage train --model yolo26n-seg.pt \
  --imgsz 960 --epochs 80 --minutes 30 --batch 4 --nbs 64 \
  --optimizer auto --warmup-epochs 3 --warmup-bias-lr 0

# 仅在已协调的 GPU 时段复现；限额详见下文。
.venvs/vision-gpu/bin/python -I -B tools/train_stem_gpu.py \
  --name yolo26s_gpu_repeat --stage train \
  --imgsz 960 --epochs 40 --minutes 30 --warmup-bias-lr 0.1

# 复现初训时使用新的 run 名称；训练预算不含最后的保存和验证。
.venvs/vision-cpu/bin/python -I -B tools/train_stem_cpu.py \
  --name yolo26s_cpu_pilot_repeat --stage pilot \
  --imgsz 640 --epochs 12 --minutes 15 --warmup-bias-lr 0.1

# 生成开发验证的逐图指标和叠图；输出目录须尚不存在。
.venvs/vision-cpu/bin/python -I -B tools/review_stem_predictions.py \
  --weights models/stem_yolo26n_seedlings_dev_v3_20260930.pt \
  --output experiments/vision/yolo26n_review_repeat \
  --imgsz 960 --conf 0.25
```

预测检查在每帧选择置信度最高的预测，使用在查看本轮结果前确定的 0.25 阈值；漏检计入 IoU=0。原图尺寸 mask 与标注逐像素比较，另按拍摄组汇总。这里的平均 IoU 与 Ultralytics 的 mask mAP 是不同指标，二者都不代表抓取成功率。所有验证图会保留，页面按 IoU 从低到高展示。

## GPU 检查与判断修正

`nvidia-smi` 返回 `Failed to initialize NVML: Driver/library version mismatch`。正在运行的内核模块版本为 **590.48.01**，`libnvidia-ml.so.1` 和 `libcuda.so.1` 均指向 **595.71.05**。最初据此先采用 CPU，但不能仅凭 NVML 查询失败推断 CUDA 不能工作。

用户追问后补做直接 CUDA 检查：`cuInit(0)` 成功，并成功创建、释放仅属于检查进程的 CUDA 上下文；显卡为 **RTX 5060 Ti**，总显存约 15.47 GiB，探测时空闲约 14.30 GiB。实际加载的 CUDA 驱动库仍为 `libcuda.so.595.71.05`。证据见 [初始化结果](../experiments/vision/cuda_probe_20260930.json) 和 [设备与空闲显存结果](../experiments/vision/cuda_device_probe_20260930.json)。该结果修正了此前把 NVML 异常当成训练阻碍的判断；完整可用性仍需真实 PyTorch 内核和训练验证。

用户已允许在当前空闲时段适度使用 GPU，现已建好独立的 `.venvs/vision-gpu`，使用 PyTorch 2.9.1+cu128 和 torchvision 0.24.1+cu128，并通过实际张量运算、AMP 与分割训练检查。这是项目内 Python 依赖，不是安装或替换系统 CUDA、驱动；完整依赖见 [GPU 版本锁定文件](../requirements-vision-gpu.lock.txt)。

GPU 入口为 `tools/train_stem_gpu.py`，仅显式选择 GPU 0，使用 2 个 CPU、workers 0；batch 在 1–4 范围内显式指定。PyTorch 分配器限额设为总显存的 30%，驱动上下文本身的开销不包含在这个限额中；每批加入空闲时间，目标训练活动比例约 25%，并非硬件级利用率配额。开始前空闲不足 8 GiB 则不启动；训练中低于 4 GiB 则结束本次训练。新增运行的完整控制台输出位于 `.tmp/<run_name>.log`，参数和版本仍在各 run 中保存。未改动系统、重启电脑或停止任何既有进程。

高分辨率训练采用 960 输入、小批量，并记录收敛、实际显存与耗时；是否采用该模型取决于验证图和新植株实验，而非仅凭模型代际。

下一批数据优先补实验室固定相机条件下的新植株：记录真实植株 ID，同株所有角度放同一集合；测试植株不参与调参。先核实拍摄距离与主茎深度有效率，再扩大采集；只提升分割分数不能解决深度缺失。
