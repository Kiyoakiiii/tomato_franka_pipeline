# 番茄幼苗移栽（FR3 + ZED 2i）

FR3 机械臂按托盘穴位夹取番茄苗，拔出后放进杯子。视觉只负责修正倾斜或偏位的苗，不需要训练模型。全图 YOLO 检测作为可选方案保留。夹爪 2026-10-15 起换成实验室的 pin-brake 夹爪（`gripper.type: pin_brake`，驱动用同学提供的 `example_pinBrakeGripper.py`）。

- 托盘流程与视觉修正原理：[docs/tray_workflow_CN.md](docs/tray_workflow_CN.md)
- 环境、标定、各步骤脚本的详细说明：[docs/pipeline_quickstart_CN.md](docs/pipeline_quickstart_CN.md)
- 实验记录的字段与统计口径：[docs/experiment_protocol_CN.md](docs/experiment_protocol_CN.md)

## 目录

```
configs/pipeline.yaml   所有参数（标 [待测] 的需实测后修改）
seedling/               核心代码：相机、感知、几何、抓取、机器人/夹爪、状态机、托盘、仿真
scripts/                入口脚本，按实验室步骤编号（见下表）
tests/                  自动测试（不需要硬件）
tools/                  环境安装脚本 + 之前 YOLO 数据整理/训练工具
calibration/            打印用标记 aruco_marker_A4.pdf；相机标定 T_flange_cam.yaml、托盘 tray.yaml（运行后生成）
data/                   原始数据 raw/、训练数据 curated/、新采集 captures/
models/                 YOLO 权重（当前用 stem_yolo11n_v3_clean_one_stem.pt）
experiments/            实验输出（每次运行一个带时间戳的目录）与 CSV 模板
docs/                   说明文档；docs/archive/ 为 9 月底的旧审查与计划
archive/legacy_v0/      第一版脚本（01_–07_、src/、config.json），已被 seedling/ 取代，仅供参考
.venvs/pipeline/        项目运行环境（Python 3.12，franky 2.0，torch，pyzed，TensorRT）
```

## 常用命令

所有命令都在项目根目录运行，使用项目环境的 Python：

```bash
cd /home/kevin/projects/tomato_franka_pipeline
P=.venvs/pipeline/bin/python
```

| 步骤 | 命令 | 会不会动机械臂 |
| --- | --- | --- |
| 环境检查 | `$P scripts/00_check_env.py --robot` | 不会 |
| 相机预览/采集 | `$P scripts/10_capture.py --perception` | 不会 |
| 离线感知 | `$P scripts/11_perceive_offline.py <帧或目录>` | 不会 |
| 机械臂状态 | `$P scripts/20_robot_state.py` | 不会 |
| 小步移动 | `$P scripts/21_robot_jog.py --dz 10` / `--home` | 会（每次确认） |
| 夹爪测试 | `$P scripts/22_gripper_test.py --open` / `--grasp-empty 3` / `--grasp 3 --hold-s 5`（每步按 Enter，需在自己的终端运行；新加入 dialout 组后要重新登录，或在命令前加 `sg dialout -c '...'`） | 只动手指 |
| 手上相机标定（当前） | `$P scripts/32_wrist_calib_collect.py --marker-mm 40` → `$P scripts/33_wrist_calib_solve.py calibration/sessions/wrist_<id> --ids 0 1 2 3 --save` | 会（每步确认；`--no-step` 不确认） |
| 标定快速检查 | `$P scripts/34_wrist_calib_check.py --marker-mm 40 [--point --marker-id 1]` | `--point` 时会 |
| 斜视角补采（工作条件下） | `$P scripts/32_wrist_calib_collect.py --marker-mm 40 --use-calib --max-angle-deg 25 --dry-run`（去掉 `--dry-run` 才动） | 会 |
| 标定独立验证采集（斜视、手腕大转角） | `$P scripts/36_wrist_calib_validate_collect.py --dry-run`（去掉才动） | 会 |
| 托盘定位（一张照片求 72 个穴位） | `$P scripts/60_tray_locate.py`（需整个托盘入镜） | 不会 |
| 托盘一致性检查（多姿态看托盘） | `$P scripts/61_tray_consistency.py --dry-run` | 会 |
| 相机重装后快速修正安装（用托盘） | `$P scripts/61_tray_consistency.py --dry-run` → 去掉 `--dry-run` 拍 → `$P scripts/63_tray_mount_fix.py experiments/tray_consistency/<id> [--save]` | 会（61） |
| 托盘定位（格子匹配 + 外轮廓交叉检查，顺便测每个土块高度） | `$P scripts/60_tray_locate.py` | 不会 |
| 夹爪开合 | `$P scripts/66_gripper.py close` / `open` / `status` | 只动手指 |
| 按坐标侧向夹取一株（分步） | `$P scripts/65_side_grasp.py --cell C5 observe|plan|pregrasp|approach|lift|lower|retreat` | 会 |
| 一株完整流程（观察→夹取→上提→放回→回总览） | `$P scripts/69_pick_cycle.py --cell C5 --put-back`；中途 `touch experiments/grasp/STOP` 在当前步骤完成后停下 | 会 |
| 回俯视总览位置 | `$P scripts/68_goto_overview.py` | 会 |
| 逐穴位实拍识茎 | `$P scripts/59_tray_cells_live.py --cells calibration/tray_grid.json --dry-run` | 会 |
| 标定在新数据上评估 / 多次合并重标 | `$P scripts/35_wrist_calib_eval.py calibration/sessions/wrist_<新>`（`--refit [--save]` 合并所有给出的 session、拟合第 5 关节偏置，逐个留出验证） | 不会 |
| 固定相机标定（备用） | `$P scripts/30_handeye_collect.py --marker-mm 100 --auto 20` → `$P scripts/31_handeye_solve.py calibration/sessions/<id> --save` | 会 |
| 手上相机观察测试（无需夹爪） | `$P scripts/56_wrist_observe.py --marker-id 0 [--dry-run]`（或 `--cell A1`） | 会（`--dry-run` 不会） |
| 茎线三角化（无需夹爪，不动） | `$P scripts/57_stem_triangulate.py`（`--orbit-deg 15` 换视角验证，会动） | 默认不会 |
| 夹爪容差实验 | `$P scripts/40_tolerance_trial.py --stage hold_transport --axis c` | 会 |
| 定位验证 | `$P scripts/41_localization_check.py --plant-id P01` | 会 |
| 示教托盘 | `$P scripts/45_teach_tray.py --rows 4 --cols 6 --grasp-height-mm 30` | 会（点动） |
| 按穴位夹取 | `$P scripts/55_run_tray.py --cells A1,A2 --mode blind --step --ask` | 会 |
| 全图检测夹取（可选） | `$P scripts/50_run_transplant.py --step --ask` | 会 |
| 结果汇总 | `$P scripts/70_summarize.py <csv>` | 不会 |
| 仿真（无硬件） | `$P scripts/55_run_tray.py --sim --mode vision --sim-tilt 20` | 不会 |
| 测试 | `$P -m pytest -q tests` | 不会 |

会动机械臂的脚本，在确认流程可靠之前都加 `--step`：每个动作执行前会在终端等你按 Enter，输入 q 中止。临时改参数用 `--set 节.键=值`，例如 `--set gripper.grasp_force_n=10`。

## 实机前提

- **机械臂**：Desk 里切到 Execution 模式并激活 FCI（指示灯变绿），外部急停处于释放状态。实验结束后关闭 FCI，再锁定关节。
- **ZED**：插在 USB 3 口上（插在 USB 2 口会报"stream failed to start"）。NEURAL 深度依赖项目环境里的 TensorRT，脚本启动时会自动设置路径。
- 不要手动运行 `archive/` 里的旧脚本：旧的 `02_robot_smoke_test.py` 一连接机器人就会自动清除错误。

## 当前进度（2026-10-13）

- 实机已验证：读取机械臂状态、上下 1 cm 移动、Franka Hand 张合与读宽度、ZED HD1080 + NEURAL 深度。
- 相机已装在手腕上（`camera.mount: wrist`）。2026-10-12 用实验室的 4 标记标定纸（DICT_4X4_50，ID 0–3，40 mm）完成手上相机标定：留出样本横向误差中位 1.5 mm、最大 3.5 mm（0.4 m 处），结果在 `calibration/T_flange_cam.yaml`。
- ZED 双目（SDK 自校准后）视差偏大约 19.2 px（0.32–0.45 m 时深度短约 5.5%），配置 `camera.disparity_offset_px` 统一修正稠密深度和茎线三角化；`34_wrist_calib_check.py` 每次会重估。
- 神经网络深度（NEURAL / NEURAL_PLUS）会把 2–4 mm 的细茎抹到背景上，不能直接用于茎的几何。茎改用**茎线双目三角化**（`seedling/stemline.py`）：左右校正图里找茎中心线再三角化，土面高度用稠密深度。2026-10-12 实拍一株苗、两个视角（相机绕苗转 15°）在基座坐标下茎根相差约 1 mm、夹取高度处 1.2 mm、方向 1.6°。
- 2026-10-13 识茎修正：茎基部发白沾土（不是绿色），现在从最低绿色像素沿"比两侧亮的脊"继续往下追到入土处，茎根取在该像素射线上；原来用直线外推，弯曲的茎基会偏约 4 mm。新增两项真正独立的检查：茎根（茎的双目深度）对比同一射线与土面（稠密深度）的交点（3 帧 0.4–1.5 mm）；每 9 行单独匹配的视差与整体拟合的残差（0.4–0.9 px，约 0.4–0.8 mm 深度）。原来的"3-D line RMS"按构造恒为 0，已删除。
- 2026-10-13 标定重做：在垫高托盘上（苗的高度）从斜起始位采了一组斜视样本（俯角 45–67°），旧标定在这组数据上横向误差中位 2.9 mm、最大 7.2 mm，不够准。原因是**机械臂第 5 关节的读数和真实几何差约 1.09°**（三次采集各自估计 1.03–1.18°，一致）：每个姿态的标记整体偏移，任何固定的相机安装都吸收不了，只给第 5 关节加一个常数偏置后残差从 5.5 px 降到 1.3 px。现在的标定（三次采集合并，`T_flange_cam.yaml` 里含 `joint_offsets_rad`）对每次采集轮流留出验证：横向误差中位 0.4–0.65 mm、最大 ≤1.25 mm，包括斜视角。`load_T_FC` 会把偏置交给 `kinematics`（FK/IK 都用修正后的角度），相机位姿一律由关节角经修正模型计算（`calibration.flange_pose`），不再用 franky 报的末端位姿（名义模型）。旧标定保留在 `calibration/T_flange_cam.wrist_wrist_20261012_171513.yaml`。
  - 影响：franky 的笛卡尔运动按名义模型执行，到了第 5 关节附近会有约 1° / 数毫米的偏差；以后接近和夹取要用修正模型做 IK 后发关节运动，或者补偿。这个偏置可能是机械臂本身的问题，同一台机械臂的其他使用者也会受影响，建议告诉实验室管理员。
  - 视差偏置三次采集分别估计为 19.2 / 18.8 / 18.2 px（斜视角那次 18.2），配置 19.2 暂不改（1 px 约等于 0.3% 深度，0.36 m 处约 1 mm）。
- 识茎目前只测过 1 株苗：用新标定重算两个视角（相机绕苗转 15°），茎根水平相差 1.1 mm、高度 1.9 mm，夹取点相差 1.8 mm，倾角 20.5° / 19.3°。
- 2026-10-12 修正：之前 franky 的全局速度系数与每次运动的系数相乘，实际速度只有配置值的 1/20；现在配置里的 `dynamics`/`fine_dynamics` 就是实际值（上限 0.2）。
- 已写好、只在仿真中跑通：托盘示教与逐穴位夹取、视觉修正、状态机。
- 手上相机观察流程已接入托盘流程（`seedling/wrist.py`）：每个穴位先规划斜视观察位（仰角 55°、距离 0.42 m，同一仰角下选关节变化最小的方位/滚转，全路径检查相机外壳/法兰/腕部离地），静止后拍照，再做穴位视觉；抓取候选也检查相机外壳不碰台面。仿真 20° 倾斜 11/12。实机待测（`56_wrist_observe.py`）。
- 2026-10-15 夹爪换成 pin-brake（串口 CH340，`/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`）。参数用同学在番茄茎上测好的：停止压力 3 kPa（`p 3`），名义宽度 2 mm（`close 2`）。`seedling/gripper.py` 的 `PinBrakeGripper` 调用同学的类，命令和时序都不变；实测空夹也会在加压约 20 ms 后（约 1.09 s）停下，所以夹爪本身判断不了有没有夹到（也没有宽度读数），夹住与否要靠视觉或拔出时的力来判断；走到全闭表示压力没建立起来（气路问题）。`open()` 会等手指张开完才返回。打开串口会让 Arduino 复位、夹爪张开，所以每个进程只开一次，夹着苗时不要再开。kevin 账户目前不在 `dialout` 组里，打不开串口，需要管理员处理。相机暂时卸下，重新装上后要重做手上相机标定（第 5 关节偏置是机械臂本身的，不用重测）。
- 2026-10-20 相机和夹爪一起重新装到手腕上后重新标定。旧标定在新数据上横向误差中位 4.3 mm、最大 11.5 mm（相机实际转了约 2.6°）；之前的标定样本里手腕（第 7 关节）几乎没转，所以手腕一转，托盘位置就偏 13–15 mm。新标定用 3 组共 46 个样本（第 7 关节 −0.72 到 2.94 rad，俯视角 52–88°），偏置加在第 2、5、6 关节（0.64° / 1.05° / −0.16°；第 5 关节的偏置和 10-13 测的一致，是机械臂本身的）。轮流留出整组验证：在第 5 关节 −0.5 到 0.6 rad、第 7 关节 ≤ 2.65 rad 范围内，误差中位约 1 mm、最大约 2.5 mm；超出这个范围实测 3–5 mm，所以观察位只在这个范围内选（`tray.observe.joint_range_rad`）。工作区域端到端检查：7 张近距离图（第 6 关节 2.5–2.7、第 7 关节最大 2.64）定位托盘，和总览图平均相差 0.8–2.6 mm（旧标定 13–15 mm）。视差偏置改为 15.8 px。
- 托盘：穴位分布固定（6 × 12，穴距 44.8 mm，`tray.model`），每次只需求位置：`60_tray_locate.py` 把左图投影成托盘顶面的 1 mm 俯视图，再用格子模板匹配，结果存到 `calibration/tray_grid.json`。
- 2026-10-22 相机支架拆下重装（先装错一次：桌面算出倾斜 11°；改正后用两个螺丝装回）。用托盘做一致性检查，偏差中位 3.8 mm、最大 10.6 mm，而且随手腕转角单调变化，说明安装位置变了约 0.5°、约 1 cm（螺丝孔间隙可以造成这么大的差别）。新增 `seedling/traycalib.py` 和 `63_tray_mount_fix.py`：把托盘格子当作已知尺寸的标定板，在多张整托盘照片里同时拟合托盘位置和相机安装修正，机械臂的关节偏置不变。两次检查共 12 张远处照片求得修正 0.61° + (−5, 4, −13) mm，留一验证中位 1.2 mm、95% 分位 2.9 mm，已保存（calibration_id 末尾为 `+trayfix_combined_whole_20261022`）。只拍到局部托盘的近处照片定位不可靠，没有使用，所以近距离（0.34 m）的精度还要用标定纸验证。托盘定位现在支持 0°/90° 两种摆放；判断哪些穴位有苗，改为看穴位中心外圈 9–15 mm 处有没有泥土的褐色（深度在叶子下面不可靠）。
- 2026-10-22 按坐标侧向夹取跑通：夹爪向下倾斜 26° 从机械臂一侧接近（相机在夹爪上方；倾斜小于约 26° 时手腕离托盘不到 7 cm），合拢点 = 穴位中心 + 沿闭合方向 10 mm（夹爪固定偏移），高度 = 这株的泥土表面 + 25 mm；手指闭合时会向前摆，张开时停在合拢点后方 13 mm。土块高度逐株测量（总览 + 观察位，两者不一致时取较高者），土块高于、平齐或低于托盘都可以。C8、C5（托盘挪动后）都一次夹准并带土提起 6 cm 后放回。托盘定位加了交叉检查：格子匹配与"外轮廓 + 已知穴距"相差超过三分之一穴距即判为错位（曾错一列），改用外轮廓。
- 待做：固定相机并完成手眼标定；用假茎跑通托盘流程；接入新夹爪；等农场的苗到了以后做真实实验。
