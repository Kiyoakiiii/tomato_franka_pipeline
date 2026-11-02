# 新流水线使用说明（2026-10-05）

> 当前主线已改为按托盘穴位夹取，见 [托盘模式说明](tray_workflow_CN.md)。本文的环境、标定、容差实验部分仍然适用；全图 YOLO 流程（`50_run_transplant.py`）保留为可选方案。

软件流水线 `seedling/` + `scripts/` 覆盖感知、三维几何、标定、抓取候选、状态机和实验记录。不接硬件时可以在仿真中端到端运行。2026-10-05 已在实机上验证了读状态、小幅移动、夹爪张合、ZED 采集（NEURAL 深度）；标定和完整夹取流程还没有在实机上跑过。旧脚本 `01_`–`07_`、`src/`、`config.json` 已移到 `archive/legacy_v0/`。

## 硬件与权限状态（2026-10-05 更新）

| 项目 | 状态 |
| --- | --- |
| `zed`、`realtime`、`video` 组 | 已由管理员加入，重新登录后生效（`ulimit -r` = 99） |
| pyzed | 已装入 `.venvs/pipeline`（ZED SDK 5.3.0 自带的 cp312 安装包） |
| 机械臂网络 | 待开机、接网线（网口 `enp129s0`，系统已有 `Franka` 网络配置） |
| ZED 相机 | 待接入，安装位置待定 |
| 夹爪 | 改用实验室新夹爪，驱动待接入（Franka Hand 代码保留） |

设备接上后先运行 `.venvs/pipeline/bin/python scripts/00_check_env.py --robot`。

## 环境

- 项目内新环境 `.venvs/pipeline`（Python 3.12）：torch 2.9.1+cu128、ultralytics 8.4.46、**franky-control 2.0.0（对应系统已装的 libfranka 0.21.2）**、opencv 5.0、scipy、scikit-image、open3d。版本锁定在 `requirements-pipeline.lock.txt`，可用 `tools/setup_pipeline_env.sh` 重建。
- 旧的 franky 3.10 构建（vendor/、build/）无法在当前系统导入，已删除；franky 改用 pip 包。
- 运行脚本时缓存都写在项目内 `.cache/`、`.config/`。

## 不接硬件现在就能跑的

```bash
P=.venvs/pipeline/bin/python
$P -m pytest -q tests                                   # 15 个测试：几何/IK/RANSAC/标定/状态机成功与失败路径
$P scripts/50_run_transplant.py --sim --trials 3        # 合成场景端到端：观察→预抓取→再观察→夹取→拔出→运输→放置
$P scripts/50_run_transplant.py --sim --sim-extraction-force 100   # 拔出阻力超限 → 停止、松开、不重试
$P scripts/11_perceive_offline.py --synthetic 20        # 已知真值下的三维定位误差
$P scripts/11_perceive_offline.py "data/raw/tomato images" --limit 40 --set perception.device=cpu   # 旧数据
$P scripts/31_handeye_solve.py --synthetic              # 手眼标定求解器自检（含留出验证）
$P scripts/70_summarize.py <某次 session>/end_to_end_trials.csv    # 计数 + Wilson 区间
```

每次运行在 `experiments/runs/<时间>_sim|real/` 下保存：配置快照、每个 trial 的 `events.jsonl`（每个状态的动作、力、夹爪宽度）、每次观察的叠加图/JSON/原始帧 npz，以及按 `experiments/templates/` 字段写的 `end_to_end_trials.csv`。

## 离线结果（2026-10-05）

- **合成场景**（无遮挡、轻噪声的圆柱茎）：茎轴位置误差中位数 0.14 mm、p95 0.48 mm，切向误差中位 0.2°，直径估计偏大约 0.4 mm。这只说明几何代码正确，不代表真实精度。
- **旧数据前 40 帧**（真实 YOLO + 旧深度）：只有 7/40 帧拟合出合格的三维茎段，主要拒绝原因是"茎上有效深度点不足"。和 9 月 30 日的数据审查一致：旧数据拍得太近（茎深度中位约 0.17–0.3 m，低于 ZED 2i 最小测距），**三维部分必须在实验室用 ~0.4–0.6 m 的距离重新采集**，不能用旧深度评估。
- **仿真端到端**：6/6 成功，单次 ~3 s（仿真动作是瞬时的，不代表实际周期）。拔出阻力超限、30 mm 外参误差导致漏夹（有界重试后失败）、无目标拒绝这几条失败路径都有测试覆盖。

## 代码结构

| 模块 | 作用 |
| --- | --- |
| `seedling/camera.py` | ZED 采集：左图、**XYZ（米，IMAGE 坐标系）**、置信度图、内参与元数据，多帧中值；npz 保存/回放；读旧数据 |
| `seedling/segmentation.py` | YOLO 分割，`retina_masks=True` 保证 mask 与原图像素对齐，可选工作区 ROI |
| `seedling/stem3d.py` | mask 清理 → 骨架最长路径（中心线）→ 滑窗分段 → 弯曲检查 → 优先内部像素 → 置信度/深度跳变过滤 → **RANSAC + PCA** 局部直线 → 长度、残差、内点率、直径；可见表面沿视线后移半径近似茎轴 |
| `seedling/grasp.py` | 茎段转基座坐标 → 侧向接近方向（偏航 × 下倾）→ 夹爪坐标系（闭合轴 ⟂ 茎、⟂ 接近）→ 工作空间、台面、手指扫掠体点云间隙、**整条动作序列 IK** → 排序 |
| `seedling/kinematics.py` | FR3 正运动学 + 阻尼最小二乘 IK（只用于事前可达性检查） |
| `seedling/robot.py` | franky 2.0 真机封装与仿真机，统一安全检查（有限值、刚体、接触点工作空间、IK）；力阈值反应（`Reaction` + `CartesianStopMotion`）；**不会自动清错** |
| `seedling/gripper.py` | Franka Hand / 串口 / 无 / 仿真；用宽度判断是否夹到东西 |
| `seedling/calibration.py` | 固定相机手眼标定：ArUco 装在夹爪上，联合求 `T_base_cam` 与 `T_ee_marker`（Park 闭式初值 + 鲁棒非线性优化） |
| `seedling/task.py` | 状态机：OBSERVE → VALIDATE → PREGRASP → REOBSERVE → ACQUIRE → EXTRACT → TRANSPORT → PLACE → VERIFY_RELEASE；RECOVERY 有界重试 |
| `seedling/synthetic.py` | 合成 RGB-D 场景（射线求交渲染茎/托盘/叶片）+ 简单世界模型，用于离线端到端和测试 |
| `seedling/trial_log.py` | session/trial 日志，CSV 列名严格对齐 `experiments/templates/` |

坐标约定写在 `seedling/__init__.py`：`T_AB` 把 B 坐标变到 A；B 基座、E franky 末端、G 夹爪接触中心（z=接近、y=闭合、x≈沿茎）、C 相机（x 右 y 下 z 前，米）。

配置集中在 `configs/pipeline.yaml`，带 `[待测]` 的都是占位值；命令行可用 `--set 节.键=值` 临时覆盖。

## 实验室上手顺序

每一步都有退出条件，前一步不过不进下一步。所有真机动作默认低速，并在每次运动前要求按 Enter（`--step`）。

1. **环境与只读状态**
   `scripts/00_check_env.py --robot` → `scripts/20_robot_state.py`
   检查：能读位姿；`FK(config flange_to_ee) vs robot EE` 误差 < 5 mm（否则 Desk 里的末端设置和 `robot.flange_to_ee` 不一致，先改配置）。
2. **小步运动**
   `scripts/21_robot_jog.py --dz 10`，再 `--home`。确认方向、速度、急停。按实际桌面改 `robot.workspace_m` 和 `robot.home_joints_rad`（home 位要不挡相机视野）。
3. **夹爪**
   `scripts/22_gripper_test.py --homing --grasp-empty 3` → 把输出的空夹宽度写入 `gripper.empty_close_width_m`；`--grasp 4` 夹已知直径木棍看压痕。
4. **相机**
   固定 ZED（建议托盘侧前方 0.4–0.6 m、斜向下）。`scripts/10_capture.py --perception` 看茎内有效深度比例和合格茎段；同时按新植株规则采集数据（自动写 `vision_frames.csv`）。
5. **手眼标定**
   打印 ArUco（DICT_4X4_50，id 0，边长 ≥ 5 cm，**实测打印尺寸**），刚性固定在手上朝向相机：
   `scripts/30_handeye_collect.py --marker-mm 60 --auto 24` → `scripts/31_handeye_solve.py calibration/sessions/<id> --save`
   看**留出样本**误差（不是拟合残差）。脚本也会打印"双目深度 − PnP 深度"，用来判断 ZED 深度是否有系统偏差。
6. **夹爪容差**（proposal 第 1 部分）
   `scripts/40_tolerance_trial.py --stage hold_transport --axis c --offsets=-6,-4,-2,0,2,4,6 --repeats 2`，再做 a、s 轴和 `extract_from_tray`。参考位姿用 c/a/s 文字点动示教。
7. **定位验证**（proposal 第 3、6 部分）
   `scripts/41_localization_check.py --plant-id P01`：视觉估计 → 机器人张开夹爪到估计位置 → 操作员点动到真正居中 → 记录 c/a/s 误差。用它对比第 6 步的容差。
8. **端到端**
   先假茎再真苗：`scripts/50_run_transplant.py --plant-id P01 --step --ask`。确认流程可信后再去掉 `--step`。`--no-reobserve` 可做"单次观察 vs 再观察"对照。
9. **汇总**：`scripts/70_summarize.py …csv`

## 设计上的取舍和已知局限

- **接近方向默认下倾 30°/45°**：实测 IK 表明水平侧向接近会让 FR3 第 6 关节贴近上限（4.52 rad），中高位置大多不可达；45° 在测试的位置全部可达。闭合方向仍垂直于茎。
- 从 home 到预抓取用关节空间运动（避免大幅转姿态的笛卡尔直线），之后接近/拔出/运输为短距离笛卡尔直线。IK 只是事前检查，**整条路径与连杆碰撞没有做完整规划**；第一次跑每个新位置都应该用 `--step`。
- 抓取是否成功目前只靠夹爪宽度 + 拔出后的再拍照（原位置是否仍有茎）；放置成功默认靠宽度判断，建议加 `--ask` 由操作员确认。
- 臂端外力估计 `O_F_ext_hat_K` 不是茎受力，只作为异常阻力/碰撞的停止条件；阈值 `extraction_force_limit_n`、`guard_force_limit_n` 需要用实验确定。
- 茎轴位置 = 可见表面拟合线沿视线后移半个估计直径，这是模型假设，需要第 7 步的独立参照来检验偏差。
- 抓取点偏向茎的较低位置（利于拔出），但必须高于 `grasp.table_z_m + min_contact_height_m`，`table_z_m` 要实测托盘上表面高度。
- 仿真里的夹持/拔出是几何规则，不是物理仿真，仿真成功率没有实验意义，只用来验证软件流程。
