# 番茄幼苗移植（FR3 + ZED 2i + pin-brake 夹爪）

FR3 机械臂按托盘穴位坐标侧向夹取番茄苗（连土块），放进红色圆盘的 6 个洞里。视觉用来定位托盘和圆盘、测每个土块的高度，以及判断哪些穴位有苗，不需要训练模型。

## 快速开始

```bash
cd /home/kevin/projects/tomato_franka_pipeline
./run_transplant.sh            # 一条命令跑完：中间按空格继续，q 停止
./run_transplant.sh run 1      # 同上，只做一株
```

运行时先检查机械臂（连不上或有报错就停下，并提示怎么处理），然后在两个地方暂停等你按键：第一次回总览位之前；每株苗开始之前，屏幕上会显示要夹哪株、放进哪个洞、从哪个方向接近。按 q 后，会等当前动作做完再停，紧急情况请直接按急停。

单独调试时可以用：`check`（机械臂状态）、`locate`（回总览位并定位）、`plan`（只规划不动）、`stop`（另开终端运行，当前这一步做完后停下）、`open`（松开夹爪）、`overview`（只回总览位）。例如 `./run_transplant.sh plan`。

每轮流程：总览拍照 → 定位托盘和圆盘 → 选最近、且夹取和放置都安全的一株 → 近距离观察测土面高度 → 侧向夹取 → 上提 100 mm → 抬高搬运 → 在圆盘顶面高度松开 → 回总览，拍照确认穴位空了、洞也少了一个 → 下一株。任何一步失败都会停下，机械臂留在失败时的位置。

运行输出保存在 `experiments/grasp/logs/`，每次运行的 JSON 记录保存在 `experiments/grasp/transplant_all_*.json`，近距离照片保存在 `experiments/grasp/closeups/`。

## 实机前提

- **机械臂**：Desk 里切到 Execution 模式并激活 FCI（指示灯变绿），外部急停处于释放状态。报错一律在 Desk 里清除，脚本从不自动清除。用完后关闭 FCI，再锁定关节。
- **ZED**：插在 USB 3 口上。NEURAL 深度依赖项目环境里的 TensorRT，`scripts/_common.py` 会自动设置库路径。
- **夹爪**：串口 CH340（`/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`），驱动是同学的 `example_pinBrakeGripper.py`（停止压力 3 kPa，名义宽度 2 mm）。夹住时泵会间歇响，这是正常的。夹爪本身判断不了有没有夹到，要靠拍照确认。
- 同一时间只能有一个脚本连接机械臂（`run_transplant.sh` 会检查）。

## 目录

```
run_transplant.sh       一键入口（见上）
configs/pipeline.yaml   所有参数（总览位 tray.overview_q、托盘型号 tray.model、速度等）
seedling/               核心代码
  camera.py               ZED 采集（左右图 + NEURAL 深度）
  calibration.py          手上相机标定（含关节偏置），load_T_FC / flange_pose
  kinematics.py           FR3 正逆运动学（含标定得到的关节偏置）
  robot.py / gripper.py   franky 机械臂封装 / pin-brake 夹爪
  traygrid.py             从一张总览图定位托盘（格子模板 + 外轮廓交叉检查）、判断有苗穴位
  disc.py                 定位红色圆盘和空洞（洞口红圈；被叶子挡住时用外圈轮廓 + 转角）
  plugsense.py            近距离测土块顶面高度
  stemline.py             颜色掩码（绿色植物 / 褐色基质）与茎线双目三角化
  wrist.py                观察位规划、机械臂各部位离地检查
  transforms.py / config.py / synthetic.py（仿真场景，只给测试用）
scripts/                入口脚本（见下表）
tests/                  自动测试（不需要硬件）：.venvs/pipeline/bin/python -m pytest -q tests
tools/                  环境安装脚本
calibration/            T_flange_cam.yaml（相机标定）、tray_grid.json / disc.json（最近一次定位）、标定采集 sessions/
experiments/            实验输出：tray/（每次定位的照片和叠加图）、grasp/（夹取计划、日志、近距离照片）
archive/                旧代码：legacy_v0/（第一版）、cleanup_20261102/（2026-11-02 整理时移出的脚本和模块）
.venvs/pipeline/        项目运行环境（Python 3.12，franky 2.0，pyzed，TensorRT）
```

## 脚本

`P=.venvs/pipeline/bin/python`，都在项目根目录运行。

| 用途 | 命令 | 会不会动 |
| --- | --- | --- |
| 环境检查 | `$P scripts/00_check_env.py [--robot]` | 不会 |
| 机械臂状态 | `$P scripts/20_robot_state.py` | 不会 |
| 小步移动 | `$P scripts/21_robot_jog.py --dz 10` / `--home` | 会（每次确认） |
| 托盘和圆盘定位 | `$P scripts/60_tray_locate.py`（离线：`--frame experiments/tray/<id>/frame.npz --no-save`） | 不会 |
| 侧向夹取一株（分步） | `$P scripts/65_side_grasp.py --cell C5 observe\|plan\|pregrasp\|approach\|lift\|lower\|retreat` | 会（plan 不会） |
| 夹爪开合 | `$P scripts/66_gripper.py close\|open\|status` | 只动手指 |
| 回总览位 | `$P scripts/68_goto_overview.py [--dry-run]` | 会 |
| 一株完整流程 | `$P scripts/69_pick_cycle.py --cell C5 --place 3`（或 `--put-back` 放回原处） | 会 |
| 放进圆盘（分步） | `$P scripts/71_place_in_disc.py plan --cell C5 --free 1,2,3` | plan 不会 |
| 全自动移植 | `$P scripts/72_transplant_all.py [--pause] [--dry-run] [--max-plants N]`（`run_transplant.sh` 用的是 `--pause`） | 会 |
| 重新确定总览位 | `$P scripts/70_overview_for_area.py --dry-run`（去掉才动，并写入配置） | 会 |
| 相机标定 | `32_wrist_calib_collect.py` 采集 → `33_wrist_calib_solve.py ... --save`；`34_wrist_calib_check.py` 快速检查；`36_wrist_calib_validate_collect.py` + `35_wrist_calib_eval.py` 独立验证 / 多次合并 | 会（每步确认） |

## 关键参数和经验

- **标定**：机械臂第 2、5、6 关节读数有小偏置（约 0.64° / 1.05° / −0.16°），和相机安装一起标定，保存在 `T_flange_cam.yaml` 的 `joint_offsets_rad` 里。相机位姿一律由关节角经修正模型计算，不用 franky 报的末端位姿。相机或支架拆装后要重新标定。
- **托盘**：6 × 12 穴，穴距 44.8 mm（`tray.model`）。总览图里必须看到整个托盘，手或机械臂不能挡在托盘上（超过 8% 的穴位比托盘顶面还高时，定位判为失败）。
- **夹取**：夹爪向下倾斜 26°，从机械臂一侧接近；接近方向依次试 0、±30、±60、±90°，选不碰其他苗的方向。合拢点在土面上方 17 mm（土面高度没测到时 35 mm），叶子多的苗用 14.5 mm；手指经过的地方有更高的土时自动抬高。茎只有 2–3 mm 粗，深度图测不准茎本身，所以夹取位置按穴位坐标，不做茎的位置修正。
- **放置**：搬运时土块底部比沿途最高处高 150 mm；在圆盘顶面高度松开，手腕离 80 mm 范围内的最高处至少 70 mm。
- **速度**：普通移动 `dynamics` 0.05（0.06 触发过加速度不连续报错），精细动作 0.03。
