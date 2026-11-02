# 明天实验室运行说明：Tomato Stem Vision + Franka Pipeline

这个文件夹是独立工具包：

```text
lab_code/tomato_franka_pipeline
```

它不会改动同学原来的 `Pin Brake` / `franky` / `libfranka` 代码。默认所有机器人动作都是 dry-run，只有加 `--execute` 才会真正控制 Franka。

## 0. 进入目录

在实验室电脑终端：

```bash
cd ~/Desktop/lab_code/tomato_franka_pipeline
```

如果实验室电脑路径不是 `~/Desktop/lab_code`，进入你实际复制的位置即可。

## 1. 先确认依赖

需要：

```bash
python3 check_environment.py
```

如果提示缺普通 Python 包，先安装：

```bash
python3 -m pip install -r requirements.txt
```

也可以手动检查：

```bash
python3 -c "import cv2, numpy, ultralytics; print('vision ok')"
python3 -c "import pyzed.sl as sl; print('zed ok')"
python3 -c "import sys; sys.path.insert(0, '../franky/franky'); import franky; print('franky ok')"
```

如果 `ultralytics` 缺失：

```bash
python3 -m pip install -r requirements.txt
```

如果 `pyzed` 缺失，需要用装了 ZED SDK 的实验室电脑运行。

注意：`pyzed` 不是 pip 包；它来自 Stereolabs ZED SDK。`franky` 目前优先使用复制来的本地目录 `../franky/franky`。

## 2. 第一步只测视觉，不动机器人

接上 ZED 2i，运行：

```bash
python3 01_vision_check.py --source zed
```

窗口快捷键：

```text
q: 退出
s: 保存当前 RGB / overlay / JSON
```

你要看三件事：

1. 能不能打开 ZED 左图。
2. YOLO 有没有在成熟苗/假茎上画出 mask。
3. 画面左上角有没有显示 approximate robot target xyz。

如果没有 ZED，只想测试模型窗口：

```bash
python3 01_vision_check.py --source webcam
```

或者用图片：

```bash
python3 01_vision_check.py --image /path/to/stem_rgb.png --once
```

## 3. 第二步只读 Franka 状态，不移动

确认 Franka 开机、FCI 可用、电脑和机器人在同一网络后运行：

```bash
python3 02_robot_smoke_test.py
```

这一步只连接并读取 state，不会移动机器人。

如果 robot IP 不是 `192.168.137.2`：

```bash
python3 02_robot_smoke_test.py --host ROBOT_IP
```

## 4. 第三步做很小的安全移动

确认急停、工作空间、桌面和夹具都安全后，只做向上 10 mm 的小动作：

```bash
python3 02_robot_smoke_test.py --execute --move-z-mm 10
```

不要第一次就往下移动。先确认坐标方向和控制链路没问题。

## 5. 第四步固定位置假茎抓取

先用橡胶管、绿色细杆或假茎，不要直接用真实植物。

先 dry-run 看计划：

```bash
python3 03_fixed_grasp_test.py
```

真正执行：

```bash
python3 03_fixed_grasp_test.py --execute --x 0.50 --y 0.00 --z 0.17
```

脚本会按这个顺序执行，并且每一步都要求按 Enter：

```text
pregrasp
open gripper
descend
close gripper
lift
```

### 夹爪设置

默认 `config.json` 里：

```json
"end_effector": "none"
```

所以机器人会移动，但不会真的关夹爪。

如果用 Franka 官方 gripper，改成：

```json
"end_effector": "franka_gripper",
"use_franka_gripper": true
```

如果用 Pin Brake / Arduino serial gripper，改成：

```json
"end_effector": "serial_gripper"
```

并检查：

```json
"serial_gripper": {
  "port": "/dev/ttyUSB0",
  "open_command": "G0",
  "close_command": "G70"
}
```

端口可能是 `/dev/ttyUSB0`、`/dev/ttyACM0`，或者其他。

## 6. 第五步做相机到桌面平面的简单标定

这是把图像 pixel 转成 robot XY 的关键步骤。运行：

```bash
python3 04_collect_planar_calibration.py --source zed --update-config
```

流程：

1. 在相机画面里放一个明显点，例如工具尖端、笔尖、标记点。
2. 让这个点对应一个你知道的机器人 base XY。
3. 在图像窗口点击该点。
4. 终端输入对应 robot x y，例如：

```text
0.500 0.000
```

至少采 4 个点，且要分布在工作区域四周。采完按 `s` 保存。

保存成功后，脚本会把 `config.json` 里的：

```json
"mode": "homography"
```

自动打开。

## 7. 第六步视觉引导抓取

先 dry-run：

```bash
python3 05_vision_guided_grasp.py --source zed
```

窗口快捷键：

```text
g: 对当前检测目标生成抓取计划；dry-run 时不会动
q: 退出
i/k: robot x 正/负方向微调
j/l: robot y 正/负方向微调
u/o: robot z 正/负方向微调
r: 重置微调
p: 保存 overlay
```

标定完成后再执行：

```bash
python3 05_vision_guided_grasp.py --source zed --execute
```

如果还没做 homography 标定，只想用粗略 manual scale 测试，必须显式加：

```bash
python3 05_vision_guided_grasp.py --source zed --execute --allow-rough-manual-scale
```

这个只适合非常低速、假茎、人工盯着急停的测试。

## 推荐明天实际目标

不要把明天目标定成“一次自动抓番茄”。建议按这个验收：

```text
上午：
1. 01_vision_check.py 能识别成熟苗照片/假茎/实际目标
2. 02_robot_smoke_test.py 能读 Franka state
3. Franka 能向上移动 10 mm

下午：
4. 03_fixed_grasp_test.py 能固定位置抓橡胶管/假茎
5. 04_collect_planar_calibration.py 采 4 个点
6. 05_vision_guided_grasp.py dry-run 给出合理 robot xyz
```

如果第 6 步 dry-run 的 robot xyz 看起来合理，再考虑 `--execute`。

## 重要安全边界

- 不要一开始就用真实植物。
- 不要第一次就运行 `--execute`。
- 不要第一次让机器人向下移动。
- `config.json` 里的 `workspace_limits_m` 是硬限制，目标超出会报错。
- `relative_dynamics_factor` 已经设成 0.04，保持低速。
- 所有真实动作前，脚本都会要求按 Enter；看到不对就输入 `q` 中止。
