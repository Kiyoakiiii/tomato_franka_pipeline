#!/usr/bin/env bash
# 番茄苗移植：一条命令跑完整个流程（总览定位 -> 由近到远夹取 -> 放进圆盘 -> 拍照确认 -> 下一株，直到圆盘放满）。
#
#   ./run_transplant.sh check        查看机械臂状态和报错（不动，不清除报错）
#   ./run_transplant.sh locate       回总览位，拍照定位托盘和圆盘（会动）
#   ./run_transplant.sh plan         只规划不动：用 locate 的结果列出夹取顺序和放哪个洞
#   ./run_transplant.sh run [N]      全自动移植，最多 N 株（默认放满圆盘为止）（会动）
#   ./run_transplant.sh stop         让正在跑的 run 在当前这一步结束后停下（另开一个终端运行）
#   ./run_transplant.sh open         松开夹爪（机械臂不动）
#   ./run_transplant.sh overview     只回总览位（会动）
#
# 建议顺序：check -> locate -> plan（看顺序对不对）-> run 1（先试一株）-> run
# 紧急情况直接按急停。报错要在 Desk 里清除，本脚本从不自动清除报错。
# 每次 run 的输出保存在 experiments/grasp/logs/，近距离照片保存在 experiments/grasp/closeups/。
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=.venvs/pipeline/bin/python
GRASP=experiments/grasp
mkdir -p "$GRASP/logs"

busy() {  # 本项目已有脚本在控制机械臂时不再开第二个连接
    local p
    p=$(pgrep -af "^([^ ]*/)?\.venvs/pipeline/bin/python[0-9.]* .*scripts/[0-9]" || true)
    if [ -n "$p" ]; then
        echo "已有本项目的脚本在运行，先等它结束（或用 ./run_transplant.sh stop）："
        echo "$p"
        exit 1
    fi
}

confirm() {
    echo "$1"
    echo "请确认：托盘和圆盘都在视野里、机械臂周围没有人和障碍、手放在急停上。"
    read -r -p "输入 y 开始，其他键取消: " a
    [ "$a" = "y" ] || { echo "已取消"; exit 0; }
}

cmd=${1:-help}
case "$cmd" in
    check)
        busy
        "$PY" scripts/20_robot_state.py
        ;;
    locate)
        busy
        confirm "机械臂将慢速回到总览位，然后拍照定位。"
        "$PY" -u scripts/68_goto_overview.py || { echo "回总览位失败，没有定位"; exit 1; }
        "$PY" -u scripts/60_tray_locate.py
        echo "定位结果: calibration/disc.json 和托盘文件；叠加图路径见上面 overlay 一行。下一步: ./run_transplant.sh plan"
        ;;
    plan)
        busy
        "$PY" -u scripts/72_transplant_all.py --dry-run
        ;;
    run)
        busy
        n=${2:-6}
        confirm "全自动移植开始，最多 $n 株。中途要停: 另开终端运行 ./run_transplant.sh stop"
        log="$GRASP/logs/run_$(date +%Y%m%d_%H%M%S).txt"
        "$PY" -u scripts/72_transplant_all.py --max-plants "$n" 2>&1 | tee "$log"
        echo "本次输出已保存: $log"
        ;;
    stop)
        touch "$GRASP/STOP"
        echo "已请求停止：当前这一步做完后停下（不是急停，紧急情况请按急停）"
        ;;
    open)
        busy
        "$PY" scripts/66_gripper.py open
        ;;
    overview)
        busy
        confirm "机械臂将先抬高，再慢速回到总览位。"
        "$PY" -u scripts/68_goto_overview.py
        ;;
    *)
        sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
