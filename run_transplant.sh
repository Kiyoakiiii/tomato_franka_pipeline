#!/usr/bin/env bash
# 番茄苗移植（scripts/run.py 程序流程）。一条命令跑完整个流程：
#
#   ./run_transplant.sh              检查机械臂 -> [空格] 回总览位、定位一次 -> 按顺序移植，直到圆盘放满（最多 6 株）
#   ./run_transplant.sh run N        同上，最多 N 株
#   ./run_transplant.sh again N      不重新定位：用上一次 locate 的结果（先看过叠加图确认无误），最多 N 株
#
# 其他命令（单独调试用）：
#   ./run_transplant.sh check        查看机械臂状态和报错（不动，不清除报错）
#   ./run_transplant.sh locate       回总览位，拍照定位托盘和圆盘（会动）；看叠加图确认后可用 again 运行
#   ./run_transplant.sh plan         只规划不动：用 locate 的结果建程序，列出每株苗放哪个洞
#   ./run_transplant.sh stop         另开一个终端运行：当前这一步做完后停下
#   ./run_transplant.sh open         松开夹爪（机械臂不动）
#   ./run_transplant.sh overview     只回总览位（会动）
#
# 紧急情况直接按急停（stop 要等当前动作做完）。报错要在 Desk 里清除，本脚本从不自动清除报错。
# 每次运行的输出保存在 experiments/grasp/logs/。
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=.venvs/pipeline/bin/python
GRASP=experiments/grasp
mkdir -p "$GRASP/logs"

busy() {  # 本项目已有脚本在控制机械臂时不再开第二个连接
    local p
    p=$(pgrep -af "^([^ ]*/)?\.venvs/pipeline/bin/python[0-9.]* .*scripts/[a-z_]+\.py" || true)
    if [ -n "$p" ]; then
        echo "已有本项目的脚本在运行，先等它结束（或用 ./run_transplant.sh stop）："
        echo "$p"
        exit 1
    fi
}

pause_key() {  # 空格继续，q 退出
    local k
    echo "=== 下一步: $1   [空格 继续 / q 停止]"
    while true; do
        IFS= read -rsn1 k
        [ "$k" = " " ] && return 0
        case "$k" in q|Q) echo "已停止"; exit 0 ;; esac
    done
}

robot_ok() {  # 读一次状态（不动）：连不上或有报错就停
    local out
    out=$("$PY" scripts/robot_state.py --no-gripper 2>&1)
    if echo "$out" | grep -q "FCI\|NetworkException\|Traceback"; then
        echo "$out" | grep -E "Exception|Error" | tail -2
        echo "连不上机械臂：请在 Desk 里切到 Execution 模式并激活 FCI，然后重新运行"
        exit 1
    fi
    if echo "$out" | grep -q "has_errors=True"; then
        echo "机械臂有报错：请先在 Desk 里清除，然后重新运行"
        exit 1
    fi
    echo "机械臂已连接，没有报错"
}

transplant() {  # $1 最多几株，$2 其余参数（例如 --no-locate）
    busy
    robot_ok
    echo "开始前确认：托盘和圆盘都在视野里、机械臂周围没有人和障碍、手放在急停上。"
    pause_key "开始移植（最多 $1 株）"
    log="$GRASP/logs/run_$(date +%Y%m%d_%H%M%S).txt"
    "$PY" -u scripts/run.py --max-plants "$1" ${2:-} 2>&1 | tee "$log"
    echo "本次输出已保存: $log"
}

cmd=${1:-}
case "$cmd" in
    "")
        transplant 6
        ;;
    run)
        transplant "${2:-6}"
        ;;
    again)
        transplant "${2:-6}" --no-locate
        ;;
    check)
        busy
        "$PY" scripts/robot_state.py
        ;;
    locate)
        busy
        pause_key "机械臂将慢速回到总览位，然后拍照定位"
        "$PY" -u scripts/goto_overview.py || { echo "回总览位失败，没有定位"; exit 1; }
        "$PY" -u scripts/locate.py
        echo "定位结果: calibration/disc.json 和托盘文件；叠加图路径见上面 overlay 一行。看过叠加图无误后: ./run_transplant.sh again N"
        ;;
    plan)
        busy
        mkdir -p .tmp
        "$PY" -c 'import json,yaml;print(json.dumps({"q": yaml.safe_load(open("configs/pipeline.yaml"))["tray"]["overview_q"]}))' > .tmp/overview_q.json
        SEEDLING_SIM_Q=.tmp/overview_q.json "$PY" -u scripts/plan.py build 2>&1 | grep -v "closing point raised\|nanobind\|leaked"
        ;;
    stop)
        touch "$GRASP/STOP"
        echo "已请求停止：当前这一步做完后停下（不是急停，紧急情况请按急停）"
        ;;
    open)
        busy
        "$PY" scripts/gripper.py open
        ;;
    overview)
        busy
        pause_key "机械臂将先抬高，再慢速回到总览位"
        "$PY" -u scripts/goto_overview.py
        ;;
    *)
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
