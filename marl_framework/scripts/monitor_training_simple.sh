#!/bin/bash
# 简化的监控脚本 - 适用于 watch 命令
# 使用: watch -n 2 ./monitor_training_simple.sh

echo "=========================================="
echo "训练监控 - $(date '+%H:%M:%S')"
echo "=========================================="

# GPU 状态 (简化)
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "  GPU%s: %3s util | %5s/%5s mem\n", $1, $2, $3, $4}'

# 我的 GPU 进程
echo ""
echo "我的 GPU 进程:"
MY_GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$MY_GPU_PROCS" ]; then
    echo "$MY_GPU_PROCS" | while IFS=',' read -r pid mem; do
        pid=$(echo $pid | tr -d ' ')
        mem=$(echo $mem | tr -d ' ')
        user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$user" = "$(whoami)" ]; then
            cmd=$(ps -o args= -p "$pid" 2>/dev/null | head -c 60)
            printf "  ✓ PID:%s | Mem:%s | %s\n" "$pid" "$mem" "$cmd"
        fi
    done
else
    echo "  (无进程)"
fi

# Python 进程 (仅我的)
echo ""
echo "Python 训练进程:"
ps aux | grep "$(whoami)" | grep "python.*main.py" | grep -v grep | \
    awk '{printf "  PID:%s | CPU:%s | Mem:%s | Time:%s\n", $2, $3, $4, $10}' | head -3

if [ $(ps aux | grep "$(whoami)" | grep "python.*main.py" | grep -v grep | wc -l) -eq 0 ]; then
    echo "  (无训练进程)"
fi

echo ""
echo "=========================================="
echo "提示: 'tail -f ../log/log_*.log' 查看日志"
