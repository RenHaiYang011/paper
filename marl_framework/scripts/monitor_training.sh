#!/bin/bash
# 训练进程监控脚本
# 用途: 实时监控 GPU 使用情况和训练进程状态

clear
echo "=========================================="
echo "  训练进程监控"
echo "  按 Ctrl+C 退出"
echo "=========================================="
echo ""

# 显示 GPU 状态
echo "📊 GPU 状态:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s\n  利用率: %s | 显存: %s / %s | 温度: %s | 功耗: %s\n\n", $1, $2, $3, $4, $5, $6, $7}'

echo ""
echo "=========================================="
echo "📌 GPU 进程详情"
echo "=========================================="

# 显示 GPU 进程
nvidia-smi pmon -c 1 2>/dev/null | grep -v "^#" | grep -v "no processes" | \
    awk 'NF {printf "GPU %s | PID: %s | Type: %s | GPU Util: %s%% | Mem: %sMB | CMD: %s\n", $1, $2, $3, $4, $5, $NF}'

# 如果没有进程
if [ $? -ne 0 ] || [ -z "$(nvidia-smi pmon -c 1 2>/dev/null | grep -v '^#')" ]; then
    echo "  (当前没有 GPU 进程)"
fi

echo ""
echo "=========================================="
echo "🐍 Python 训练进程 (当前用户: $(whoami))"
echo "=========================================="

# 显示当前用户的 Python 进程
ps aux | grep -E "$(whoami).*python" | grep -v grep | grep -v "monitor_training" | \
    awk '{printf "PID: %s | CPU: %s%% | MEM: %s%% | CMD: ", $2, $3, $4; for(i=11;i<=NF;i++) printf "%s ", $i; printf "\n"}'

# 如果没有 Python 进程
if [ ${PIPESTATUS[1]} -ne 0 ]; then
    echo "  (当前没有运行的 Python 进程)"
fi

echo ""
echo "=========================================="
echo "📁 项目相关进程"
echo "=========================================="

# 查找 paper 项目相关进程
PAPER_PROCS=$(ps aux | grep -E "paper.*python|main\.py" | grep -v grep | grep -v monitor_training)
if [ -n "$PAPER_PROCS" ]; then
    echo "$PAPER_PROCS" | awk '{
        printf "✓ PID: %s | User: %s | CPU: %s%% | MEM: %s%% | Started: %s %s\n", 
        $2, $1, $3, $4, $9, $10
        printf "  CMD: "
        for(i=11;i<=NF;i++) printf "%s ", $i
        printf "\n\n"
    }'
else
    echo "  (没有找到 paper 项目的训练进程)"
fi

echo "=========================================="
echo "💡 提示:"
echo "  - 使用 'watch -n 2 ./monitor_training.sh' 持续监控"
echo "  - 使用 'tail -f ../log/training.log' 查看训练日志"
echo "  - 使用 'kill <PID>' 停止特定进程"
echo "=========================================="
