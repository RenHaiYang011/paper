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
echo "📌 GPU 进程详情 (所有用户)"
echo "=========================================="

# 显示所有 GPU 进程
ALL_PROCS=$(nvidia-smi pmon -c 1 2>/dev/null | grep -v "^#" | grep -v "no processes")
if [ -n "$ALL_PROCS" ]; then
    echo "$ALL_PROCS" | awk 'NF {
        gpu=$1; pid=$2; type=$3; sm=$4; mem=$5; cmd=$NF
        
        # 获取进程用户
        user=$(ps -o user= -p '"$pid"' 2>/dev/null || echo "unknown")
        
        # 标记当前用户的进程
        if [ "$user" = "$(whoami)" ]; then
            printf "✓ [我的] "
        else
            printf "  [%s] ", user
        fi
        
        printf "GPU %s | PID: %s | Type: %s | GPU Util: %s%% | Mem: %sMB | CMD: %s\n", gpu, pid, type, sm, mem, cmd
    }'
else
    echo "  (当前没有 GPU 进程)"
fi

echo ""
echo "=========================================="
echo "✨ 我的 GPU 进程"
echo "=========================================="

# 只显示当前用户的 GPU 进程
MY_GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$MY_GPU_PROCS" ]; then
    echo "$MY_GPU_PROCS" | while IFS=, read -r pid cmd mem; do
        proc_user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$proc_user" = "$(whoami)" ]; then
            # 找出在哪个GPU上
            gpu_id=$(nvidia-smi pmon -c 1 2>/dev/null | grep "$pid" | awk '{print $1}')
            echo "  GPU $gpu_id | PID: $pid | 显存: $mem | 命令: $cmd"
            
            # 显示完整命令
            full_cmd=$(ps -o cmd= -p "$pid" 2>/dev/null)
            if [ -n "$full_cmd" ]; then
                echo "    完整命令: $full_cmd"
            fi
        fi
    done
else
    echo "  (你当前没有 GPU 进程)"
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
