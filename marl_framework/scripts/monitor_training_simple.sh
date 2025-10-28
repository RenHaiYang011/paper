#!/bin/bash
# 简化的监控脚本 - 适用于 watch 命令
# 使用: watch -n 2 ./monitor_training_simple.sh

echo "=========================================="
echo "训练监控 [$(whoami)] - $(date '+%H:%M:%S')"
echo "=========================================="

# GPU 状态 (简化但完整)
echo ""
echo "📊 GPU 状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
    awk -F', ' '{printf "GPU%s: %4s使用 | %6s/%5s显存 | %2s°C | %6s功耗\n", $1, $2, $3, $4, $5, $6}'

# 我的 GPU 进程 (详细)
echo ""
echo "✨ 我的 GPU 进程:"
MY_FOUND=0
nvidia-smi pmon -c 1 2>/dev/null | grep -v "^#" | grep -v "no processes" | while read -r line; do
    gpu=$(echo "$line" | awk '{print $1}')
    pid=$(echo "$line" | awk '{print $2}')
    type=$(echo "$line" | awk '{print $3}')
    sm=$(echo "$line" | awk '{print $4}')
    mem=$(echo "$line" | awk '{print $5}')
    cmd=$(echo "$line" | awk '{print $6}')
    
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$user" = "$(whoami)" ]; then
        MY_FOUND=1
        # 获取完整命令
        full_cmd=$(ps -o args= -p "$pid" 2>/dev/null | head -c 50)
        # 获取运行时间
        runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
        # 获取 CPU 和内存
        cpu_mem=$(ps -o %cpu,%mem -p "$pid" 2>/dev/null | tail -1)
        
        printf "  GPU%s | PID:%s | 显存:%sMB | GPU使用:%s%% | 运行:%s\n" "$gpu" "$pid" "$mem" "$sm" "$runtime"
        printf "  命令: %s\n" "$full_cmd"
        printf "  CPU:%s | 内存:%s\n" $(echo $cpu_mem | awk '{print $1"%", $2"%"}')
    fi
done

# 如果没找到 GPU 进程,检查是否有 Python 进程
if [ $MY_FOUND -eq 0 ]; then
    PYTHON_PROC=$(ps aux | grep "$(whoami)" | grep "python.*main.py" | grep -v grep | head -1)
    if [ -n "$PYTHON_PROC" ]; then
        pid=$(echo "$PYTHON_PROC" | awk '{print $2}')
        cpu=$(echo "$PYTHON_PROC" | awk '{print $3}')
        mem=$(echo "$PYTHON_PROC" | awk '{print $4}')
        runtime=$(echo "$PYTHON_PROC" | awk '{print $10}')
        cmd=$(echo "$PYTHON_PROC" | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}' | head -c 50)
        echo "  ⚠ PID:$pid | CPU:${cpu}% | Mem:${mem}% | 运行:${runtime}"
        echo "  命令: $cmd"
        echo "  (GPU信息未获取,可能在CPU阶段)"
    else
        echo "  (无训练进程)"
    fi
fi

# Python 训练进程详情
echo ""
echo "🐍 Python main.py 进程:"
MY_TRAINING=$(ps aux | grep "$(whoami)" | grep "python.*main.py" | grep -v grep)
if [ -n "$MY_TRAINING" ]; then
    echo "$MY_TRAINING" | while read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        cpu=$(echo "$line" | awk '{print $3}')
        mem=$(echo "$line" | awk '{print $4}')
        vsz=$(echo "$line" | awk '{print $5}')
        rss=$(echo "$line" | awk '{print $6}')
        started=$(echo "$line" | awk '{print $9}')
        runtime=$(echo "$line" | awk '{print $10}')
        
        printf "  PID:%s | CPU:%s%% | Mem:%s%% | 开始:%s | 运行:%s\n" "$pid" "$cpu" "$mem" "$started" "$runtime"
        printf "  VSZ:%sKB | RSS:%sKB\n" "$vsz" "$rss"
    done
else
    echo "  (无 main.py 进程)"
fi

# 训练日志最新状态 (如果存在)
echo ""
echo "📄 最新训练日志:"
LATEST_LOG=$(ls -t ../log/log_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    tail -3 "$LATEST_LOG" 2>/dev/null | grep -E "Training step|Environment step|Episode" | tail -2
else
    echo "  (无日志文件)"
fi

echo ""

echo "=========================================="
echo "💡 'tail -f ../log/log_*.log' 查看完整日志"
