#!/bin/bash
# Linux日志调试脚本

echo "=== MARL Framework Linux日志调试 ==="

# 检查当前目录
echo "📁 当前目录: $(pwd)"
echo "📁 目录内容:"
ls -la

# 检查Python环境
echo -e "\n🐍 Python环境:"
python3 --version
which python3

# 运行日志测试
echo -e "\n🧪 运行日志测试..."
python3 test_linux_logging.py

# 检查log目录
echo -e "\n📝 检查log目录:"
if [ -d "log" ]; then
    echo "✅ log目录存在"
    echo "📋 log目录权限: $(stat -c '%a' log)"
    echo "📄 log目录内容:"
    ls -la log/
    
    # 检查是否有日志文件
    if ls log/log_*.log 1> /dev/null 2>&1; then
        echo "✅ 找到日志文件:"
        for logfile in log/log_*.log; do
            echo "  📄 $logfile ($(stat -c '%s' "$logfile") 字节)"
            echo "     最新内容:"
            tail -n 5 "$logfile" | sed 's/^/     /'
        done
    else
        echo "❌ 没有找到日志文件"
    fi
else
    echo "❌ log目录不存在"
    echo "🔧 尝试创建log目录..."
    mkdir -p log
    if [ -d "log" ]; then
        echo "✅ log目录创建成功"
    else
        echo "❌ log目录创建失败"
    fi
fi

# 检查res目录
echo -e "\n📊 检查res目录:"
if [ -d "res" ]; then
    echo "✅ res目录存在"
    echo "📋 res目录权限: $(stat -c '%a' res)"
    echo "📄 res目录内容:"
    ls -la res/
else
    echo "❌ res目录不存在"
    echo "🔧 尝试创建res目录..."
    mkdir -p res
    if [ -d "res" ]; then
        echo "✅ res目录创建成功"
    else
        echo "❌ res目录创建失败"
    fi
fi

# 检查磁盘空间
echo -e "\n💾 磁盘空间检查:"
df -h .

# 检查Python包导入
echo -e "\n📦 检查Python包导入:"
python3 -c "
try:
    import constants
    print('✅ constants模块导入成功')
    print(f'   LOG_DIR: {constants.LOG_DIR}')
    print(f'   EXPERIMENTS_FOLDER: {constants.EXPERIMENTS_FOLDER}')
except Exception as e:
    print(f'❌ constants模块导入失败: {e}')

try:
    from logger import setup_logger
    print('✅ logger模块导入成功')
except Exception as e:
    print(f'❌ logger模块导入失败: {e}')
"

echo -e "\n💡 Linux训练监控命令:"
echo "# 实时查看最新日志:"
echo "tail -f log/log_*.log"
echo ""
echo "# 查看训练进度:"
echo "watch -n 10 'cat res/training_progress.json 2>/dev/null || echo \"进度文件尚未创建\"'"
echo ""
echo "# 监控目录变化:"
echo "watch -n 5 'ls -la log/ res/'"

echo -e "\n✅ 调试完成"