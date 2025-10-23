#!/bin/bash
# GPU加速训练启动脚本 (Linux)
# 解决 GLIBCXX 版本冲突问题

echo "=========================================="
echo "  MARL GPU训练启动脚本"
echo "=========================================="
echo ""

# 设置库路径，优先使用conda环境中的库
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 显示环境信息
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python)"
echo "库路径: $LD_LIBRARY_PATH"
echo ""

# 验证GLIBCXX版本
echo "检查GLIBCXX版本..."
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep "GLIBCXX_3.4.29" > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ GLIBCXX_3.4.29 可用"
else
    echo "✗ GLIBCXX_3.4.29 未找到"
fi
echo ""

# 检查CUDA
echo "检查CUDA..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 开始训练
echo "=========================================="
echo "  开始训练..."
echo "=========================================="
echo ""

python main.py

echo ""
echo "=========================================="
echo "  训练完成"
echo "=========================================="
