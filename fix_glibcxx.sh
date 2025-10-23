#!/bin/bash
# 快速修复脚本 - 设置环境变量解决GLIBCXX问题

echo "=========================================="
echo "  修复 GLIBCXX 版本冲突"
echo "=========================================="
echo ""

# 方案1: 临时环境变量（推荐）
echo "方案1: 设置临时环境变量"
echo "运行以下命令："
echo ""
echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "cd ~/paper_v2/paper/marl_framework"
echo "python main.py"
echo ""
echo "=========================================="
echo ""

# 方案2: 添加到conda环境激活脚本（永久）
echo "方案2: 永久修复（添加到conda环境）"
echo "运行以下命令："
echo ""
echo "mkdir -p \$CONDA_PREFIX/etc/conda/activate.d"
echo "echo 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' > \$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo ""
echo "然后重新激活环境："
echo "conda deactivate"
echo "conda activate paper_ipp_marl"
echo ""
echo "=========================================="
echo ""

# 方案3: 使用提供的脚本
echo "方案3: 使用启动脚本（最简单）"
echo "运行以下命令："
echo ""
echo "cd ~/paper_v2/paper/marl_framework"
echo "chmod +x run_training.sh"
echo "./run_training.sh"
echo ""
echo "=========================================="

# 询问用户选择
echo ""
read -p "是否现在执行方案2（永久修复）? (y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "正在配置..."
    
    # 创建目录
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    
    # 创建激活脚本
    cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
# 设置库路径，优先使用conda环境的库
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
    
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    
    echo "✓ 配置完成！"
    echo ""
    echo "请运行以下命令重新激活环境："
    echo "  conda deactivate"
    echo "  conda activate paper_ipp_marl"
    echo ""
    echo "然后就可以正常训练了："
    echo "  cd ~/paper_v2/paper/marl_framework"
    echo "  python main.py"
else
    echo ""
    echo "已取消。你可以使用方案1或方案3。"
fi
