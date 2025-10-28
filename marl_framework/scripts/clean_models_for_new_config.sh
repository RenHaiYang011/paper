#!/bin/bash
# 清理旧模型文件,用于改变配置后重新训练

echo "============================================"
echo "  清理旧模型 - 配置变更后使用"
echo "============================================"

cd ~/paper_v2/paper/marl_framework

# 备份现有模型
if [ -f "log/best_model.pth" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="model_backup_${timestamp}"
    mkdir -p "$backup_dir"
    
    echo ""
    echo "📦 备份现有模型到: $backup_dir"
    cp log/best_model*.pth "$backup_dir/" 2>/dev/null
    
    echo "✓ 备份完成"
    ls -lh "$backup_dir/"
fi

# 删除旧模型
echo ""
echo "🗑️  删除旧模型文件..."
rm -f log/best_model*.pth
rm -f checkpoints/*.pth

echo "✓ 清理完成"
echo ""
echo "============================================"
echo "  现在可以使用新配置训练了"
echo "============================================"
echo ""
echo "运行训练:"
echo "  export CONFIG_FILE_PATH=configs/params_fast.yaml"
echo "  ./scripts/train_with_backup.sh new_config_test"
