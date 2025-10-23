#!/bin/bash
# 更新代码中的路径引用
# 使用方法: bash update_paths.sh

set -e

echo "=========================================="
echo "  更新代码路径引用"
echo "=========================================="
echo ""

cd marl_framework

# 1. 更新 constants.py 中的配置文件路径
echo "1. 更新 constants.py..."
sed -i 's|CONFIG_FILE_PATH = load_from_env("CONFIG_FILE_PATH", str, "params.yaml")|CONFIG_FILE_PATH = load_from_env("CONFIG_FILE_PATH", str, "configs/params.yaml")|' constants.py
echo "✓ constants.py 已更新"

# 2. 更新 train_with_backup.sh 中的路径
echo ""
echo "2. 更新 scripts/train_with_backup.sh..."
sed -i 's|MARL_DIR="$HOME/paper_v2/paper/marl_framework"|MARL_DIR="$(cd "$(dirname "$0")/.." && pwd)"|' scripts/train_with_backup.sh
sed -i 's|grep -E "n_episodes:\|batch_size:\|n_agents:\|budget:" "$MARL_DIR/params.yaml"|grep -E "n_episodes:\|batch_size:\|n_agents:\|budget:" "$MARL_DIR/configs/params.yaml"|' scripts/train_with_backup.sh
sed -i 's|if \[ -f "$MARL_DIR/params.yaml" \]; then|if [ -f "$MARL_DIR/configs/params.yaml" ]; then|g' scripts/train_with_backup.sh
sed -i 's|cp "$MARL_DIR/params.yaml" "$BACKUP_DIR/params_backup.yaml"|cp "$MARL_DIR"/configs/params*.yaml "$BACKUP_DIR/"|' scripts/train_with_backup.sh
sed -i 's|配置文件: $(ls $MARL_DIR/params\*.yaml 2>/dev/null|配置文件: $(ls $MARL_DIR/configs/params*.yaml 2>/dev/null|' scripts/train_with_backup.sh
echo "✓ train_with_backup.sh 已更新"

# 3. 更新 manage_training_history.sh 中的路径
echo ""
echo "3. 更新 scripts/manage_training_history.sh..."
sed -i 's|MARL_DIR="$HOME/paper_v2/paper/marl_framework"|MARL_DIR="$(cd "$(dirname "$0")/.." && pwd)"|' scripts/manage_training_history.sh
echo "✓ manage_training_history.sh 已更新"

echo ""
echo "=========================================="
echo "  路径更新完成！"
echo "=========================================="
echo ""
echo "修改内容:"
echo "  ✓ constants.py: 默认配置路径 → configs/params.yaml"
echo "  ✓ train_with_backup.sh: 使用相对路径,自动查找项目根目录"
echo "  ✓ manage_training_history.sh: 使用相对路径"
echo ""
echo "⚠️  重要: 现在使用脚本时需要指定相对路径:"
echo ""
echo "  旧命令: ./train_with_backup.sh production_v1"
echo "  新命令: ./scripts/train_with_backup.sh production_v1"
echo ""
echo "  或者在scripts目录中运行:"
echo "  cd scripts && ./train_with_backup.sh production_v1"
echo ""
