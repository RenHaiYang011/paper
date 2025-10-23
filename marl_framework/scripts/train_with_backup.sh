#!/bin/bash
# 训练启动脚本 - 自动备份历史结果
# 使用方法: ./train_with_backup.sh [实验名称]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo -e "  MARL训练 - 自动备份版本"
echo -e "==========================================${NC}"
echo ""

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 获取实验名称
if [ -z "$1" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_NAME="exp_${TIMESTAMP}"
    echo -e "${YELLOW}未指定实验名称，使用时间戳: ${EXP_NAME}${NC}"
else
    EXP_NAME="$1"
    echo -e "${GREEN}实验名称: ${EXP_NAME}${NC}"
fi
echo ""

# 路径定义
MARL_DIR="$HOME/paper_v2/paper/marl_framework"
LOG_DIR="$MARL_DIR/log"
BACKUP_ROOT="$MARL_DIR/training_history"
BACKUP_DIR="$BACKUP_ROOT/${EXP_NAME}"

cd "$MARL_DIR"

# 检查log目录是否存在且非空
if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR)" ]; then
    echo -e "${YELLOW}检测到现有训练日志，开始备份...${NC}"
    
    # 创建备份目录
    mkdir -p "$BACKUP_DIR"
    
    # 统计文件
    NUM_FILES=$(ls -1 "$LOG_DIR" | wc -l)
    TOTAL_SIZE=$(du -sh "$LOG_DIR" | cut -f1)
    
    echo "  源目录: $LOG_DIR"
    echo "  备份到: $BACKUP_DIR"
    echo "  文件数: $NUM_FILES"
    echo "  总大小: $TOTAL_SIZE"
    
    # 执行备份（保留目录结构）
    cp -r "$LOG_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true
    
    # 验证备份
    if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR)" ]; then
        echo -e "${GREEN}✓ 备份完成！${NC}"
        
        # 显示备份的关键文件
        if [ -f "$BACKUP_DIR/best_model.pth" ]; then
            MODEL_SIZE=$(du -h "$BACKUP_DIR/best_model.pth" | cut -f1)
            echo "  ✓ best_model.pth (${MODEL_SIZE})"
        fi
        
        TB_FILES=$(ls "$BACKUP_DIR"/events.out.tfevents.* 2>/dev/null | wc -l)
        if [ $TB_FILES -gt 0 ]; then
            echo "  ✓ TensorBoard日志 (${TB_FILES}个文件)"
        fi
        
        # 创建元数据文件
        cat > "$BACKUP_DIR/metadata.txt" << EOF
实验名称: ${EXP_NAME}
备份时间: $(date +"%Y-%m-%d %H:%M:%S")
源目录: ${LOG_DIR}
文件数量: ${NUM_FILES}
总大小: ${TOTAL_SIZE}
配置文件: $(ls $MARL_DIR/params*.yaml 2>/dev/null | xargs -n1 basename)
EOF
        
        # 备份当前配置
        if [ -f "$MARL_DIR/params.yaml" ]; then
            cp "$MARL_DIR/params.yaml" "$BACKUP_DIR/params_backup.yaml"
            echo "  ✓ 配置文件已备份"
        fi
        
    else
        echo -e "${RED}✗ 备份失败${NC}"
    fi
    echo ""
    
    # 清理旧日志
    echo -e "${YELLOW}清理旧日志...${NC}"
    rm -rf "$LOG_DIR"/*
    echo -e "${GREEN}✓ 日志目录已清理${NC}"
    echo ""
else
    echo -e "${GREEN}日志目录为空，无需备份${NC}"
    echo ""
fi

# 显示历史备份
if [ -d "$BACKUP_ROOT" ]; then
    BACKUP_COUNT=$(ls -1 "$BACKUP_ROOT" | wc -l)
    if [ $BACKUP_COUNT -gt 0 ]; then
        echo -e "${BLUE}历史训练记录 (${BACKUP_COUNT}个):${NC}"
        ls -lht "$BACKUP_ROOT" | tail -n +2 | head -5 | while read line; do
            echo "  $(echo $line | awk '{print $9, "("$5")"}')"
        done
        if [ $BACKUP_COUNT -gt 5 ]; then
            echo "  ... 还有 $((BACKUP_COUNT-5)) 个"
        fi
        echo ""
    fi
fi

# 显示配置信息
echo -e "${BLUE}当前训练配置:${NC}"
if [ -f "$MARL_DIR/params.yaml" ]; then
    grep -E "n_episodes:|batch_size:|n_agents:|budget:" "$MARL_DIR/params.yaml" | sed 's/^/  /'
fi
echo ""

# 确认开始训练
echo -e "${YELLOW}准备开始训练...${NC}"
read -p "按Enter继续，或Ctrl+C取消: "
echo ""

# 开始训练
echo -e "${GREEN}=========================================="
echo -e "  开始训练: ${EXP_NAME}"
echo -e "==========================================${NC}"
echo ""

python main.py

# 训练完成
echo ""
echo -e "${GREEN}=========================================="
echo -e "  训练完成！"
echo -e "==========================================${NC}"
echo ""

# 显示结果位置
echo -e "${BLUE}训练结果:${NC}"
echo "  日志目录: $LOG_DIR"
echo "  备份目录: $BACKUP_DIR"
echo ""

# 检查最终模型
if [ -f "$LOG_DIR/best_model.pth" ]; then
    MODEL_SIZE=$(du -h "$LOG_DIR/best_model.pth" | cut -f1)
    echo -e "${GREEN}✓ 最佳模型已保存 (${MODEL_SIZE})${NC}"
fi

# 提示查看结果
echo ""
echo -e "${BLUE}查看训练结果:${NC}"
echo "  TensorBoard: cd $LOG_DIR && tensorboard --logdir ."
echo "  备份列表: ls -lh $BACKUP_ROOT"
echo ""
