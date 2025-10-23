#!/bin/bash
# 查看和管理训练历史
# 使用方法: ./manage_training_history.sh [命令] [参数]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BACKUP_ROOT="$HOME/paper_v2/paper/marl_framework/training_history"

# 帮助信息
show_help() {
    echo "训练历史管理工具"
    echo ""
    echo "用法: $0 [命令] [参数]"
    echo ""
    echo "命令:"
    echo "  list              列出所有训练记录"
    echo "  show <name>       显示指定训练的详细信息"
    echo "  compare <n1> <n2> 比较两次训练"
    echo "  restore <name>    恢复指定训练到当前log目录"
    echo "  delete <name>     删除指定训练记录"
    echo "  clean [days]      清理N天前的记录(默认30天)"
    echo "  tensorboard <name> 启动指定训练的TensorBoard"
    echo ""
    echo "示例:"
    echo "  $0 list"
    echo "  $0 show exp_20251023_143000"
    echo "  $0 compare exp1 exp2"
    echo "  $0 restore exp_20251023_143000"
}

# 列出所有训练
list_trainings() {
    if [ ! -d "$BACKUP_ROOT" ]; then
        echo -e "${YELLOW}还没有任何训练记录${NC}"
        return
    fi
    
    COUNT=$(ls -1 "$BACKUP_ROOT" 2>/dev/null | wc -l)
    if [ $COUNT -eq 0 ]; then
        echo -e "${YELLOW}还没有任何训练记录${NC}"
        return
    fi
    
    echo -e "${BLUE}训练历史记录 (共${COUNT}个):${NC}"
    echo ""
    printf "%-30s %-20s %-10s %-15s\n" "实验名称" "日期" "大小" "模型"
    echo "--------------------------------------------------------------------------------"
    
    for exp_dir in $(ls -t "$BACKUP_ROOT"); do
        EXP_PATH="$BACKUP_ROOT/$exp_dir"
        
        # 获取日期
        if [ -f "$EXP_PATH/metadata.txt" ]; then
            DATE=$(grep "备份时间:" "$EXP_PATH/metadata.txt" | cut -d: -f2- | xargs)
        else
            DATE=$(stat -c %y "$EXP_PATH" | cut -d' ' -f1,2 | cut -d. -f1)
        fi
        
        # 获取大小
        SIZE=$(du -sh "$EXP_PATH" 2>/dev/null | cut -f1)
        
        # 检查是否有模型
        if [ -f "$EXP_PATH/best_model.pth" ]; then
            MODEL="${GREEN}✓${NC}"
        else
            MODEL="${RED}✗${NC}"
        fi
        
        printf "%-30s %-20s %-10s %-15b\n" "$exp_dir" "$DATE" "$SIZE" "$MODEL"
    done
    echo ""
}

# 显示训练详情
show_training() {
    local EXP_NAME=$1
    local EXP_PATH="$BACKUP_ROOT/$EXP_NAME"
    
    if [ ! -d "$EXP_PATH" ]; then
        echo -e "${RED}错误: 训练记录不存在: $EXP_NAME${NC}"
        return 1
    fi
    
    echo -e "${BLUE}=========================================="
    echo -e "  训练详情: $EXP_NAME"
    echo -e "==========================================${NC}"
    echo ""
    
    # 显示元数据
    if [ -f "$EXP_PATH/metadata.txt" ]; then
        cat "$EXP_PATH/metadata.txt"
        echo ""
    fi
    
    # 文件列表
    echo -e "${CYAN}文件列表:${NC}"
    ls -lh "$EXP_PATH" | tail -n +2 | awk '{printf "  %-30s %10s\n", $9, $5}'
    echo ""
    
    # 模型信息
    if [ -f "$EXP_PATH/best_model.pth" ]; then
        MODEL_SIZE=$(du -h "$EXP_PATH/best_model.pth" | cut -f1)
        echo -e "${GREEN}✓ 最佳模型: ${MODEL_SIZE}${NC}"
    else
        echo -e "${RED}✗ 无模型文件${NC}"
    fi
    
    # 检查点模型
    CHECKPOINTS=$(ls "$EXP_PATH"/best_model_*.pth 2>/dev/null | wc -l)
    if [ $CHECKPOINTS -gt 0 ]; then
        echo -e "${GREEN}✓ 检查点: ${CHECKPOINTS}个${NC}"
    fi
    
    # TensorBoard事件文件
    TB_FILES=$(ls "$EXP_PATH"/events.out.tfevents.* 2>/dev/null | wc -l)
    if [ $TB_FILES -gt 0 ]; then
        echo -e "${GREEN}✓ TensorBoard日志: ${TB_FILES}个文件${NC}"
    fi
    
    echo ""
}

# 恢复训练
restore_training() {
    local EXP_NAME=$1
    local EXP_PATH="$BACKUP_ROOT/$EXP_NAME"
    local LOG_DIR="$HOME/paper_v2/paper/marl_framework/log"
    
    if [ ! -d "$EXP_PATH" ]; then
        echo -e "${RED}错误: 训练记录不存在: $EXP_NAME${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}将恢复训练: $EXP_NAME${NC}"
    echo "  源: $EXP_PATH"
    echo "  目标: $LOG_DIR"
    echo ""
    
    # 检查当前log目录
    if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR)" ]; then
        echo -e "${RED}警告: 当前log目录不为空！${NC}"
        read -p "是否备份当前内容? (y/n): " backup_current
        if [ "$backup_current" = "y" ]; then
            TEMP_BACKUP="$BACKUP_ROOT/temp_$(date +%s)"
            mkdir -p "$TEMP_BACKUP"
            cp -r "$LOG_DIR"/* "$TEMP_BACKUP/"
            echo -e "${GREEN}✓ 已备份到: $TEMP_BACKUP${NC}"
        fi
    fi
    
    # 清理并恢复
    rm -rf "$LOG_DIR"/*
    cp -r "$EXP_PATH"/* "$LOG_DIR/"
    
    echo -e "${GREEN}✓ 恢复完成！${NC}"
    echo ""
}

# 删除训练记录
delete_training() {
    local EXP_NAME=$1
    local EXP_PATH="$BACKUP_ROOT/$EXP_NAME"
    
    if [ ! -d "$EXP_PATH" ]; then
        echo -e "${RED}错误: 训练记录不存在: $EXP_NAME${NC}"
        return 1
    fi
    
    SIZE=$(du -sh "$EXP_PATH" | cut -f1)
    echo -e "${RED}警告: 即将删除训练记录!${NC}"
    echo "  名称: $EXP_NAME"
    echo "  大小: $SIZE"
    echo ""
    read -p "确认删除? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        rm -rf "$EXP_PATH"
        echo -e "${GREEN}✓ 已删除${NC}"
    else
        echo "已取消"
    fi
}

# 清理旧记录
clean_old() {
    local DAYS=${1:-30}
    
    echo "查找${DAYS}天前的训练记录..."
    
    if [ ! -d "$BACKUP_ROOT" ]; then
        echo "无训练记录"
        return
    fi
    
    OLD_DIRS=$(find "$BACKUP_ROOT" -maxdepth 1 -type d -mtime +$DAYS | tail -n +2)
    COUNT=$(echo "$OLD_DIRS" | grep -c "^" || echo 0)
    
    if [ $COUNT -eq 0 ]; then
        echo "没有找到${DAYS}天前的记录"
        return
    fi
    
    echo "找到 $COUNT 个旧记录:"
    echo "$OLD_DIRS" | xargs -n1 basename | sed 's/^/  /'
    echo ""
    
    read -p "删除这些记录? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        echo "$OLD_DIRS" | xargs rm -rf
        echo -e "${GREEN}✓ 已清理 $COUNT 个旧记录${NC}"
    else
        echo "已取消"
    fi
}

# 启动TensorBoard
start_tensorboard() {
    local EXP_NAME=$1
    local EXP_PATH="$BACKUP_ROOT/$EXP_NAME"
    
    if [ ! -d "$EXP_PATH" ]; then
        echo -e "${RED}错误: 训练记录不存在: $EXP_NAME${NC}"
        return 1
    fi
    
    # 检查是否有TensorBoard文件
    TB_FILES=$(ls "$EXP_PATH"/events.out.tfevents.* 2>/dev/null | wc -l)
    if [ $TB_FILES -eq 0 ]; then
        echo -e "${RED}错误: 该训练没有TensorBoard日志${NC}"
        return 1
    fi
    
    echo -e "${GREEN}启动TensorBoard...${NC}"
    echo "日志目录: $EXP_PATH"
    echo ""
    echo "在浏览器中打开: http://localhost:6006"
    echo "按Ctrl+C停止"
    echo ""
    
    tensorboard --logdir "$EXP_PATH" --host 0.0.0.0 --port 6006
}

# 主逻辑
case "${1:-list}" in
    list)
        list_trainings
        ;;
    show)
        if [ -z "$2" ]; then
            echo "错误: 请指定训练名称"
            echo "用法: $0 show <name>"
            exit 1
        fi
        show_training "$2"
        ;;
    restore)
        if [ -z "$2" ]; then
            echo "错误: 请指定训练名称"
            echo "用法: $0 restore <name>"
            exit 1
        fi
        restore_training "$2"
        ;;
    delete)
        if [ -z "$2" ]; then
            echo "错误: 请指定训练名称"
            echo "用法: $0 delete <name>"
            exit 1
        fi
        delete_training "$2"
        ;;
    clean)
        clean_old "${2:-30}"
        ;;
    tensorboard|tb)
        if [ -z "$2" ]; then
            echo "错误: 请指定训练名称"
            echo "用法: $0 tensorboard <name>"
            exit 1
        fi
        start_tensorboard "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "错误: 未知命令 '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac
