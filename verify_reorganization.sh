#!/bin/bash
# 验证重组后的项目结构
# 使用方法: bash verify_reorganization.sh

echo "=========================================="
echo "  验证项目重组"
echo "=========================================="
echo ""

cd marl_framework

# 检查目录结构
echo "1. 检查目录结构..."
MISSING=0

if [ ! -d "configs" ]; then
    echo "✗ configs/ 目录不存在"
    MISSING=1
else
    echo "✓ configs/ 目录存在"
fi

if [ ! -d "docs" ]; then
    echo "✗ docs/ 目录不存在"
    MISSING=1
else
    echo "✓ docs/ 目录存在"
fi

if [ ! -d "scripts" ]; then
    echo "✗ scripts/ 目录不存在"
    MISSING=1
else
    echo "✓ scripts/ 目录存在"
fi

# 检查配置文件
echo ""
echo "2. 检查配置文件..."
for file in params.yaml params_balanced.yaml params_fast.yaml params_test.yaml; do
    if [ -f "configs/$file" ]; then
        echo "✓ configs/$file"
    else
        echo "✗ configs/$file 缺失"
        MISSING=1
    fi
done

# 检查文档
echo ""
echo "3. 检查文档..."
for file in TRAINING_LOG_MANAGEMENT.md CONFIG_SELECTION_GUIDE.md GPU_BOTTLENECK_ANALYSIS.md; do
    if [ -f "docs/$file" ]; then
        echo "✓ docs/$file"
    else
        echo "✗ docs/$file 缺失"
        MISSING=1
    fi
done

# 检查脚本
echo ""
echo "4. 检查脚本..."
for file in train_with_backup.sh manage_training_history.sh run_training.sh; do
    if [ -f "scripts/$file" ]; then
        echo "✓ scripts/$file"
        # 检查执行权限
        if [ -x "scripts/$file" ]; then
            echo "  ✓ 有执行权限"
        else
            echo "  ⚠ 缺少执行权限,运行: chmod +x scripts/$file"
        fi
    else
        echo "✗ scripts/$file 缺失"
        MISSING=1
    fi
done

# 检查旧文件是否还存在
echo ""
echo "5. 检查旧位置是否已清理..."
OLD_FILES=0
for file in params.yaml params_balanced.yaml params_fast.yaml train_with_backup.sh manage_training_history.sh; do
    if [ -f "$file" ]; then
        echo "⚠ 旧文件仍存在: $file (应该已移动到子目录)"
        OLD_FILES=1
    fi
done

if [ $OLD_FILES -eq 0 ]; then
    echo "✓ 旧文件已清理"
fi

# 测试Python导入
echo ""
echo "6. 测试Python配置加载..."
python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())
try:
    os.environ['CONFIG_FILE_PATH'] = 'configs/params.yaml'
    import constants
    print('✓ constants.py 可以正常导入')
    if 'configs' in constants.CONFIG_FILE_PATH:
        print('✓ 配置文件路径正确:', constants.CONFIG_FILE_PATH)
    else:
        print('✗ 配置文件路径不正确:', constants.CONFIG_FILE_PATH)
        sys.exit(1)
except Exception as e:
    print('✗ 导入失败:', str(e))
    sys.exit(1)
" 2>&1

# 总结
echo ""
echo "=========================================="
if [ $MISSING -eq 0 ] && [ $OLD_FILES -eq 0 ]; then
    echo "  ✓ 项目重组验证通过!"
    echo "=========================================="
    echo ""
    echo "可以安全提交:"
    echo "  git status"
    echo "  git commit -m 'refactor: reorganize project structure'"
    echo "  git push"
else
    echo "  ✗ 验证失败,请检查以上错误"
    echo "=========================================="
    exit 1
fi
echo ""
