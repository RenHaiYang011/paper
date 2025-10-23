#!/bin/bash
# 项目文件重组脚本 - 使用git mv保留历史记录
# 使用方法: bash reorganize_project.sh

set -e

echo "=========================================="
echo "  项目文件重组脚本"
echo "=========================================="
echo ""

cd marl_framework

# 创建目录结构
echo "1. 创建目录结构..."
mkdir -p configs
mkdir -p docs
mkdir -p scripts

# 移动配置文件到 configs/
echo ""
echo "2. 移动配置文件到 configs/..."
git mv params.yaml configs/params.yaml
git mv params_balanced.yaml configs/params_balanced.yaml
git mv params_fast.yaml configs/params_fast.yaml
git mv params_test.yaml configs/params_test.yaml
echo "✓ 配置文件已移动"

# 移动文档到 docs/
echo ""
echo "3. 移动文档到 docs/..."
git mv TRAINING_LOG_MANAGEMENT.md docs/
git mv CONFIG_SELECTION_GUIDE.md docs/
git mv GPU_BOTTLENECK_ANALYSIS.md docs/
echo "✓ 文档已移动"

# 移动脚本到 scripts/
echo ""
echo "4. 移动脚本到 scripts/..."
git mv train_with_backup.sh scripts/
git mv manage_training_history.sh scripts/
git mv run_training.sh scripts/
echo "✓ 脚本已移动"

echo ""
echo "=========================================="
echo "  文件移动完成！"
echo "=========================================="
echo ""
echo "新的目录结构:"
echo "marl_framework/"
echo "  ├── configs/           # 所有配置文件"
echo "  │   ├── params.yaml"
echo "  │   ├── params_balanced.yaml"
echo "  │   ├── params_fast.yaml"
echo "  │   └── params_test.yaml"
echo "  ├── docs/              # 所有文档"
echo "  │   ├── TRAINING_LOG_MANAGEMENT.md"
echo "  │   ├── CONFIG_SELECTION_GUIDE.md"
echo "  │   └── GPU_BOTTLENECK_ANALYSIS.md"
echo "  ├── scripts/           # 所有脚本"
echo "  │   ├── train_with_backup.sh"
echo "  │   ├── manage_training_history.sh"
echo "  │   └── run_training.sh"
echo "  └── ..."
echo ""
echo "⚠️  注意: 文件已通过 'git mv' 移动,保留了历史记录"
echo "⚠️  现在需要更新代码中的路径引用!"
echo ""
echo "下一步:"
echo "  1. 运行: bash update_paths.sh  (更新代码中的路径)"
echo "  2. 测试代码能否正常运行"
echo "  3. 提交: git commit -m 'refactor: reorganize project structure'"
echo "  4. 推送: git push"
echo ""
