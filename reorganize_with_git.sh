#!/bin/bash
# Git 安全的项目重组脚本
# 使用 git mv 命令保留文件历史记录

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  项目重组 - Git安全版本"
echo "=========================================="
echo ""

# 进入项目根目录
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)
MARL_DIR="$PROJECT_ROOT/marl_framework"

echo "项目路径: $PROJECT_ROOT"
echo ""

# 创建新目录
echo "1. 创建目录结构..."
mkdir -p "$MARL_DIR/configs"
mkdir -p "$MARL_DIR/docs"
mkdir -p "$MARL_DIR/scripts"
echo "   ✓ configs/"
echo "   ✓ docs/"
echo "   ✓ scripts/"
echo ""

# 移动配置文件 (使用 git mv 保留历史)
echo "2. 移动配置文件到 configs/ ..."
if [ -f "$MARL_DIR/params.yaml" ]; then
    git mv "$MARL_DIR/params.yaml" "$MARL_DIR/configs/params.yaml"
    echo "   ✓ params.yaml"
fi

if [ -f "$MARL_DIR/params_balanced.yaml" ]; then
    git mv "$MARL_DIR/params_balanced.yaml" "$MARL_DIR/configs/params_balanced.yaml"
    echo "   ✓ params_balanced.yaml"
fi

if [ -f "$MARL_DIR/params_fast.yaml" ]; then
    git mv "$MARL_DIR/params_fast.yaml" "$MARL_DIR/configs/params_fast.yaml"
    echo "   ✓ params_fast.yaml"
fi

if [ -f "$MARL_DIR/params_test.yaml" ]; then
    git mv "$MARL_DIR/params_test.yaml" "$MARL_DIR/configs/params_test.yaml"
    echo "   ✓ params_test.yaml"
fi
echo ""

# 移动脚本文件
echo "3. 移动脚本文件到 scripts/ ..."
if [ -f "$MARL_DIR/train_with_backup.sh" ]; then
    git mv "$MARL_DIR/train_with_backup.sh" "$MARL_DIR/scripts/train_with_backup.sh"
    echo "   ✓ train_with_backup.sh"
fi

if [ -f "$MARL_DIR/manage_training_history.sh" ]; then
    git mv "$MARL_DIR/manage_training_history.sh" "$MARL_DIR/scripts/manage_training_history.sh"
    echo "   ✓ manage_training_history.sh"
fi

if [ -f "$MARL_DIR/run_training.sh" ]; then
    git mv "$MARL_DIR/run_training.sh" "$MARL_DIR/scripts/run_training.sh"
    echo "   ✓ run_training.sh"
fi

if [ -f "$PROJECT_ROOT/fix_glibcxx.sh" ]; then
    git mv "$PROJECT_ROOT/fix_glibcxx.sh" "$MARL_DIR/scripts/fix_glibcxx.sh"
    echo "   ✓ fix_glibcxx.sh"
fi
echo ""

# 移动文档文件
echo "4. 移动文档文件到 docs/ ..."

# marl_framework下的文档
if [ -f "$MARL_DIR/TRAINING_LOG_MANAGEMENT.md" ]; then
    git mv "$MARL_DIR/TRAINING_LOG_MANAGEMENT.md" "$MARL_DIR/docs/TRAINING_LOG_MANAGEMENT.md"
    echo "   ✓ TRAINING_LOG_MANAGEMENT.md"
fi

if [ -f "$MARL_DIR/CONFIG_SELECTION_GUIDE.md" ]; then
    git mv "$MARL_DIR/CONFIG_SELECTION_GUIDE.md" "$MARL_DIR/docs/CONFIG_SELECTION_GUIDE.md"
    echo "   ✓ CONFIG_SELECTION_GUIDE.md"
fi

if [ -f "$MARL_DIR/GPU_BOTTLENECK_ANALYSIS.md" ]; then
    git mv "$MARL_DIR/GPU_BOTTLENECK_ANALYSIS.md" "$MARL_DIR/docs/GPU_BOTTLENECK_ANALYSIS.md"
    echo "   ✓ GPU_BOTTLENECK_ANALYSIS.md"
fi

# 项目根目录的文档
if [ -f "$PROJECT_ROOT/GPU_TRAINING_GUIDE.md" ]; then
    git mv "$PROJECT_ROOT/GPU_TRAINING_GUIDE.md" "$MARL_DIR/docs/GPU_TRAINING_GUIDE.md"
    echo "   ✓ GPU_TRAINING_GUIDE.md"
fi

if [ -f "$PROJECT_ROOT/GPU_OPTIMIZATION.md" ]; then
    git mv "$PROJECT_ROOT/GPU_OPTIMIZATION.md" "$MARL_DIR/docs/GPU_OPTIMIZATION.md"
    echo "   ✓ GPU_OPTIMIZATION.md"
fi

if [ -f "$PROJECT_ROOT/GPU_UTILIZATION_FIX.md" ]; then
    git mv "$PROJECT_ROOT/GPU_UTILIZATION_FIX.md" "$MARL_DIR/docs/GPU_UTILIZATION_FIX.md"
    echo "   ✓ GPU_UTILIZATION_FIX.md"
fi

if [ -f "$PROJECT_ROOT/GLIBCXX_FIX.md" ]; then
    git mv "$PROJECT_ROOT/GLIBCXX_FIX.md" "$MARL_DIR/docs/GLIBCXX_FIX.md"
    echo "   ✓ GLIBCXX_FIX.md"
fi

if [ -f "$PROJECT_ROOT/TRAINING_OPTIMIZATION.md" ]; then
    git mv "$PROJECT_ROOT/TRAINING_OPTIMIZATION.md" "$MARL_DIR/docs/TRAINING_OPTIMIZATION.md"
    echo "   ✓ TRAINING_OPTIMIZATION.md"
fi
echo ""

# 创建 configs/README.md
echo "5. 创建配置说明文档..."
cat > "$MARL_DIR/configs/README.md" << 'EOF'
# 训练配置文件

## 📁 配置文件说明

### params.yaml (默认/完整配置)
- **用途**: 完整训练,最佳性能
- **Budget**: 14 步
- **Episodes**: 1500
- **Batch size**: 64
- **训练时间**: 40-80 小时
- **推荐场景**: 论文实验、最终模型

### params_balanced.yaml ⭐ 推荐
- **用途**: 平衡配置,性价比最高
- **Budget**: 12 步
- **Episodes**: 1000
- **Batch size**: 48
- **训练时间**: 20-30 小时
- **推荐场景**: 生产部署、日常训练

### params_fast.yaml
- **用途**: 快速测试,验证代码
- **Budget**: 8 步
- **Episodes**: 500
- **Batch size**: 32
- **训练时间**: 10-15 小时
- **推荐场景**: 算法验证、调试

### params_test.yaml
- **用途**: 单元测试、CI/CD
- **Budget**: 4 步
- **Episodes**: 10
- **Batch size**: 8
- **训练时间**: <1 小时

## 🚀 使用方法

```bash
# 使用默认配置
python main.py

# 使用指定配置
CONFIG_FILE_PATH=configs/params_balanced.yaml python main.py

# 使用训练脚本
cd scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh exp_name
```

## ⚙️ 配置参数对比

| 参数 | params.yaml | params_balanced.yaml | params_fast.yaml |
|------|-------------|---------------------|------------------|
| budget | 14 | 12 | 8 |
| n_episodes | 1500 | 1000 | 500 |
| batch_size | 64 | 48 | 32 |
| data_passes | 5 | 3 | 3 |
| 训练步数 | ~4,800 | ~3,333 | ~2,083 |
| 预期时间 | 40-80h | 20-30h | 10-15h |
| 模型性能 | 100% | 93-95% | 80-85% |

## 📝 自定义配置

复制现有配置文件并修改:

```bash
cp params_balanced.yaml params_custom.yaml
# 编辑 params_custom.yaml
CONFIG_FILE_PATH=configs/params_custom.yaml python main.py
```

## 🔗 相关文档

- [配置选择指南](../docs/CONFIG_SELECTION_GUIDE.md)
- [训练优化](../docs/TRAINING_OPTIMIZATION.md)
- [GPU使用指南](../docs/GPU_TRAINING_GUIDE.md)
EOF

echo "   ✓ configs/README.md"
echo ""

# 创建 docs/README.md
echo "6. 创建文档索引..."
cat > "$MARL_DIR/docs/README.md" << 'EOF'
# 项目文档

## 📚 文档索引

### 训练相关

- **[CONFIG_SELECTION_GUIDE.md](CONFIG_SELECTION_GUIDE.md)** - 配置选择完整指南
  - 三种配置对比 (快速/平衡/完整)
  - 不同场景推荐配置
  - Budget参数详细分析

- **[TRAINING_LOG_MANAGEMENT.md](TRAINING_LOG_MANAGEMENT.md)** - 训练日志管理
  - 自动备份机制
  - 历史记录管理
  - TensorBoard使用

- **[TRAINING_OPTIMIZATION.md](TRAINING_OPTIMIZATION.md)** - 训练优化建议
  - 参数调优策略
  - 收敛加速方法
  - 常见问题解决

### GPU相关

- **[GPU_BOTTLENECK_ANALYSIS.md](GPU_BOTTLENECK_ANALYSIS.md)** ⭐ 重要
  - GPU低利用率根本原因
  - CPU瓶颈详细分析
  - 性能优化方案

- **[GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)** - GPU训练配置
  - GPU环境设置
  - CUDA配置
  - 多GPU使用

- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)** - GPU优化技巧
  - 混合精度训练
  - 显存优化
  - 批次大小调优

- **[GPU_UTILIZATION_FIX.md](GPU_UTILIZATION_FIX.md)** - GPU利用率修复
  - 诊断步骤
  - 常见问题
  - 解决方案

### 环境配置

- **[GLIBCXX_FIX.md](GLIBCXX_FIX.md)** - GLIBCXX库版本问题
  - Linux服务器库冲突
  - 永久解决方案
  - conda环境配置

## 🚀 快速开始

### 1. 新手入门

```bash
# 阅读顺序:
1. CONFIG_SELECTION_GUIDE.md  # 选择合适的配置
2. TRAINING_LOG_MANAGEMENT.md  # 了解训练流程
3. GPU_TRAINING_GUIDE.md       # 配置GPU环境
```

### 2. 遇到问题

| 问题 | 查阅文档 |
|------|---------|
| GPU利用率很低 | GPU_BOTTLENECK_ANALYSIS.md |
| 训练太慢 | TRAINING_OPTIMIZATION.md |
| 库版本冲突 | GLIBCXX_FIX.md |
| 显存不足 | GPU_OPTIMIZATION.md |
| 不知道用什么配置 | CONFIG_SELECTION_GUIDE.md |

### 3. 最佳实践

**推荐配置**: `configs/params_balanced.yaml`

```bash
cd ../scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh production_v1
```

**理由**:
- ✅ 训练时间合理 (20-30小时)
- ✅ 性能接近最优 (93-95%)
- ✅ GPU利用率相对较高 (~15%)
- ✅ 适合实际部署

## 📊 关键发现总结

### GPU利用率问题 (重要!)

```
问题: 4张RTX A6000,GPU利用率<10%
原因: CPU数据预处理瓶颈 (actor/transformations.py)
影响: 训练速度慢,硬件浪费90%

短期方案: 使用params_balanced.yaml (减少budget)
长期方案: 重构数据准备流程为GPU操作
```

详见: [GPU_BOTTLENECK_ANALYSIS.md](GPU_BOTTLENECK_ANALYSIS.md)

### 配置选择建议

```
快速测试:  params_fast.yaml (10-15h, 80-85%性能)
日常使用:  params_balanced.yaml (20-30h, 93-95%性能) ⭐推荐
论文发表:  params.yaml (40-80h, 100%性能)
```

详见: [CONFIG_SELECTION_GUIDE.md](CONFIG_SELECTION_GUIDE.md)

## 🔗 外部资源

- [PyTorch文档](https://pytorch.org/docs/)
- [COMA算法论文](https://arxiv.org/abs/1705.08926)
- [TensorBoard使用指南](https://www.tensorflow.org/tensorboard)

## 📝 文档更新

最后更新: 2025-01-23

如有问题或建议,请提issue或联系维护者。
EOF

echo "   ✓ docs/README.md"
echo ""

# 创建 scripts/README.md
echo "7. 创建脚本说明..."
cat > "$MARL_DIR/scripts/README.md" << 'EOF'
# 训练脚本

## 📜 脚本说明

### train_with_backup.sh ⭐ 主要训练脚本
**用途**: 自动备份+训练

**使用方法**:
```bash
# 基本用法
./train_with_backup.sh <实验名称>

# 使用指定配置
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh production_v1

# 示例
./train_with_backup.sh exp_baseline
./train_with_backup.sh test_reward_weights
```

**功能**:
- ✅ 自动备份旧的训练日志
- ✅ 保存到 `training_history/<实验名>`
- ✅ 备份配置文件
- ✅ 生成元数据
- ✅ 显示训练进度

### manage_training_history.sh
**用途**: 管理历史训练记录

**使用方法**:
```bash
# 交互式菜单
./manage_training_history.sh

# 功能:
1. 列出所有历史记录
2. 查看实验详情
3. 恢复历史模型
4. 删除旧记录
5. 对比不同实验
```

### run_training.sh
**用途**: 简单训练启动(无备份)

**使用方法**:
```bash
# 快速启动
./run_training.sh

# 使用指定配置
CONFIG_FILE_PATH=configs/params_fast.yaml ./run_training.sh
```

**注意**: 会覆盖现有日志,建议使用 `train_with_backup.sh`

### fix_glibcxx.sh
**用途**: 修复Linux服务器GLIBCXX库版本问题

**使用方法**:
```bash
# 一次性执行
./fix_glibcxx.sh

# 自动配置conda环境
source ~/.bashrc  # 重新加载
```

**功能**:
- ✅ 检测库版本冲突
- ✅ 配置LD_LIBRARY_PATH
- ✅ 永久写入conda激活脚本
- ✅ 验证修复效果

详见: [../docs/GLIBCXX_FIX.md](../docs/GLIBCXX_FIX.md)

## 🚀 完整训练流程

### 在Linux服务器上

```bash
# 1. 进入脚本目录
cd ~/paper_v2/paper/marl_framework/scripts

# 2. 给脚本执行权限(首次)
chmod +x *.sh

# 3. 修复库问题(首次)
./fix_glibcxx.sh

# 4. 选择配置并开始训练
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh production_v1

# 5. 监控训练
watch -n 1 nvidia-smi  # 另一个终端

# 6. 查看TensorBoard
cd ../log
tensorboard --logdir . --port 6006
```

### 在Windows本地开发

```powershell
# 进入脚本目录
cd E:\code\paper_code\paper\marl_framework\scripts

# 设置配置
$env:CONFIG_FILE_PATH = "configs\params_fast.yaml"

# 运行测试
python ..\main.py
```

## 📊 训练监控

### 实时监控GPU

```bash
# 方式1: nvidia-smi
watch -n 1 nvidia-smi

# 方式2: 详细信息
nvidia-smi dmon -i 0 -s pucvmet

# 方式3: 特定指标
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1
```

### 查看训练日志

```bash
# 实时日志
tail -f ../log/training.log

# 搜索错误
grep -i error ../log/training.log

# 查看TensorBoard
cd ../log
tensorboard --logdir .
```

### 管理历史记录

```bash
# 列出所有实验
./manage_training_history.sh

# 查看特定实验
cd ../training_history/<exp_name>
cat metadata.txt
ls -lh *.pth
```

## ⚙️ 配置优先级

```
1. 环境变量 CONFIG_FILE_PATH (最高)
2. constants.py 默认值 (configs/params.yaml)
```

示例:
```bash
# 方式1: 环境变量(推荐)
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh exp1

# 方式2: 修改constants.py
# CONFIG_FILE_PATH = "configs/params_balanced.yaml"

# 方式3: 临时环境变量
export CONFIG_FILE_PATH=configs/params_balanced.yaml
./train_with_backup.sh exp1
```

## 🔧 故障排除

### 问题1: 权限被拒绝
```bash
# 解决方案
chmod +x *.sh
```

### 问题2: 找不到配置文件
```bash
# 检查路径
ls ../configs/

# 使用正确路径(相对于marl_framework/)
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh exp1
```

### 问题3: GLIBCXX版本错误
```bash
# 运行修复脚本
./fix_glibcxx.sh

# 验证
python -c "import torch; print(torch.__version__)"
```

### 问题4: GPU不可用
```bash
# 检查CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 检查配置
grep -i device ../constants.py
```

## 📝 脚本自定义

### 修改训练脚本

```bash
# 编辑备份目录
vi train_with_backup.sh
# 修改: BACKUP_ROOT="$MARL_DIR/my_training_results"

# 修改确认提示
# 注释掉: read -p "按Enter继续..."
```

### 添加自定义脚本

```bash
# 创建新脚本
cat > my_training.sh << 'EOF'
#!/bin/bash
# 自定义训练流程
export CONFIG_FILE_PATH=configs/params_balanced.yaml
python ../main.py --custom-args
EOF

chmod +x my_training.sh
```

## 🔗 相关文档

- [配置文件说明](../configs/README.md)
- [训练日志管理](../docs/TRAINING_LOG_MANAGEMENT.md)
- [配置选择指南](../docs/CONFIG_SELECTION_GUIDE.md)

## 💡 最佳实践

1. **始终使用备份脚本**: `train_with_backup.sh` 而不是直接运行 `python main.py`
2. **实验命名规范**: 使用描述性名称,如 `baseline_v1`, `reward_tuning_exp3`
3. **配置版本控制**: 每次实验自动备份配置文件
4. **定期清理**: 使用 `manage_training_history.sh` 删除无用的历史记录
5. **监控资源**: 训练时开启 `nvidia-smi` 监控
EOF

echo "   ✓ scripts/README.md"
echo ""

echo "=========================================="
echo "  重组完成!"
echo "=========================================="
echo ""
echo "现在请提交更改到Git:"
echo ""
echo "  cd $PROJECT_ROOT"
echo "  git status"
echo "  git add -A"
echo '  git commit -m "refactor: reorganize project structure'
echo ''
echo '  - Move config files to configs/'
echo '  - Move scripts to scripts/'
echo '  - Move docs to docs/'
echo '  - Add README files for each directory"'
echo ""
echo "  git push"
echo ""
echo "新的目录结构:"
echo "  marl_framework/"
echo "  ├── configs/          # 所有配置文件"
echo "  ├── docs/             # 所有文档"
echo "  ├── scripts/          # 所有脚本"
echo "  ├── actor/"
echo "  ├── agent/"
echo "  ├── critic/"
echo "  └── ..."
echo ""
