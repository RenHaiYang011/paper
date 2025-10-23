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
