# 训练日志管理说明

## ⚠️ 重要提示

**每次训练都会覆盖之前的日志和模型！**

代码中硬编码使用固定的`log/`目录，每次运行会覆盖：
- TensorBoard日志 (events.out.tfevents.*)
- 最佳模型 (best_model.pth)
- 检查点模型 (best_model_300.pth, best_model_400.pth等)
- 轨迹图像

---

## 🚀 解决方案：使用备份脚本

### 方案1: 自动备份训练（推荐）

使用提供的 `train_with_backup.sh` 脚本：

```bash
cd ~/paper_v2/paper/marl_framework

# 首次使用，赋予执行权限
chmod +x train_with_backup.sh

# 运行训练（自动备份）
./train_with_backup.sh 实验名称

# 示例
./train_with_backup.sh exp_baseline
./train_with_backup.sh exp_optimized_weights
./train_with_backup.sh exp_large_batch
```

**功能**:
- ✅ 自动检测并备份现有训练结果
- ✅ 创建时间戳命名的备份目录
- ✅ 备份所有模型、日志和配置
- ✅ 显示历史训练记录
- ✅ 自动设置环境变量（GLIBCXX）

---

### 方案2: 手动备份

在开始新训练前：

```bash
cd ~/paper_v2/paper/marl_framework

# 创建备份目录
BACKUP_NAME="exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p training_history/$BACKUP_NAME

# 备份日志和模型
cp -r log/* training_history/$BACKUP_NAME/

# 备份配置
cp params.yaml training_history/$BACKUP_NAME/params_backup.yaml

# 开始新训练
python main.py
```

---

## 📁 目录结构

使用备份脚本后的目录结构：

```
marl_framework/
├── log/                              # 当前训练（每次覆盖）
│   ├── best_model.pth
│   ├── best_model_300.pth
│   ├── best_model_400.pth
│   ├── events.out.tfevents.*
│   └── plots/
│
└── training_history/                 # 历史训练（自动备份）
    ├── exp_20251023_140000/
    │   ├── best_model.pth
    │   ├── events.out.tfevents.*
    │   ├── params_backup.yaml
    │   └── metadata.txt
    ├── exp_20251023_160000/
    ├── exp_baseline/
    └── exp_optimized/
```

---

## 🛠️ 训练历史管理工具

使用 `manage_training_history.sh` 管理历史记录：

```bash
# 赋予执行权限
chmod +x manage_training_history.sh

# 列出所有训练
./manage_training_history.sh list

# 查看训练详情
./manage_training_history.sh show exp_20251023_140000

# 恢复某次训练到当前log目录
./manage_training_history.sh restore exp_baseline

# 启动某次训练的TensorBoard
./manage_training_history.sh tensorboard exp_baseline

# 删除指定训练记录
./manage_training_history.sh delete exp_old

# 清理30天前的记录
./manage_training_history.sh clean 30

# 查看帮助
./manage_training_history.sh help
```

---

## 📊 完整训练流程示例

### 场景1: 基线实验

```bash
cd ~/paper_v2/paper/marl_framework

# 1. 启动训练（自动命名）
./train_with_backup.sh baseline_v1

# 2. 查看所有训练记录
./manage_training_history.sh list

# 3. 查看TensorBoard
tensorboard --logdir log --port 6006
```

### 场景2: 对比实验

```bash
# 实验1: 基线配置
./train_with_backup.sh exp_batch64

# 实验2: 修改配置后
nano params.yaml  # 修改batch_size=128
./train_with_backup.sh exp_batch128

# 比较结果
./manage_training_history.sh show exp_batch64
./manage_training_history.sh show exp_batch128

# 同时查看两个实验的TensorBoard
tensorboard --logdir training_history/exp_batch64:batch64,training_history/exp_batch128:batch128
```

### 场景3: 恢复最佳模型

```bash
# 查看历史训练
./manage_training_history.sh list

# 恢复之前的最佳训练
./manage_training_history.sh restore exp_baseline

# 继续在此基础上评估或微调
python evaluate.py  # 假设有评估脚本
```

---

## 💡 最佳实践

### 1. 命名规范

建议使用有意义的实验名称：

```bash
# 好的命名
./train_with_backup.sh baseline_4agents
./train_with_backup.sh collision_weight_2.0
./train_with_backup.sh batch128_lr0.0001

# 避免
./train_with_backup.sh test
./train_with_backup.sh exp1
```

### 2. 定期清理

```bash
# 每月清理一次旧记录
./manage_training_history.sh clean 30

# 或手动删除不需要的
./manage_training_history.sh delete exp_test123
```

### 3. 记录实验配置

在元数据中记录关键参数：

```bash
# 在训练前记录
echo "实验说明: 测试新的奖励权重配置" > training_note.txt
echo "coverage_weight: 0.15" >> training_note.txt
echo "collision_weight: 2.0" >> training_note.txt
```

---

## 🔍 查看训练结果

### TensorBoard对比

```bash
# 对比多个实验
tensorboard --logdir \
  training_history/exp1:实验1,\
  training_history/exp2:实验2,\
  training_history/exp3:实验3 \
  --port 6006
```

### 提取最佳模型

```bash
# 找到最佳训练
./manage_training_history.sh list

# 复制最佳模型
cp training_history/exp_best/best_model.pth ../models/best_model_final.pth
```

---

## ⚙️ 自动化脚本说明

### train_with_backup.sh

**功能**:
- 自动检测现有日志
- 创建时间戳备份
- 备份模型、日志和配置
- 设置环境变量
- 启动训练
- 显示训练历史

**位置**: `marl_framework/train_with_backup.sh`

### manage_training_history.sh

**功能**:
- 列出所有训练记录
- 查看训练详情
- 恢复历史训练
- 启动TensorBoard
- 删除旧记录
- 清理过期备份

**位置**: `marl_framework/manage_training_history.sh`

---

## 🚨 常见问题

### Q1: 忘记备份，已经覆盖了怎么办？
A: 如果已经覆盖，旧数据无法恢复。建议：
- 立即停止新训练（Ctrl+C）
- 至少保存当前的配置文件
- 以后使用备份脚本

### Q2: 备份占用太多空间？
A: 
```bash
# 查看备份大小
du -sh training_history/*

# 清理旧备份
./manage_training_history.sh clean 30

# 仅保留最佳模型，删除TensorBoard日志
rm training_history/exp_old/events.out.tfevents.*
```

### Q3: 如何对比不同训练？
A:
```bash
# 方法1: TensorBoard对比
tensorboard --logdir training_history/

# 方法2: 使用管理脚本
./manage_training_history.sh show exp1
./manage_training_history.sh show exp2
```

### Q4: 脚本权限问题？
A:
```bash
# 赋予执行权限
chmod +x train_with_backup.sh
chmod +x manage_training_history.sh

# 验证
ls -l *.sh
```

---

## 📝 快速参考

### 训练命令

```bash
# 带备份的训练
./train_with_backup.sh 实验名称

# 普通训练（会覆盖）
python main.py
```

### 管理命令

```bash
# 查看列表
./manage_training_history.sh list

# 查看详情
./manage_training_history.sh show <name>

# 恢复训练
./manage_training_history.sh restore <name>

# TensorBoard
./manage_training_history.sh tensorboard <name>
```

---

**建议**: 始终使用 `train_with_backup.sh` 进行训练，避免重要结果被覆盖！
