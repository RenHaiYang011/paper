# 项目重组指南

## 🎯 目标

将配置文件、脚本、文档分别整理到独立文件夹,使项目结构更清晰。

## 📁 新的目录结构

```
paper/
├── marl_framework/
│   ├── configs/          # ✨ 所有配置文件
│   │   ├── params.yaml
│   │   ├── params_balanced.yaml
│   │   ├── params_fast.yaml
│   │   ├── params_test.yaml
│   │   └── README.md
│   │
│   ├── scripts/          # ✨ 所有脚本
│   │   ├── train_with_backup.sh
│   │   ├── manage_training_history.sh
│   │   ├── run_training.sh
│   │   ├── fix_glibcxx.sh
│   │   └── README.md
│   │
│   ├── docs/             # ✨ 所有文档
│   │   ├── CONFIG_SELECTION_GUIDE.md
│   │   ├── GPU_BOTTLENECK_ANALYSIS.md
│   │   ├── TRAINING_LOG_MANAGEMENT.md
│   │   ├── GPU_TRAINING_GUIDE.md
│   │   ├── GPU_OPTIMIZATION.md
│   │   ├── GPU_UTILIZATION_FIX.md
│   │   ├── GLIBCXX_FIX.md
│   │   ├── TRAINING_OPTIMIZATION.md
│   │   └── README.md
│   │
│   ├── actor/
│   ├── agent/
│   ├── critic/
│   ├── mapping/
│   ├── missions/
│   ├── sensors/
│   ├── utils/
│   ├── main.py
│   ├── constants.py      # 已更新默认配置路径
│   └── ...
│
└── README.md
```

## 🚀 执行步骤

### 在 Git Bash (Windows) 或 Linux 上:

```bash
# 1. 确保在项目根目录
cd /e/code/paper_code/paper  # Windows Git Bash
# 或
cd ~/paper_v2/paper           # Linux

# 2. 给脚本执行权限
chmod +x reorganize_with_git.sh

# 3. 执行重组脚本
./reorganize_with_git.sh
```

脚本会:
- ✅ 使用 `git mv` 移动文件(保留Git历史)
- ✅ 自动创建 configs/、scripts/、docs/ 目录
- ✅ 为每个目录生成 README.md 说明文档
- ✅ 更新 constants.py 的默认配置路径

### 在 PowerShell (Windows):

如果你只在Windows上操作:

```powershell
# 1. 进入项目目录
cd E:\code\paper_code\paper

# 2. 使用 Git Bash 运行脚本
"C:\Program Files\Git\bin\bash.exe" reorganize_with_git.sh

# 或者使用 WSL
wsl bash reorganize_with_git.sh
```

## 📝 提交到Git

脚本执行完后:

```bash
# 1. 查看更改
git status

# 2. 添加所有更改
git add -A

# 3. 提交
git commit -m "refactor: reorganize project structure

- Move config files to configs/
- Move scripts to scripts/  
- Move docs to docs/
- Add README files for each directory
- Update constants.py default config path"

# 4. 推送到远程
git push
```

## ✅ 验证重组

```bash
# 查看新结构
tree -L 2 marl_framework/

# 或
ls -la marl_framework/configs/
ls -la marl_framework/scripts/
ls -la marl_framework/docs/
```

## 🔄 更新后的使用方式

### 训练命令更新

**之前**:
```bash
cd marl_framework
./train_with_backup.sh exp1
```

**之后**:
```bash
cd marl_framework/scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh exp1

# 或者使用相对路径
cd marl_framework
CONFIG_FILE_PATH=configs/params_balanced.yaml scripts/train_with_backup.sh exp1
```

### 配置文件路径更新

**constants.py 已自动更新**:
```python
# 旧: "params.yaml"
# 新: "configs/params.yaml"
CONFIG_FILE_PATH = load_from_env("CONFIG_FILE_PATH", str, "configs/params.yaml")
```

**环境变量使用**:
```bash
# 之前
export CONFIG_FILE_PATH=params_balanced.yaml

# 之后
export CONFIG_FILE_PATH=configs/params_balanced.yaml
```

## 🔍 如果遇到问题

### 问题1: 找不到配置文件

**错误**: `FileNotFoundError: configs/params.yaml`

**解决**:
```bash
# 确认文件存在
ls marl_framework/configs/

# 如果文件还在旧位置,手动移动
cd marl_framework
git mv params.yaml configs/params.yaml
git mv params_balanced.yaml configs/params_balanced.yaml
git commit -m "move config files"
```

### 问题2: 脚本路径错误

**错误**: `./train_with_backup.sh: No such file or directory`

**解决**:
```bash
# 使用新路径
cd marl_framework/scripts
./train_with_backup.sh exp1

# 或使用完整路径
marl_framework/scripts/train_with_backup.sh exp1
```

### 问题3: Git报错

**错误**: `fatal: not under version control`

**解决**:
```bash
# 确保在Git仓库根目录
cd /e/code/paper_code/paper
git status

# 如果文件已经移动,只需添加
git add -A
git commit -m "reorganize structure"
```

## 🎯 重组后的优势

### ✅ 结构清晰
- 配置、脚本、文档分离
- 易于查找和维护

### ✅ Git历史保留
- 使用 `git mv` 保留文件历史
- 可以追溯每个文件的变更记录

### ✅ 文档完善
- 每个目录都有README说明
- 新成员快速上手

### ✅ 向后兼容
- constants.py自动适配新路径
- 环境变量依然有效

## 📚 相关文档

重组后的文档位置:

- **配置说明**: `marl_framework/configs/README.md`
- **脚本说明**: `marl_framework/scripts/README.md`
- **文档索引**: `marl_framework/docs/README.md`
- **配置选择指南**: `marl_framework/docs/CONFIG_SELECTION_GUIDE.md`
- **GPU瓶颈分析**: `marl_framework/docs/GPU_BOTTLENECK_ANALYSIS.md`

## 💡 下一步

重组完成后:

1. **在Linux服务器上同步**:
```bash
cd ~/paper_v2/paper
git pull
```

2. **开始训练**:
```bash
cd marl_framework/scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh production_v1
```

3. **查看文档**:
```bash
cat marl_framework/docs/README.md
```

---

**准备好了吗?执行 `./reorganize_with_git.sh` 开始重组!** 🚀
