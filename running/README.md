# Running 工具集

本目录包含用于运行和监控训练的辅助工具。

## 📜 文件说明

### monitor_gpu.ps1
**用途**: 实时监控 GPU 使用情况 (Windows PowerShell)

**使用方法**:
```powershell
# 在 PowerShell 中运行
.\running\monitor_gpu.ps1

# 或在项目根目录
powershell -File running\monitor_gpu.ps1
```

**功能**:
- 实时显示 GPU 利用率
- 显示显存使用情况
- 显示 GPU 温度和功耗
- 自动刷新 (类似 Linux 的 `watch -n 1 nvidia-smi`)

---

### start_training.ps1
**用途**: 在 Windows 上启动训练 (PowerShell 脚本)

**使用方法**:
```powershell
# 基本使用
.\running\start_training.ps1

# 或指定配置
$env:CONFIG_FILE_PATH = "marl_framework\configs\params_balanced.yaml"
.\running\start_training.ps1
```

**功能**:
- 自动激活 conda 环境
- 设置 Python 路径
- 启动训练脚本
- 错误处理和日志记录

---

### test_gpu.py
**用途**: 测试 GPU 环境和 PyTorch 安装

**使用方法**:
```bash
# 在项目根目录
python running/test_gpu.py

# 或使用 conda 环境
conda activate marl
python running/test_gpu.py
```

**测试内容**:
- ✅ PyTorch 版本
- ✅ CUDA 是否可用
- ✅ GPU 设备信息 (名称、数量、显存)
- ✅ cuDNN 版本
- ✅ 简单的 CNN 前向/反向传播测试
- ✅ GPU 计算性能测试

**预期输出**:
```
PyTorch Version: 1.13.0+cu117
CUDA Available: True
CUDA Version: 11.7
Device Count: 4
Current Device: 0
Device Name: NVIDIA RTX A6000
...
✓ All GPU tests passed!
```

---

## 🚀 典型使用场景

### 场景 1: Windows 本地开发测试

```powershell
# 1. 测试 GPU 环境
python running\test_gpu.py

# 2. 启动训练
$env:CONFIG_FILE_PATH = "marl_framework\configs\params_fast.yaml"
.\running\start_training.ps1

# 3. 监控 GPU (另一个终端)
.\running\monitor_gpu.ps1
```

### 场景 2: 验证环境配置

```powershell
# 测试 GPU 是否正常工作
python running\test_gpu.py

# 如果测试失败,检查:
# - PyTorch 版本是否匹配 CUDA 版本
# - NVIDIA 驱动是否安装
# - conda 环境是否正确激活
```

### 场景 3: 性能监控

```powershell
# 在一个终端启动训练
.\running\start_training.ps1

# 在另一个终端监控
.\running\monitor_gpu.ps1
```

---

## 💡 注意事项

### monitor_gpu.ps1
- **平台**: 仅限 Windows + PowerShell
- **要求**: 安装 NVIDIA GPU 驱动
- **等效命令**: 类似 Linux 的 `watch -n 1 nvidia-smi`

### start_training.ps1
- **平台**: 仅限 Windows + PowerShell
- **要求**: 正确配置 conda 环境
- **建议**: 在 Linux 服务器上使用 `marl_framework/scripts/train_with_backup.sh`

### test_gpu.py
- **平台**: 跨平台 (Windows/Linux)
- **要求**: PyTorch 和 CUDA 正确安装
- **用途**: 环境验证,不是训练脚本

---

## 🔄 Linux 服务器对应工具

如果你在 Linux 服务器上训练,使用以下工具:

| Windows 工具 | Linux 对应工具 | 说明 |
|-------------|---------------|------|
| `monitor_gpu.ps1` | `watch -n 1 nvidia-smi` | 监控 GPU |
| `start_training.ps1` | `marl_framework/scripts/train_with_backup.sh` | 启动训练 |
| `test_gpu.py` | `python running/test_gpu.py` | 测试 GPU (相同) |

### Linux 服务器训练推荐流程:

```bash
# 1. 测试 GPU
python running/test_gpu.py

# 2. 启动训练
cd marl_framework/scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh production_v1

# 3. 监控 GPU (另一个终端)
watch -n 1 nvidia-smi
```

---

## 📊 GPU 监控输出示例

### monitor_gpu.ps1 输出:
```
============================================
    GPU 监控 - 实时刷新
============================================

时间: 2025-01-23 20:30:15

GPU 0: NVIDIA RTX A6000
  利用率: 15%
  显存: 3072 MB / 49140 MB (6.3%)
  温度: 45°C
  功耗: 85W / 300W

GPU 1: NVIDIA RTX A6000
  利用率: 0%
  显存: 0 MB / 49140 MB (0%)
  温度: 35°C
  功耗: 25W / 300W

[按 Ctrl+C 退出]
```

### test_gpu.py 输出:
```
========================================
GPU Environment Test
========================================

PyTorch Version: 1.13.0+cu117
CUDA Available: True
CUDA Version: 11.7
cuDNN Version: 8500

Device Count: 4
Current Device: 0
Device Name: NVIDIA RTX A6000
Device Capability: 8.6
Total Memory: 48318 MB

========================================
Running CNN Test
========================================

Forward pass: ✓
Backward pass: ✓
GPU computation time: 0.025s

========================================
✓ All GPU tests passed!
========================================
```

---

## 🔗 相关文档

- **训练脚本**: [marl_framework/scripts/README.md](../marl_framework/scripts/README.md)
- **GPU 配置**: [marl_framework/docs/GPU_TRAINING_GUIDE.md](../marl_framework/docs/GPU_TRAINING_GUIDE.md)
- **性能分析**: [marl_framework/docs/GPU_BOTTLENECK_ANALYSIS.md](../marl_framework/docs/GPU_BOTTLENECK_ANALYSIS.md)

---

## 🛠️ 故障排除

### 问题 1: test_gpu.py 报错 "CUDA not available"

**原因**: PyTorch 没有正确安装 CUDA 支持

**解决**:
```bash
# 重新安装 PyTorch
pip uninstall torch
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 问题 2: monitor_gpu.ps1 无法运行

**错误**: "无法加载脚本,因为在此系统上禁止运行脚本"

**解决**:
```powershell
# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题 3: start_training.ps1 找不到 conda

**原因**: conda 环境未正确配置

**解决**:
```powershell
# 初始化 conda
conda init powershell
# 重启 PowerShell
conda activate marl
```

---

**提示**: 生产环境训练建议在 Linux 服务器上进行,Windows 工具主要用于本地开发和测试。
