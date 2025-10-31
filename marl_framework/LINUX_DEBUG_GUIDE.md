# Linux日志问题诊断指南

## 问题描述
在Linux环境下训练时，log目录中的日志文件为空，无法实时监控训练进度。

## 解决方案概述
我们已经实现了以下改进：

1. **FlushingFileHandler**: 自定义日志处理器，确保每条日志立即写入磁盘
2. **实时文件同步**: 使用 `os.fsync()` 强制Linux文件系统同步
3. **配置化路径**: 支持通过YAML配置自定义日志和结果存储路径
4. **实时进度保存**: 每50步保存一次训练进度到JSON文件

## 快速诊断步骤

### 1. 运行快速诊断脚本
```bash
cd ~/paper_search/paper/marl_framework
python3 quick_linux_debug.py
```

这个脚本会测试：
- 原生文件写入功能
- FlushingFileHandler功能
- 完整的logger设置
- 目录权限和磁盘空间

### 2. 运行完整诊断脚本
```bash
python3 debug_linux_logging.py
```

这个脚本会进行全面的系统检查。

### 3. 运行简单的日志测试
```bash
python3 test_linux_logging.py
```

## 训练监控命令

### 实时查看日志
```bash
# 查看最新的日志文件
tail -f log/log_*.log

# 或者查看所有日志文件
tail -f log/log_*.log
```

### 监控训练进度
```bash
# 每10秒查看一次训练进度
watch -n 10 'cat res/training_progress.json'

# 监控目录变化
watch -n 5 'ls -la log/ res/'
```

### 检查文件大小变化
```bash
# 监控日志文件大小
watch -n 2 'ls -lh log/log_*.log'
```

## 常见问题解决

### 问题1: 目录权限问题
```bash
# 检查目录权限
ls -la log/ res/

# 如果权限不足，修复权限
chmod 755 log/ res/
chmod 644 log/*.log res/*.json
```

### 问题2: 磁盘空间不足
```bash
# 检查磁盘空间
df -h .

# 清理旧的日志文件（如果需要）
find log/ -name "log_*.log" -mtime +7 -delete
```

### 问题3: Python环境问题
```bash
# 检查Python版本
python3 --version

# 检查模块导入
python3 -c "import constants; print('constants模块正常')"
python3 -c "from logger import setup_logger; print('logger模块正常')"
```

## 代码更新说明

### 更新的文件
1. **logger.py**: 
   - 实现了FlushingFileHandler类
   - 每次写入后立即flush()和fsync()
   - 确保日志实时写入磁盘

2. **constants.py**:
   - 支持配置化路径设置
   - 默认存储在marl_framework/log和marl_framework/res

3. **coma_mission.py**:
   - 每50步保存一次训练进度
   - TensorBoard每20步flush一次
   - 训练完成时标记完成状态

4. **main.py**:
   - 修复了logger初始化顺序
   - 避免重复设置logger

### 配置文件更新
在 `params.yaml` 或 `params_fast.yaml` 中添加：
```yaml
paths:
  log_dir: "log"        # 相对于marl_framework的路径
  results_dir: "res"    # 相对于marl_framework的路径
```

## 验证成功的标志

运行训练后，您应该看到：

1. **日志文件实时更新**:
   ```bash
   ls -la log/
   # 应该看到 log_TIMESTAMP.log 文件
   ```

2. **日志文件有内容**:
   ```bash
   tail log/log_*.log
   # 应该看到实时的训练日志
   ```

3. **训练进度文件**:
   ```bash
   cat res/training_progress.json
   # 应该看到最新的训练进度
   ```

4. **TensorBoard事件文件**:
   ```bash
   ls -la res/events.out.tfevents.*
   # 应该看到TensorBoard文件在增长
   ```

## 如果问题仍然存在

1. **检查系统日志**:
   ```bash
   dmesg | tail
   journalctl -u your_service --no-pager
   ```

2. **使用strace调试**:
   ```bash
   strace -e write python3 main.py
   ```

3. **检查文件句柄**:
   ```bash
   lsof | grep log
   ```

4. **联系支持**: 如果以上步骤都无法解决问题，请提供：
   - 诊断脚本的完整输出
   - 系统信息 (`uname -a`)
   - Python版本 (`python3 --version`)
   - 磁盘空间信息 (`df -h`)