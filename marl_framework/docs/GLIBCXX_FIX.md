# GLIBCXX 版本冲突解决方案

## 问题描述
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

## 原因
系统的 `libstdc++.so.6` 版本较旧，但conda环境中有新版本。程序默认加载了系统的旧版本。

## 解决方案

### 🚀 方案1: 快速启动（临时，推荐首次测试）

在Linux服务器上运行：

```bash
cd ~/paper_v2/paper/marl_framework

# 设置环境变量并运行
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python main.py
```

或者使用提供的脚本：

```bash
cd ~/paper_v2/paper/marl_framework
chmod +x run_training.sh
./run_training.sh
```

---

### ✅ 方案2: 永久修复（推荐，一劳永逸）

运行修复脚本：

```bash
cd ~/paper_v2/paper
chmod +x fix_glibcxx.sh
./fix_glibcxx.sh
```

按提示选择 `y` 来自动配置。

**或者手动配置**：

```bash
# 1. 创建conda激活脚本目录
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# 2. 创建环境变量脚本
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

# 3. 赋予执行权限
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 4. 重新激活环境
conda deactivate
conda activate paper_ipp_marl
```

配置完成后，每次激活环境都会自动设置正确的库路径。

---

### 🔧 方案3: 仅针对训练命令（备用）

创建一个别名：

```bash
# 添加到 ~/.bashrc
echo 'alias train_marl="export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && cd ~/paper_v2/paper/marl_framework && python main.py"' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc

# 之后直接运行
train_marl
```

---

## 验证修复

运行以下命令验证：

```bash
# 1. 激活环境
conda activate paper_ipp_marl

# 2. 检查库路径
echo $LD_LIBRARY_PATH

# 3. 验证GLIBCXX版本
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

# 4. 测试导入
python -c "import seaborn; import matplotlib; print('✓ 导入成功')"

# 5. 测试GPU
cd ~/paper_v2/paper
python test_gpu.py
```

如果都成功，说明修复完成！

---

## 快速命令参考

### 训练命令（修复后）

```bash
# 激活环境
conda activate paper_ipp_marl

# 进入目录
cd ~/paper_v2/paper/marl_framework

# 开始训练
python main.py

# 或使用脚本
./run_training.sh
```

### 监控GPU

在另一个终端：

```bash
watch -n 1 nvidia-smi
```

### 查看TensorBoard

```bash
cd ~/paper_v2/paper/marl_framework/log
tensorboard --logdir . --host 0.0.0.0 --port 6006
```

然后在本地浏览器访问: `http://服务器IP:6006`

---

## 常见问题

### Q: 为什么会有这个问题？
A: PIL库依赖的某些二进制文件需要较新的libstdc++，系统自带的版本太旧。

### Q: 这会影响其他程序吗？
A: 不会。这个设置只在paper_ipp_marl环境中生效，不影响系统或其他conda环境。

### Q: 每次都要设置吗？
A: 使用方案2（永久修复）后，每次激活环境都会自动设置，无需手动操作。

### Q: 如果还是报错怎么办？
A: 
1. 确认环境已激活: `conda info --envs`
2. 检查库路径: `echo $LD_LIBRARY_PATH`
3. 验证库存在: `ls -l $CONDA_PREFIX/lib/libstdc++.so.6`
4. 查看详细错误: `python -v main.py`

---

## 文件说明

- `fix_glibcxx.sh` - 自动修复脚本（交互式）
- `marl_framework/run_training.sh` - 训练启动脚本（包含环境设置）
- 本文档 - 问题说明和解决方案

---

**建议**: 使用方案2进行永久修复，之后就可以像平常一样直接运行 `python main.py` 了。
