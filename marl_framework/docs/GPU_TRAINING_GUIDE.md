# GPU加速训练 - 快速开始指南

## 🚀 已完成的优化

您的项目已经配置好GPU加速训练！以下是已完成的优化：

### ✅ 1. 硬件环境
- **GPU型号**: NVIDIA RTX A6000 × 4
- **CUDA版本**: 11.7
- **总显存**: 48GB per GPU
- **状态**: ✓ 已验证可用

### ✅ 2. 代码优化
- **主程序** (`marl_framework/main.py`): 启用GPU并显示详细信息
- **网络初始化** (`marl_framework/coma_wrapper.py`): 所有模型自动移动到GPU
- **混合精度训练**: Actor和Critic Learner已集成AMP加速
- **cuDNN优化**: 启用benchmark模式自动调优卷积

### ✅ 3. 配置优化 (`marl_framework/params.yaml`)
```yaml
networks:
  device: "cuda"      # 启用GPU
  batch_size: 128     # 优化批次大小
  batch_number: 5
  data_passes: 5
```

### ✅ 4. 奖励函数优化
```yaml
experiment:
  coverage_weight: 0.15   # 提高覆盖奖励
  distance_weight: 0.0    # 允许远距离探索
  footprint_weight: 0.5   # 中等重叠惩罚
  collision_weight: 2.0   # 强力避碰
  collision_distance: 5.0 # 5米安全距离
```

---

## 📖 使用指南

### 方法1: 自动化启动 (推荐)

```powershell
# 1. 启动训练 (带交互式确认)
.\start_training.ps1

# 2. 在另一个终端监控GPU
.\monitor_gpu.ps1
```

### 方法2: 手动启动

```powershell
# 1. 测试GPU环境
python test_gpu.py

# 2. 开始训练
cd marl_framework
python main.py

# 3. 监控GPU (另一个终端)
nvidia-smi -l 1
```

### 方法3: 查看TensorBoard

```powershell
cd marl_framework\log
tensorboard --logdir .
# 然后在浏览器打开 http://localhost:6006
```

---

## 📊 预期性能

### 训练时间对比
| 配置 | 预计时间 (1500 episodes) | 加速比 |
|------|-------------------------|--------|
| CPU | 20-30小时 | 1x |
| GPU (优化前) | 3-5小时 | 6-8x |
| **GPU (优化后)** | **1.5-3小时** | **10-15x** |

### GPU利用率目标
- **GPU利用率**: 70-90%
- **显存使用**: 8-16GB (48GB中)
- **批次吞吐量**: >100 samples/sec
- **每步耗时**: <1秒

---

## 🛠️ 工具脚本说明

### 1. `test_gpu.py` - GPU环境测试
**功能**:
- 检测CUDA可用性
- 测试GPU计算性能
- 验证CNN网络前向传播
- 测试项目网络导入

**使用**:
```bash
python test_gpu.py
```

**预期输出**:
```
CUDA可用: True
GPU 0: NVIDIA RTX A6000 (47.41 GB)
GPU加速比: 2-3x
CNN前向传播: ~10-20ms
✓ 所有网络已成功移动到GPU
```

### 2. `start_training.ps1` - 自动化训练启动
**功能**:
- 检查CUDA环境
- 显示训练配置
- 可选清理旧日志
- 启动训练并显示状态

**使用**:
```powershell
.\start_training.ps1
```

### 3. `monitor_gpu.ps1` - 实时GPU监控
**功能**:
- 实时显示GPU状态 (温度、利用率、显存)
- 监控训练进程
- 显示最新日志
- 自动刷新 (5秒间隔)

**使用**:
```powershell
.\monitor_gpu.ps1
```

---

## 🎯 训练参数说明

### 核心参数 (`marl_framework/params.yaml`)

```yaml
environment:
  x_dim: 50              # 地图大小 50x50米
  y_dim: 50
  seed: 3                # 随机种子

experiment:
  constraints:
    spacing: 5           # 移动步长5米
    budget: 14           # 每个episode 14步
    num_actions: 6       # 6个动作(上下左右+升降)
    
  missions:
    n_agents: 4          # 4架无人机
    n_episodes: 1500     # 训练1500个episode
    
  # 奖励权重 (已优化)
  coverage_weight: 0.15   # 覆盖新区域
  footprint_weight: 0.5   # 避免重叠
  collision_weight: 2.0   # 避免碰撞
  collision_distance: 5.0 # 安全距离5米

networks:
  device: "cuda"
  batch_size: 128        # GPU优化批次
  batch_number: 5
  data_passes: 5
  gamma: 0.99           # 折扣因子
  
  actor:
    learning_rate: 0.00001
    hidden_dim: 128
    
  critic:
    learning_rate: 0.0001
    fc1_dim: 64
```

---

## 📈 监控训练进度

### 1. 命令行输出
训练时会显示:
```
Training step: 150/1875, Step Time: 4.32s, ETA: 02:04:23
Environment step: 9600
```

### 2. TensorBoard指标
重要指标:
- `trainReturn/Episode/mean` - 训练回报
- `evalReturn/Episode/mean` - 评估回报 (每50步)
- `Actor/Loss` - Actor损失
- `Critic/Loss` - Critic损失
- `Sampled_actions` - 动作分布
- `Altitudes` - 高度选择分布

### 3. GPU监控
```powershell
nvidia-smi
```
关注:
- GPU利用率应该 >70%
- 显存使用 8-16GB
- 温度 <80°C

---

## 🔧 故障排查

### 问题1: CUDA Out of Memory
**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 在 params.yaml 中减小批次
networks:
  batch_size: 64  # 从128减到64
```

### 问题2: GPU利用率低 (<30%)
**可能原因**:
1. 批次太小 → 增加`batch_size`
2. 数据加载慢 → 检查数据预处理
3. CPU瓶颈 → 查看CPU使用率

**解决方案**:
```yaml
networks:
  batch_size: 256  # 尝试增大
```

### 问题3: 训练不收敛
**检查**:
1. 学习率是否合适
2. 奖励权重是否平衡
3. TensorBoard损失曲线

**调整**:
```yaml
networks:
  actor:
    learning_rate: 0.00005  # 调整学习率
```

### 问题4: 程序崩溃
**常见原因**:
1. 显存不足 → 减小batch_size
2. CUDA版本不匹配 → 检查PyTorch版本
3. 路径错误 → 确保在正确目录

---

## 📁 输出文件说明

### 训练输出
```
marl_framework/log/
├── best_model.pth           # 最佳模型
├── best_model_300.pth       # 300步检查点
├── best_model_400.pth       # 400步检查点
├── best_model_500.pth       # 500步检查点
├── best_model_600.pth       # 600步检查点
├── events.out.tfevents.*    # TensorBoard日志
└── plots/                   # 轨迹可视化
```

---

## 🎓 最佳实践

### 训练前
1. ✅ 运行 `python test_gpu.py` 验证环境
2. ✅ 检查 `params.yaml` 配置
3. ✅ 备份重要的旧模型
4. ✅ 启动GPU监控

### 训练中
1. ✅ 定期检查TensorBoard
2. ✅ 监控GPU温度和利用率
3. ✅ 观察损失和回报趋势
4. ✅ 每隔一段时间保存中间结果

### 训练后
1. ✅ 评估最佳模型性能
2. ✅ 分析TensorBoard数据
3. ✅ 保存重要的配置和结果
4. ✅ 记录超参数和性能指标

---

## 📚 进阶优化

### 如果需要更快的训练速度

#### 1. 使用更大的批次 (如果显存充足)
```yaml
networks:
  batch_size: 256  # 从128增加
```

#### 2. 减少日志频率
```yaml
logging:
  figure_interval: 50       # 减少绘图频率
  histogram_interval: 500   # 减少直方图频率
```

#### 3. 调整训练episode数
```yaml
missions:
  n_episodes: 1000  # 从1500减少
  patience: 50      # 更早保存模型
```

### 如果需要更好的性能

#### 1. 调整奖励权重
```yaml
experiment:
  coverage_weight: 0.2   # 增强覆盖
  collision_weight: 3.0  # 更强避碰
```

#### 2. 调整学习率
```yaml
actor:
  learning_rate: 0.00005  # 更激进
critic:
  learning_rate: 0.0005
```

---

## 🆘 获取帮助

如果遇到问题:
1. 查看 `GPU_OPTIMIZATION.md` 详细文档
2. 检查 `marl_framework/log/*.log` 日志文件
3. 运行 `python test_gpu.py` 诊断环境
4. 使用 `nvidia-smi` 检查GPU状态

---

## ✨ 快速命令参考

```powershell
# 测试GPU
python test_gpu.py

# 启动训练
.\start_training.ps1

# 监控GPU
.\monitor_gpu.ps1

# 监控GPU (命令行)
nvidia-smi -l 1

# TensorBoard
cd marl_framework\log
tensorboard --logdir .

# 检查进程
Get-Process python

# 清理显存
python -c "import torch; torch.cuda.empty_cache()"
```

---

**配置完成**: ✓  
**环境状态**: 就绪  
**下一步**: 运行 `.\start_training.ps1` 开始训练  
**预计完成**: 2-3小时

🎉 **祝训练顺利！**
