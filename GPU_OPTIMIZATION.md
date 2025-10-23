# GPU加速训练配置说明

## 系统信息
- **GPU型号**: NVIDIA RTX A6000 × 4
- **CUDA版本**: 11.7
- **GPU总内存**: ~48GB per GPU

## 已完成的优化

### 1. 主程序优化 (`marl_framework/main.py`)
✅ **GPU设备配置**
- 启用 `cuda:0` (使用第一块GPU)
- 启用 cuDNN benchmark 加速
- 显示GPU信息和内存状态

```python
constants.DEVICE = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True  # 自动优化卷积算法
torch.backends.cudnn.deterministic = False  # 允许非确定性以提高速度
```

### 2. 网络初始化优化 (`marl_framework/coma_wrapper.py`)
✅ **确保模型在GPU上**
- Actor Network → GPU
- Critic Network → GPU
- Target Critic Network → GPU

```python
self.actor_network = ActorNetwork(self.params).to(constants.DEVICE)
self.critic_network = CriticNetwork(self.params).to(constants.DEVICE)
self.target_critic_network = copy.deepcopy(self.critic_network).to(constants.DEVICE)
```

### 3. 批处理大小优化 (`marl_framework/params.yaml`)
✅ **增大批次以充分利用GPU**
```yaml
networks:
  batch_size: 128  # 从60增加到128
  batch_number: 5
  data_passes: 5
```

### 4. 混合精度训练 (AMP)
✅ **已集成在Learner中**
- `actor/learner.py`: 使用 `torch.cuda.amp.autocast`
- `critic/learner.py`: 使用 `torch.cuda.amp.GradScaler`

这可以提供约2-3倍的训练速度提升，同时减少内存使用。

### 5. 网络优化设置
✅ **已在代码中启用**
- `torch.autograd.set_detect_anomaly(False)` - 禁用异常检测以提速
- cuDNN自动调优 - 为固定输入大小优化卷积

## 预期性能提升

### 训练速度对比
| 配置 | 预计训练时间 (1500 episodes) | 加速比 |
|------|------------------------------|--------|
| CPU | ~20-30小时 | 1x |
| GPU (优化前) | ~3-5小时 | 5-8x |
| **GPU (优化后)** | **~1.5-3小时** | **10-15x** |

### GPU利用率目标
- **目标**: 70-90% GPU利用率
- **内存使用**: 预计8-16GB (48GB中)
- **批次处理**: 128 samples/batch × 5 batches

## 使用方法

### 1. 验证GPU环境
```powershell
cd E:\code\paper_code\paper
python test_gpu.py
```

这将测试:
- CUDA可用性
- GPU性能基准
- 项目网络加载

### 2. 开始训练
```powershell
cd marl_framework
python main.py
```

### 3. 监控GPU使用
在另一个终端运行:
```powershell
# 方法1: nvidia-smi (推荐)
nvidia-smi -l 1

# 方法2: 持续监控
watch -n 1 nvidia-smi
```

### 4. 查看训练日志
```powershell
# TensorBoard
cd marl_framework/log
tensorboard --logdir .
```

## 高级优化选项

### 选项1: 使用更大的批次 (如果内存充足)
```yaml
# params.yaml
networks:
  batch_size: 256  # 可尝试更大
  batch_number: 5
```

### 选项2: 数据并行 (未实现，需要时可添加)
如果想使用多块GPU进行数据并行:
```python
# 在coma_wrapper.py中
self.actor_network = torch.nn.DataParallel(
    ActorNetwork(self.params), 
    device_ids=[0, 1, 2, 3]
).to(constants.DEVICE)
```

### 选项3: 减少日志频率
```yaml
# params.yaml (如果添加此配置)
logging:
  figure_interval: 50      # 从20增加到50
  histogram_interval: 500  # 从200增加到500
```

## 故障排查

### 问题1: CUDA Out of Memory
**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小批次大小
   ```yaml
   batch_size: 64  # 从128减到64
   ```
2. 清理GPU缓存
   ```python
   torch.cuda.empty_cache()
   ```

### 问题2: GPU利用率低
**症状**: nvidia-smi显示利用率<30%

**可能原因**:
1. 数据加载瓶颈 - 检查是否在等待数据
2. 批次太小 - 增加batch_size
3. CPU预处理慢 - 优化数据预处理

### 问题3: 训练速度没有提升
**检查清单**:
- [ ] 确认params.yaml中 `device: "cuda"`
- [ ] 运行test_gpu.py验证CUDA可用
- [ ] 检查main.py日志是否显示"Using GPU"
- [ ] 使用nvidia-smi确认GPU有负载

## 性能监控指标

### 关键指标
1. **GPU利用率**: 目标 >70%
2. **GPU内存使用**: 目标 10-20GB
3. **训练步骤时间**: 目标 <1秒/step
4. **吞吐量**: 目标 >100 samples/sec

### TensorBoard监控
```
训练步骤时间
GPU内存使用
损失函数变化
奖励趋势
```

## 配置文件总结

### params.yaml - 关键GPU设置
```yaml
networks:
  device: "cuda"          # 启用GPU
  batch_size: 128         # GPU优化的批次大小
  batch_number: 5
  data_passes: 5
  
experiment:
  missions:
    n_agents: 4
    n_episodes: 1500
  constraints:
    budget: 14
```

## 训练时间估算

### 当前配置 (4 agents, budget=14, 1500 episodes)
- **总训练步数**: ~1,875 steps
  ```
  episodes × (batch_size × batch_number) / ((budget+1) × n_agents)
  = 1500 × (128 × 5) / (15 × 4)
  = 1500 × 640 / 60
  ≈ 16,000 环境步 / 60 = 约267轮批次训练
  ```

- **预计时间**: 2-3小时 (使用GPU)
- **每步时间**: ~4-6秒

## 最佳实践

1. ✅ **训练前运行test_gpu.py**
2. ✅ **使用nvidia-smi监控GPU**
3. ✅ **定期检查TensorBoard**
4. ✅ **保存检查点**
5. ✅ **记录超参数**

## 备注

- RTX A6000有48GB内存，可以支持更大的批次大小
- 混合精度训练(AMP)已启用，可获得额外加速
- 如需多GPU训练，可添加DataParallel包装器
- 当前配置针对单GPU优化，充分利用第一块GPU即可

---

**配置完成时间**: 2025年10月23日  
**测试状态**: ✓ 配置完成，等待运行验证  
**下一步**: 运行 `python test_gpu.py` 验证环境
