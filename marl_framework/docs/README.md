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
