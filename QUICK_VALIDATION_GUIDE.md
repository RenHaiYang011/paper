# 🚀 快速验证指南

## 快速开始

### 方式1: 使用配置文件参数 (推荐)
```powershell
cd marl_framework
python main.py --config configs/params_quick_test.yaml
```

### 方式2: 设置环境变量
```powershell
cd marl_framework
$env:CONFIG_FILE_PATH="configs/params_quick_test.yaml"
python main.py
```

### 方式3: 修改constants.py默认配置
在 `marl_framework/constants.py` 中修改:
```python
CONFIG_FILE_PATH = "configs/params_quick_test.yaml"
```

---

## ⚡ 三种验证配置对比

| 配置 | Episodes | Budget | Agents | Batch | 预计时间 | 性能 | 用途 |
|------|----------|--------|--------|-------|----------|------|------|
| **params_quick_test.yaml** | 100 | 30 | 2 | 16 | 1-2小时 | 60-70% | 🚀 **功能快速验证** |
| **params_fast.yaml** | 500 | 8 | 4 | 32 | 8-12小时 | 80-85% | ⚡ 快速训练测试 |
| **params.yaml** | 1500 | 50 | 4 | 128 | 24-36小时 | 95-100% | 🎯 完整性能训练 |

---

## 🎯 快速验证配置详情 (params_quick_test.yaml)

### 核心参数
```yaml
Episodes: 100        # 仅100轮，快速验证
Budget: 30步         # 每轮30步
Agents: 2个          # 2智能体减少计算
Batch Size: 16       # 最小batch
Data Passes: 1       # 单次数据遍历
区域: 2个           # 高优先级+中优先级
障碍物: 5个         # 关键位置验证
```

### 优化策略
```yaml
学习率: 提高 (0.0001 actor, 0.0005 critic)
奖励强度: 提高 (150.0目标发现)
探索率: 提高 (0.6 max, 0.1 min)
目标密度: 提高 (0.7高优先级)
阈值: 降低 (0.75发现, 0.80完成)
```

### 完整功能保留
- ✅ 13通道观测空间
- ✅ 6层级奖励架构
- ✅ 前沿驱动探索
- ✅ 三重协同机制
- ✅ 区域优先搜索
- ✅ 障碍物避障
- ✅ 27动作空间

---

## 📊 验证检查清单

### 1. 训练过程监控
```powershell
# 实时监控GPU使用
nvidia-smi -l 1

# 监控训练日志
Get-Content -Path "marl_framework/log/training.log" -Wait -Tail 50
```

### 2. 功能验证项

#### ✅ 多层级奖励验证
查看日志中的奖励分解:
```
Reward breakdown:
  - Utility: 0.XX
  - Discovery: XX.XX  (目标发现时应有大奖励)
  - Frontier: X.XX
  - Region: X.XX
  - Coordination: X.XX
  - Penalty: -X.XX
```

#### ✅ 前沿探索验证
查看前沿检测输出:
```
Frontier points detected: XX
Agent 0 distance to frontier: X.XX
Frontier reward: X.XX
```

#### ✅ 协同机制验证
查看协同指标:
```
Coordination metrics:
  - Overlap penalty: 0.XX
  - Division quality: 0.XX
  - Collaboration events: X
```

#### ✅ 区域搜索验证
查看区域覆盖:
```
Region coverage:
  - High priority: XX%
  - Medium priority: XX%
```

#### ✅ 障碍物避障验证
查看碰撞统计:
```
Collision stats:
  - Collisions avoided: XX
  - Safety margin violations: X
```

#### ✅ 目标发现验证
查看发现记录:
```
Target discovered at step XX by agent X
Total targets found: X/5
Discovery rate: XX%
```

---

## 🔍 验证成功标准

### 最低标准 (1-2小时训练)
- ✅ 训练正常运行无错误
- ✅ 所有奖励模块正常计算
- ✅ 至少发现1-2个目标
- ✅ 避障功能正常工作
- ✅ 协同指标有数值输出
- ✅ 学习曲线显示收敛趋势

### 理想标准
- ✅ 平均奖励呈上升趋势
- ✅ 目标发现率 > 40%
- ✅ 碰撞率 < 5%
- ✅ 区域覆盖率 > 60%
- ✅ 协同质量 > 0.5

---

## 📈 快速分析结果

### 训练完成后检查

#### 1. 查看学习曲线
```python
# 在results目录查看:
# - episode_rewards.png
# - target_discovery_rate.png
# - coordination_quality.png
```

#### 2. 查看最终轨迹
```python
# 3D轨迹可视化应显示:
# - 智能体轨迹 (不同颜色)
# - 目标位置 (红色球体)
# - 障碍物 (灰色圆柱)
# - 搜索区域 (半透明框)
```

#### 3. 查看性能指标
```python
# 查看最终统计:
cd marl_framework/res
cat final_metrics.txt
```

---

## 🐛 常见问题排查

### 问题1: CUDA内存不足
```yaml
# 解决: 进一步减小batch size
batch_size: 8  # 从16改为8
n_agents: 2    # 保持2个智能体
```

### 问题2: 训练速度仍慢
```yaml
# 解决: 进一步减少episodes
n_episodes: 50   # 从100改为50
budget: 20       # 从30改为20
```

### 问题3: 奖励不收敛
```yaml
# 解决: 提高学习率
actor:
  learning_rate: 0.0002  # 提高到0.0002
critic:
  learning_rate: 0.001   # 提高到0.001
```

### 问题4: 目标发现率低
```yaml
# 解决: 提高目标密度和奖励
target_probability: 0.8           # 更多目标
target_discovery_reward: 200.0    # 更大奖励
```

---

## 🎯 验证流程建议

### 阶段1: 功能验证 (1-2小时)
```powershell
# 使用quick_test配置
python main.py --config configs/params_quick_test.yaml
```
**目标**: 确认所有功能正常，无bug

### 阶段2: 快速训练 (8-12小时) 
```powershell
# 功能确认后，使用fast配置
python main.py --config configs/params_fast.yaml
```
**目标**: 获得80-85%性能，验证算法有效性

### 阶段3: 完整训练 (24-36小时)
```powershell
# 准备发表论文前，使用标准配置
python main.py --config configs/params.yaml
```
**目标**: 获得95-100%最优性能

---

## 📊 预期输出文件

### 训练过程文件
```
marl_framework/log/
  ├── training.log              # 详细训练日志
  ├── rewards_history.csv       # 奖励历史
  └── metrics_history.csv       # 各项指标历史
```

### 结果文件
```
marl_framework/res/
  ├── episode_rewards.png       # 学习曲线
  ├── trajectories_3d.png       # 3D轨迹
  ├── coverage_map.png          # 覆盖率图
  ├── coordination_metrics.png  # 协同质量
  ├── final_metrics.txt         # 最终统计
  └── model_checkpoints/        # 模型检查点
```

---

## 💡 快速验证技巧

### 1. 实时监控关键指标
```python
# 可以修改logger输出频率查看更多细节
# 在 marl_framework/logger.py 中调整
```

### 2. 中断后继续训练
```python
# 训练支持断点恢复
# 模型会自动保存在res/checkpoints/
```

### 3. 对比不同配置
```powershell
# 可以同时运行多个配置对比
# 使用不同的results_dir避免冲突
```

### 4. GPU监控脚本
```powershell
# 使用提供的监控脚本
.\running\monitor_gpu.ps1
```

---

## 🚀 开始验证

推荐验证流程:
```powershell
# 1. 进入项目目录
cd E:\code\paper_code\paper

# 2. 激活Python环境 (如果有虚拟环境)
# conda activate your_env

# 3. 运行快速验证
cd marl_framework
python main.py --config configs/params_quick_test.yaml

# 4. 监控训练 (新开终端)
Get-Content -Path "log/training.log" -Wait -Tail 30

# 5. 等待1-2小时查看结果
# 训练完成后查看 res/ 目录下的结果文件
```

预计1-2小时后，您应该能看到:
- ✅ 完整的训练日志
- ✅ 学习曲线图
- ✅ 3D轨迹可视化  
- ✅ 所有功能验证通过
- ✅ 初步的性能指标

**祝验证顺利！** 🎉
