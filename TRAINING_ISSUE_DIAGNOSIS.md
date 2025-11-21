# 🔍 训练问题诊断报告 - 更新版

## 问题描述
训练300步后，航线轨迹仍然没有变化，表明模型**没有学习**或**学到的策略没有被使用**。

## 🎯 核心问题：Episode vs Training Step 混淆

### 关键概念澄清

**Episode（任务轮次）**:
- 一次完整的搜索任务执行
- 包含 budget+1 = 51 个时间步
- 4个智能体并行行动
- 产生 51 × 4 = 204 个训练样本

**Training Step（训练步数）**:
- 一次网络参数更新
- 需要 batch_size × batch_number = 128 × 4 = 512 个样本
- 需要约 512/204 ≈ 2.5 个episodes才能进行一次训练

**换算关系**:
```
1 training step ≈ 2.5 episodes
100 training steps ≈ 250 episodes
300 training steps ≈ 750 episodes  ← 你现在的位置
```

### 探索率计算基于Episode，不是Training Step！

代码中的探索率计算（`actor/network.py`）:
```python
def get_action_index(self, ..., num_episode: int, ...):
    if num_episode > self.eps_anneal_phase:
        eps = self.eps_min
    else:
        eps = self.eps_max - num_episode / self.eps_anneal_phase * (
            self.eps_max - self.eps_min
        )
```

这里的`num_episode`是**episode编号**，不是training step！

## 根本原因分析

### 之前的配置（已修复一次但仍有问题）

```yaml
eps_max: 0.6
eps_min: 0.05
eps_anneal_phase: 600  # ← 这是基于episodes的！
```

在300个training steps（≈750 episodes）时：
```
eps = 0.6 - (750/600) × (0.6-0.05) = 0.6 - 1.25 × 0.55 = -0.09
```

由于超过了`eps_anneal_phase`，探索率被设为`eps_min = 0.05`。

**但问题是**: 在前600个episodes（≈240 training steps）之前，探索率都很高！
- 在第100个training step（≈250 episodes）时: eps ≈ 0.37
- 在第200个training step（≈500 episodes）时: eps ≈ 0.14  
- 在第240个training step（≈600 episodes）时: eps = 0.05

**所以前200-240个training steps，模型仍在大量随机探索！**

## ✅ 最终修复方案

```yaml
experiment:
  missions:
    eps_max: 0.3              # 降低初始探索率
    eps_min: 0.05             # 标准最小探索率
    eps_anneal_phase: 100     # 在前100个EPISODES内完成探索→利用转变
                              # 这相当于前40个training steps
```

### 新配置的探索率曲线

| Episodes | Training Steps | 探索率 | 说明 |
|---------|---------------|--------|------|
| 0-25 | 0-10 | 0.30→0.23 | 初期探索 |
| 25-50 | 10-20 | 0.23→0.18 | 快速降低 |
| 50-75 | 20-30 | 0.18→0.11 | 继续降低 |
| 75-100 | 30-40 | 0.11→0.05 | 接近最小值 |
| 100+ | 40+ | 0.05 | 95%利用策略 |
| **750** | **300** | **0.05** | **应该看到学习效果** |

### 为什么这样设置

1. **eps_max: 0.3** - 初期就让策略有70%的主导权
2. **eps_anneal_phase: 100** - 在40个training steps内完成探索→利用转变
3. **结果**: 从第40步开始，模型就以95%的策略+5%探索运行

## 验证方法

### 1. 运行诊断脚本
```powershell
.\running\diagnose_epsilon.ps1 -CurrentTrainingStep 300 -CurrentEpisode 750
```

### 2. 检查训练历史
```powershell
Import-Csv marl_framework\res\training_history.csv | Select-Object -First 10
Import-Csv marl_framework\res\training_history.csv | Select-Object -Last 10
```

应该看到episode_return有明显上升。

### 3. 对比轨迹
重新开始训练后，在不同步数保存轨迹：
- **20 steps（50 episodes）**: 应该有30%→23%探索率，开始有规律
- **40 steps（100 episodes）**: 5%探索率，明确的搜索策略
- **100+ steps**: 完全基于学习的策略

## 如果还是没效果

### 可能的其他问题

#### 1. 学习率过低
```yaml
actor:
  learning_rate: 0.0001  # 尝试提高到 0.0003
critic:
  learning_rate: 0.0005  # 尝试提高到 0.001
```

#### 2. 批次设置导致更新太慢
```yaml
batch_size: 128       # 尝试减小到 64
batch_number: 4       # 尝试增加到 8
data_passes: 5        # 尝试减少到 3
```

#### 3. 奖励信号问题
检查TensorBoard中的奖励值：
- 如果奖励都接近0，说明奖励信号太弱
- 如果奖励方差很大，说明奖励不稳定

#### 4. 网络没有真正更新
检查梯度：
```powershell
tensorboard --logdir=marl_framework\log
```
查看 Actor/gradient_norm 和 Critic/gradient_norm，应该有明显变化。

## 立即行动计划

1. ✅ **配置已修复**:
   ```yaml
   eps_max: 0.3
   eps_min: 0.05
   eps_anneal_phase: 100
   ```

2. **清理并重新训练**:
   ```powershell
   Remove-Item marl_framework\log\* -Recurse -Force -ErrorAction SilentlyContinue
   Remove-Item marl_framework\res\* -Recurse -Force -ErrorAction SilentlyContinue
   cd marl_framework
   python main.py --config configs/params.yaml
   ```

3. **监控训练**:
   ```powershell
   # 在另一个终端
   .\running\diagnose_epsilon.ps1
   .\running\monitor_training.ps1
   ```

4. **保存不同阶段的轨迹图**，应该看到明显变化：
   - 20 steps: 开始出现规律
   - 40 steps: 明确的区域搜索
   - 100 steps: 优化的搜索策略

## 预期结果

修复后，在300 training steps（750 episodes）时：
- ✅ 探索率应该是 **5%**（而不是之前的30%+）
- ✅ 轨迹应该展现明确的**区域优先搜索策略**
- ✅ 高优先级区域应该有**更密集的覆盖**
- ✅ 智能体之间应该有**协同分工**
- ✅ Episode回报应该**逐渐上升**

---

**总结**: 
1. 核心问题是 `eps_anneal_phase` 的值与 episode-training step 的换算关系
2. 必须理解探索率基于**episode编号**，不是training step
3. 修复后的配置让模型在前40个training steps就转向利用学习策略
4. 如果还没效果，检查学习率和奖励信号

```

应该看到eps从0.7逐渐降到0.05。

### 2. 检查网络更新
在TensorBoard中查看:
- Actor/loss: 应该逐渐下降
- Actor/gradient_norm: 应该有明显变化
- Episode_return: 应该逐渐上升

### 3. 对比轨迹
- 前100步: 应该看到混乱的探索
- 200-400步: 开始出现有规律的搜索模式
- 500+步: 应该看到明显的区域搜索策略

## 其他潜在问题

### 学习率
当前设置可能偏低:
```yaml
actor:
  learning_rate: 0.0001  # 可以尝试 0.0003
critic:
  learning_rate: 0.0005  # 可以尝试 0.001
```

### 批次设置
```yaml
batch_size: 128       # 较大
batch_number: 4       # 较少
data_passes: 5        # 每个batch重复5次
```

这个设置较为保守，可以考虑:
```yaml
batch_size: 64        # 减小以增加更新频率
batch_number: 8       # 增加以收集更多样化的数据
data_passes: 3        # 减少以加快训练速度
```

## 立即行动计划

1. **停止当前训练**
2. **修改 params.yaml**:
   ```yaml
   eps_max: 0.6
   eps_min: 0.05
   eps_anneal_phase: 600
   ```
3. **清理旧模型**:
   ```powershell
   Remove-Item marl_framework/log/* -Recurse -Force
   Remove-Item marl_framework/res/* -Recurse -Force
   ```
4. **重新开始训练**
5. **在100、300、500步时检查轨迹变化**

## 预期结果

修复后应该看到:
- **50步**: 随机探索，轨迹混乱
- **200步**: 开始出现有目的性的移动
- **400步**: 明显的区域搜索模式
- **550步**: 高效的优先级区域覆盖策略

---

**总结**: 主要问题是 `eps_anneal_phase: 50000` 这个设置完全错误，导致模型在整个训练过程中都在随机探索，学习到的策略根本没有被使用。修复这个参数后，训练应该能正常进行。
