# 🔍 训练问题诊断报告

## 问题描述
训练550步后，航线轨迹与50步时完全相同，表明模型**没有学习**。

## 根本原因分析

### 1. **探索率衰减设置错误** ❌ (最严重)
```yaml
eps_max: 0.8
eps_min: 0.15
eps_anneal_phase: 50000  # ← 这是关键问题！
```

**问题**:
- `eps_anneal_phase: 50000` 表示需要50000个**episode**才能衰减
- 但你的配置是 `n_episodes: 1200`
- 实际上在550个**training steps**时，可能只运行了几十个episodes
- 探索率计算: `eps = 0.8 - (episode / 50000) * (0.8 - 0.15)`
- 在前1200个episodes内，探索率几乎不变，始终接近0.8

**结果**:
- 80%的时间都在随机探索
- 只有20%的时间按照学习的策略行动
- 模型学习到的策略根本没有机会被使用
- **这就是为什么550步和50步轨迹完全一样！**

### 2. 探索率过高
```yaml
eps_max: 0.8   # 初始80%随机
eps_min: 0.15  # 最终15%随机
```

即使探索率正常衰减，15%的最小探索率也偏高。标准设置应该是5%左右。

### 3. Episode vs Training Step 混淆
- **Episode**: 完整的任务执行过程（50个时间步）
- **Training Step**: 每次网络参数更新
- 你的配置混淆了这两个概念

训练步数计算:
```
training_steps ≈ n_episodes × budget × n_agents / (batch_size × batch_number)
training_steps ≈ 1200 × 50 × 4 / (128 × 4) ≈ 469步
```

所以550步已经接近训练结束，但探索率才从0.8降到约0.79！

## 解决方案

### 🎯 方案1: 快速修复（立即可用）

修改 `params.yaml` 中的探索率设置:

```yaml
experiment:
  missions:
    eps_max: 0.7              # 从0.8降到0.7
    eps_min: 0.05             # 从0.15降到0.05
    eps_anneal_phase: 800     # 从50000改为800 (约覆盖2/3训练)
```

**理由**:
- `eps_anneal_phase: 800` 基于预估的训练步数469×1.7≈800
- 在前2/3的训练中完成探索→利用的转变
- 后1/3的训练用学习到的策略进行精调

### 🎯 方案2: 标准配置（推荐）

```yaml
experiment:
  missions:
    n_episodes: 1200
    eps_max: 0.5              # 更保守的初始探索
    eps_min: 0.05             # 标准最小探索
    eps_anneal_phase: 600     # 更激进的衰减
```

### 🎯 方案3: 基于episode的探索衰减（最准确）

修改代码使用episode而不是training_step来计算探索率。

当前代码 (`actor/network.py` line 106-110):
```python
if num_episode > self.eps_anneal_phase:
    eps = self.eps_min
else:
    eps = self.eps_max - num_episode / self.eps_anneal_phase * (
        self.eps_max - self.eps_min
    )
```

这里的`num_episode`应该是真实的episode编号，然后设置:
```yaml
eps_anneal_phase: 800  # 在800个episodes内完成探索衰减
```

## 验证方法

### 1. 检查探索率
在训练日志中查找eps值:
```
grep "eps" training.log | head -20
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
