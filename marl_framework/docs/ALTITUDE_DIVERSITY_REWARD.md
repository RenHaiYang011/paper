# 高度多样性奖励机制 (Altitude Diversity Reward)

## 📋 功能概述

为了鼓励UAV在不同高度进行探索,新增了**高度多样性奖励机制**。该机制通过奖励agents在垂直方向的探索行为,使其能够:
1. 利用不同高度的传感器特性(高度越高,覆盖范围越大但噪声越高)
2. 增加3D空间的探索效率
3. 避免所有agents停留在相同高度

## 🎯 设计原理

### 奖励组成

高度多样性奖励由两部分组成:

```python
altitude_bonus = altitude_diversity_weight * (altitude_variance * 0.01 + mean_altitude_change * 0.1)
```

1. **空间多样性 (Altitude Variance)**
   - 计算所有agents当前高度的方差
   - 鼓励agents分散在不同高度层
   - 系数: 0.01

2. **时间变化性 (Mean Altitude Change)**
   - 计算每个agent相邻时刻的高度变化
   - 鼓励agents进行垂直移动
   - 系数: 0.1

### 数学表达

$$
\text{Altitude Bonus} = w_{alt} \times \left( 0.01 \times \text{Var}(h_1, h_2, ..., h_n) + 0.1 \times \frac{1}{n}\sum_{i=1}^{n}|h_i^{t+1} - h_i^{t}| \right)
$$

其中:
- $w_{alt}$: `altitude_diversity_weight` (可配置)
- $h_i^t$: agent $i$ 在时刻 $t$ 的高度
- $n$: agents数量

## ⚙️ 配置方法

### 在配置文件中设置

```yaml
experiment:
  altitude_diversity_weight: 0.5  # 推荐范围: 0.0 - 1.0
```

### 权重建议

| 权重值 | 效果 | 适用场景 |
|--------|------|----------|
| 0.0    | 不启用 | 固定高度任务 |
| 0.3    | 轻微鼓励 | 主要关注水平覆盖,略微鼓励高度变化 |
| **0.5** | **平衡** | **推荐默认值,平衡3D探索和任务目标** |
| 0.8    | 强烈鼓励 | 需要充分利用不同高度的传感器特性 |
| 1.0+   | 极度强调 | 可能过度追求高度变化而忽略主要任务 |

## 📊 效果监控

### TensorBoard可视化

训练过程中会自动记录:

```
Bonuses/Altitude_Diversity
```

可以通过以下命令查看:
```bash
tensorboard --logdir=log/
```

### 预期效果

启用该奖励后,应该观察到:
1. ✅ 航线在不同高度层之间变化
2. ✅ agents不会全部聚集在同一高度
3. ✅ 平均高度标准差 > 2.0 (spacing=5的情况)
4. ✅ 每个episode有明显的高度变化次数

## 🔧 实现细节

### 代码位置

- **奖励计算**: `marl_framework/utils/reward.py` - `get_global_reward()`
- **参数传递**: `marl_framework/coma_wrapper.py` - `COMAWrapper.__init__()`
- **配置文件**: `marl_framework/configs/params.yaml` 和 `params_fast.yaml`

### 计算流程

```python
# 1. 提取所有agents的当前高度
altitudes = [pos[2] for pos in next_positions]

# 2. 计算空间方差
altitude_variance = np.var(altitudes)

# 3. 计算时间变化
altitude_changes = [abs(next_pos[2] - prev_pos[2]) for prev_pos, next_pos in zip(prev_positions, next_positions)]
mean_altitude_change = np.mean(altitude_changes)

# 4. 组合奖励
altitude_bonus = altitude_diversity_weight * (altitude_variance * 0.01 + mean_altitude_change * 0.1)

# 5. 添加到总奖励
absolute_reward += altitude_bonus
```

### 异常处理

- 如果 `next_positions` 为 `None`,奖励自动为0
- 如果位置数据格式错误,使用 try-except 保护,不影响其他奖励计算
- 单个agent情况下只计算时间变化,不计算空间方差

## 🧪 调试和验证

### 快速测试

在训练脚本中添加日志:

```python
# 在 coma_wrapper.py 中
if altitude_bonus > 0:
    print(f"Step {t}: Altitude bonus = {altitude_bonus:.4f}")
    print(f"  Altitudes: {[pos[2] for pos in next_positions]}")
    print(f"  Variance: {altitude_variance:.2f}")
    print(f"  Mean change: {mean_altitude_change:.2f}")
```

### 验证方法

1. **检查配置读取**:
   ```python
   print(f"Altitude diversity weight: {self.altitude_diversity_weight}")
   ```

2. **检查奖励数值**:
   - 查看 TensorBoard: `Bonuses/Altitude_Diversity`
   - 应该大于0且随训练变化

3. **检查航线输出**:
   ```python
   import numpy as np
   altitudes = episode_data['agent_altitudes']
   print(f"Altitude range: {np.min(altitudes)} - {np.max(altitudes)}")
   print(f"Altitude std: {np.std(altitudes)}")
   ```

## 📈 与其他奖励的平衡

### 奖励权重对比

```yaml
# 当前推荐配置
coverage_weight: 0.15           # 主要任务目标
footprint_weight: 0.5           # 避免重叠
collision_weight: 2.0           # 安全约束
altitude_diversity_weight: 0.5  # 3D探索鼓励
distance_weight: 0.0            # 关闭(允许自由探索)
```

### 权重调优建议

- 如果高度变化**太少**: 增加 `altitude_diversity_weight` 到 0.8-1.0
- 如果高度变化**太频繁**: 减小到 0.2-0.3
- 如果**忽略主要任务**: 增加 `coverage_weight`,减小 `altitude_diversity_weight`

## 🚀 使用示例

### 启用高度多样性奖励

```yaml
# configs/params_altitude_test.yaml
experiment:
  altitude_diversity_weight: 0.5
  constraints:
    min_altitude: 5
    max_altitude: 15
    spacing: 5
    num_actions: 6  # 确保包含上升/下降动作
```

### 训练命令

```bash
export CONFIG_FILE_PATH=configs/params_altitude_test.yaml
python main.py
```

### 对比实验

```bash
# 基线实验(无高度奖励)
export CONFIG_FILE_PATH=configs/params_baseline.yaml  # altitude_diversity_weight: 0.0
python main.py

# 高度奖励实验
export CONFIG_FILE_PATH=configs/params_altitude.yaml  # altitude_diversity_weight: 0.5
python main.py
```

## 📝 注意事项

1. **动作空间要求**:
   - `num_actions = 6`: 包含上升/下降动作(推荐)
   - `num_actions = 27`: 完整3D动作空间(最佳)
   - `num_actions = 4` 或 `9`: 仅2D平面,无高度变化(不适用)

2. **高度范围设置**:
   - `min_altitude` 和 `max_altitude` 应该有足够的差值
   - 推荐至少3个高度层级 (如 5, 10, 15)

3. **传感器模型配合**:
   - 确保 `sensor.model.type = "altitude_dependent"`
   - 不同高度应该有明显的传感器性能差异

4. **训练时间**:
   - 学习3D策略可能需要更长的训练时间
   - 建议增加 `n_episodes` 或减小 `eps_anneal_phase`

## 🔬 实验结果参考

### 预期改进

- ✅ 高度标准差: 从 ~0.5 提升到 2-3
- ✅ 高度变化频率: 从 <5% 提升到 15-25%
- ✅ 传感器利用效率: 更好的高度-噪声权衡
- ✅ 覆盖效率: 可能略有提升(由于更灵活的3D路径)

### 可能的副作用

- ⚠️ 训练初期可能出现过度的上下移动
- ⚠️ 如果权重过大,可能忽略水平覆盖任务
- ⚠️ 需要更多训练时间收敛到最优策略

## 📚 相关文档

- [GPU_BOTTLENECK_ANALYSIS.md](GPU_BOTTLENECK_ANALYSIS.md) - 训练性能分析
- [CONFIG_SELECTION_GUIDE.md](CONFIG_SELECTION_GUIDE.md) - 配置文件选择指南
- `agent/action_space.py` - 动作空间定义
- `sensors/models/sensor_models.py` - 高度相关的传感器模型

---

**版本**: 1.0  
**创建日期**: 2025-10-28  
**最后更新**: 2025-10-28
