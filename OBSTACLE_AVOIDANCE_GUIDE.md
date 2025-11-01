# 🚁 障碍物避障功能使用指南

## 📋 功能概述

本项目已集成完整的3D障碍物避障功能，包括：
- ✅ **障碍物管理器** - 管理环境中的障碍物
- ✅ **动作掩码** - 自动屏蔽会导致碰撞的动作
- ✅ **奖励惩罚** - 对接近/碰撞障碍物进行惩罚
- ✅ **可视化** - 3D图中显示黄色实心金字塔障碍物

---

## 🚀 快速开始

### 1. 启用障碍物避障

在配置文件中启用障碍物：

```yaml
experiment:
  obstacles:
    enable: true              # 启用障碍物避障
    safety_margin: 2.0        # 安全边界（米）
    collision_penalty: 50.0   # 碰撞惩罚值
```

### 2. 定义障碍物

#### 方式A：使用固定障碍物

```yaml
experiment:
  obstacles:
    enable: true
    fixed_obstacles:
      - x: 12.0      # 障碍物中心X坐标（米）
        y: 10.0      # 障碍物中心Y坐标（米）
        z: 0.0       # 障碍物底部高度（通常为0）
        height: 15.0 # 障碍物高度（米）
        radius: 2.75 # 障碍物半径（米）
      - x: 26.0
        y: 20.0
        z: 0.0
        height: 12.0
        radius: 3.0
```

#### 方式B：自动生成随机障碍物

如果不指定 `fixed_obstacles`，系统会根据环境大小自动生成合适数量的障碍物。

### 3. 配置奖励权重

```yaml
reward:
  obstacle_penalty_weight: 1.0  # 障碍物惩罚权重
  collision_weight: 10.0         # 智能体间碰撞权重
```

---

## 🔧 集成到现有代码

### 在 Mission 中集成

```python
from marl_framework.utils.obstacle_manager import ObstacleManager

class YourMission:
    def __init__(self, params):
        # 创建障碍物管理器
        self.obstacle_manager = ObstacleManager(params)
        
        # 设置障碍物
        obstacles = self._load_or_generate_obstacles()
        self.obstacle_manager.set_obstacles(obstacles)
        
        # 创建动作空间（传入障碍物管理器）
        self.action_space = AgentActionSpace(
            params, 
            obstacle_manager=self.obstacle_manager
        )
```

### 在奖励计算中使用

```python
from marl_framework.utils.reward import get_global_reward

reward = get_global_reward(
    # ... 其他参数 ...
    obstacle_manager=self.obstacle_manager,
    obstacle_penalty_weight=1.0,
    # ... 其他参数 ...
)
```

---

## 📊 避障工作原理

### 1. 动作掩码机制

```python
# Agent 每步会自动调用避障掩码
action_mask = action_space.get_action_mask(position)
action_mask = action_space.apply_collision_mask(...)  # 避免智能体间碰撞
action_mask = action_space.apply_obstacle_mask(...)   # 避免障碍物碰撞 ✨
```

### 2. 碰撞检测

- **位置检测**: 检查某个位置是否在障碍物内
- **路径检测**: 检查从A点到B点的路径是否穿过障碍物
- **安全边界**: 在障碍物周围设置安全缓冲区

### 3. 奖励惩罚

```python
# 完全碰撞 -> 最大惩罚
if in_obstacle:
    penalty = collision_penalty  # 例如: -50

# 进入安全边界 -> 渐变惩罚
elif in_safety_margin:
    penalty = collision_penalty * (1 - distance/safety_margin) * 0.5
```

---

## 🧪 测试避障功能

运行测试脚本：

```bash
python test_obstacle_avoidance.py
```

测试脚本会：
1. ✅ 初始化障碍物管理器
2. ✅ 测试碰撞检测
3. ✅ 生成避障轨迹
4. ✅ 生成3D可视化图

---

## 🎨 可视化说明

障碍物在3D图中显示为：
- **形状**: 黄色实心金字塔（从 z=0.5 开始）
- **边缘**: 橙色边框线
- **位置**: 对应实际的 (x, y) 世界坐标
- **高度**: 从地面向上延伸指定高度

---

## ⚙️ 参数调优建议

### 安全边界 (safety_margin)

```yaml
safety_margin: 2.0  # 推荐: 2-3米
```
- **过小**: 智能体可能撞上障碍物
- **过大**: 限制过多，影响探索效率

### 碰撞惩罚 (collision_penalty)

```yaml
collision_penalty: 50.0  # 推荐: 30-100
```
- **过小**: 智能体不重视避障
- **过大**: 智能体过度保守，不敢靠近

### 惩罚权重 (obstacle_penalty_weight)

```yaml
obstacle_penalty_weight: 1.0  # 推荐: 0.5-2.0
```
- 调整此参数平衡"探索效率"和"安全性"

---

## 📝 使用示例

### 简单场景（3个障碍物）

```python
obstacles = [
    {'x': 10, 'y': 10, 'z': 0, 'height': 12, 'radius': 2.5},
    {'x': 25, 'y': 20, 'z': 0, 'height': 15, 'radius': 3.0},
    {'x': 40, 'y': 35, 'z': 0, 'height': 10, 'radius': 2.0}
]
```

### 复杂场景（密集障碍物）

```python
# 自动生成5-10个随机障碍物
obstacle_manager = ObstacleManager(params)
# 不调用 set_obstacles()，让系统自动生成
```

---

## 🐛 故障排查

### 问题1: 所有动作都被屏蔽

**原因**: 智能体被困在障碍物包围中  
**解决**: 
- 减小 `safety_margin`
- 调整障碍物分布，确保有通道
- 检查障碍物 `radius` 是否过大

### 问题2: 智能体仍然撞上障碍物

**原因**: 
- 动作掩码未正确应用
- 障碍物数据格式错误

**解决**:
```python
# 确保在 Agent.step() 中调用
action_mask = action_space.apply_obstacle_mask(...)
```

### 问题3: 训练不收敛

**原因**: 惩罚过重，限制探索  
**解决**: 
- 降低 `collision_penalty`
- 降低 `obstacle_penalty_weight`
- 增加 exploration epsilon

---

## 📚 相关文件

- `marl_framework/utils/obstacle_manager.py` - 障碍物管理器
- `marl_framework/agent/action_space.py` - 动作空间（包含避障掩码）
- `marl_framework/utils/reward.py` - 奖励函数（包含障碍物惩罚）
- `marl_framework/utils/plotting.py` - 3D可视化
- `test_obstacle_avoidance.py` - 完整测试脚本
- `configs/params_obstacle_avoidance.yaml` - 示例配置

---

## ✨ 特性总结

| 特性 | 状态 | 说明 |
|------|------|------|
| 3D障碍物检测 | ✅ | 支持圆柱形障碍物 |
| 动作掩码 | ✅ | 自动屏蔽碰撞动作 |
| 路径检测 | ✅ | 检测路径是否穿过障碍物 |
| 安全边界 | ✅ | 可配置的安全缓冲区 |
| 渐变惩罚 | ✅ | 根据距离计算惩罚 |
| 3D可视化 | ✅ | 黄色实心金字塔 |
| 配置灵活 | ✅ | 支持固定/随机障碍物 |

---

## 🎯 下一步

1. 运行 `python test_obstacle_avoidance.py` 验证功能
2. 根据场景调整障碍物参数
3. 在训练中观察避障行为
4. 查看生成的3D可视化图

🎉 **现在你已经拥有完整的障碍物避障功能！**
