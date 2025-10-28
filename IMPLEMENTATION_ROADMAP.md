# 高级静态搜索框架实现路线图

## 📋 创新点实施优先级

### 阶段1: 内在奖励机制 (解决稀疏奖励) - 高优先级 ⭐⭐⭐

#### 1.1 基于覆盖率的驱动 (Coverage-driven)
**状态**: ✅ 已有基础 (search_region_manager追踪覆盖)
**需要添加**:
- [ ] 覆盖衰减机制 (模拟"遗忘",鼓励重访)
- [ ] 覆盖率梯度奖励 (访问低覆盖区域奖励更高)

**实现位置**: 
- `utils/reward.py` - 添加 `get_coverage_exploration_reward()`
- `mapping/search_regions.py` - 添加覆盖衰减更新

#### 1.2 前沿探测驱动 (Frontier-based) ⭐⭐⭐
**状态**: ❌ 需要新实现
**核心思想**: 奖励智能体探索已探索/未探索区域的边界

**需要实现**:
- [ ] 前沿检测算法 (识别覆盖图边界)
- [ ] 前沿图生成 (frontier_map)
- [ ] 前沿奖励计算 (基于到最近前沿的距离)

**实现位置**:
- `mapping/frontier_detection.py` (新文件)
- `utils/reward.py` - 添加 `get_frontier_reward()`
- `actor/transformations.py` - 添加 `frontier_map` 到观察

#### 1.3 好奇心驱动 (Curiosity-driven)
**状态**: ❌ 需要新实现 (可选,复杂度高)
**核心思想**: 使用预测误差作为内在奖励

**需要实现**:
- [ ] 前向预测模型 (预测下一步观测)
- [ ] 预测误差计算
- [ ] 预测误差图维护

**实现位置**:
- `models/curiosity_module.py` (新文件)
- 可先跳过,聚焦前沿驱动

---

### 阶段2: 协同机制改进 - 高优先级 ⭐⭐⭐

#### 2.1 抗重叠惩罚 (Anti-overlap Penalty)
**状态**: ✅ 部分实现 (redundant_search_penalty)
**需要增强**:
- [ ] 路径重叠度计算 (不仅是观测重叠)
- [ ] 实时路径预测 (预测未来N步路径)
- [ ] 动态重叠惩罚 (基于重叠程度)

**实现位置**:
- `utils/coordination.py` (新文件)
- `utils/reward.py` - 增强 `redundant_search` 计算

#### 2.2 区域分工奖励 (Division-of-labor)
**状态**: ❌ 需要新实现
**核心思想**: 鼓励智能体分散到不同区域

**需要实现**:
- [ ] 区域占用检测 (哪些智能体在哪些区域)
- [ ] 分工度量 (衡量智能体分布均匀性)
- [ ] 分工奖励计算

**实现位置**:
- `mapping/search_regions.py` - 添加 `get_region_occupancy()`
- `utils/reward.py` - 添加 `get_division_reward()`

#### 2.3 协同发现奖励 (Collaborative Discovery)
**状态**: ❌ 需要新实现
**核心思想**: 奖励多机协同搜索高优先级区域

**实现位置**:
- `utils/reward.py` - 添加 `get_collaboration_reward()`

---

### 阶段3: 搜索专用状态表示 - 中优先级 ⭐⭐

#### 3.1 搜索置信度图 (Search Confidence Map)
**状态**: ❌ 需要新实现
**核心思想**: 维护每个网格的"被彻底搜索"的置信度

**需要实现**:
- [ ] 置信度更新规则 (基于传感器模型)
- [ ] 置信度图维护
- [ ] 将置信度图加入状态

**实现位置**:
- `mapping/confidence_map.py` (新文件)
- `actor/transformations.py` - 添加到观察

#### 3.2 联合搜索前景图 (Joint Search Prospect Map)
**状态**: ❌ 需要新实现
**核心思想**: 融合所有智能体的观测数据

**实现位置**:
- `mapping/search_regions.py` - 添加 `get_joint_prospect_map()`

#### 3.3 前沿图 (Frontier Map)
**状态**: ❌ 需要新实现 (与1.2关联)
**实现**: 见阶段1.2

---

### 阶段4: 评估指标与实验框架 - 高优先级 ⭐⭐⭐

#### 4.1 搜索核心指标
**需要实现**:
- [ ] 首次发现时间追踪
- [ ] 平均发现时间计算
- [ ] 任务完成时间记录
- [ ] 目标发现率统计

**实现位置**:
- `missions/coma_mission.py` - 扩展指标记录
- `utils/metrics.py` (新文件)

#### 4.2 搜索效率指标
**需要实现**:
- [ ] 覆盖率时间曲线
- [ ] 路径重复度计算
- [ ] 重叠度量化

**实现位置**:
- `utils/metrics.py`

#### 4.3 协同效能指标
**需要实现**:
- [ ] 协同效率 = (单机总时间) / (多机实际时间)
- [ ] 负载均衡度
- [ ] 通信开销统计

**实现位置**:
- `utils/metrics.py`

#### 4.4 消融实验框架
**需要实现**:
- [ ] 配置管理器 (批量运行不同配置)
- [ ] 结果对比分析工具
- [ ] 自动化实验脚本

**实现位置**:
- `scripts/ablation_study.py` (新文件)
- `scripts/benchmark_runner.py` (新文件)

---

## 🎯 实施建议

### 推荐实施顺序:

1. **第一周**: 实现前沿探测驱动 (1.2)
   - 核心创新点,影响最大
   - 实现难度中等
   - 立即可见效果

2. **第二周**: 实现协同机制 (2.1, 2.2)
   - 抗重叠惩罚增强
   - 区域分工奖励
   - 提升多机协同效果

3. **第三周**: 实现评估指标 (4.1, 4.2, 4.3)
   - 搭建完整评估体系
   - 为论文提供数据支撑

4. **第四周**: 消融实验与基准测试 (4.4)
   - 系统性验证各组件贡献
   - 生成论文实验数据

### MVP (最小可行产品):
如果时间紧张,优先实现:
1. ✅ 前沿探测驱动 (1.2)
2. ✅ 抗重叠惩罚增强 (2.1)
3. ✅ 核心搜索指标 (4.1)
4. ✅ 简单消融实验 (4.4)

这4个功能就足以构成一篇高质量论文。

---

## 📝 实现细节参考

### 前沿检测算法示例:
```python
def detect_frontiers(coverage_map, threshold=0.3):
    """
    检测覆盖图中的前沿(已探索/未探索边界)
    
    Args:
        coverage_map: 覆盖图 (0-1之间的值)
        threshold: 判断已探索的阈值
    
    Returns:
        frontier_map: 前沿图 (边界位置为1)
    """
    explored = (coverage_map > threshold).astype(float)
    unexplored = (coverage_map <= threshold).astype(float)
    
    # 使用形态学操作找边界
    from scipy.ndimage import binary_dilation
    
    explored_dilated = binary_dilation(explored)
    frontier = explored_dilated & unexplored
    
    return frontier.astype(float)
```

### 前沿奖励计算示例:
```python
def get_frontier_reward(position, frontier_map, spacing):
    """
    计算智能体到最近前沿的奖励
    
    奖励 = exp(-distance_to_frontier / decay_constant)
    """
    # 找到最近的前沿点
    frontier_positions = np.argwhere(frontier_map > 0.5)
    
    if len(frontier_positions) == 0:
        return 0.0
    
    # 计算距离
    pos_idx = position_to_index(position, spacing)
    distances = np.linalg.norm(frontier_positions - pos_idx, axis=1)
    min_distance = np.min(distances)
    
    # 指数衰减奖励
    reward = np.exp(-min_distance / 5.0)
    
    return reward
```

### 区域分工度量示例:
```python
def calculate_division_score(agent_positions, regions):
    """
    计算区域分工得分
    
    完美分工 (每个智能体在不同区域) = 1.0
    完全重叠 (所有智能体在同一区域) = 0.0
    """
    # 统计每个区域的智能体数量
    region_counts = {}
    for pos in agent_positions:
        region = get_region_at_position(pos)
        if region:
            region_counts[region.name] = region_counts.get(region.name, 0) + 1
    
    # 计算分布的均匀性 (使用基尼系数的逆)
    if not region_counts:
        return 0.0
    
    counts = list(region_counts.values())
    # 理想情况: 每个区域1个智能体
    # 实际情况: counts的分布
    
    # 使用标准差度量不均匀性
    std = np.std(counts)
    max_std = np.sqrt(len(agent_positions)**2 / len(regions))
    
    division_score = 1.0 - (std / max_std)
    
    return division_score
```

---

## 🔬 实验设计建议

### 消融实验设计:

**实验1: 内在奖励消融**
```
Baseline: 无内在奖励
+ Coverage: 基线 + 覆盖驱动
+ Frontier: 基线 + 前沿驱动  
+ Both: 基线 + 覆盖 + 前沿
```

**实验2: 协同机制消融**
```
Baseline: 无协同机制
+ Overlap: 基线 + 抗重叠惩罚
+ Division: 基线 + 区域分工
+ Full: 所有协同机制
```

**实验3: 通信范围分析**
```
通信范围: [0m, 10m, 15m, 25m, ∞]
对比指标: 完成时间, 覆盖率, 协同效率
```

### 基准场景设计:

**场景集1: 地图大小**
- 小地图: 30x30 (快速验证)
- 中地图: 50x50 (标准场景)
- 大地图: 70x70 (扩展性测试)

**场景集2: 目标分布**
- 均匀分布
- 聚类分布 (2-3个热点)
- 随机分布

**场景集3: 先验信息**
- 无先验 (均匀概率)
- 部分先验 (区域概率)
- 完整先验 (精确概率图)

---

## 📊 预期论文贡献

基于这套框架,你的论文可以有以下贡献:

1. **理论贡献**:
   - 提出基于前沿探测的内在奖励机制
   - 设计区域分工驱动的协同策略
   - 证明在静态搜索中优于传统方法

2. **实验贡献**:
   - 完整的消融实验验证各组件贡献
   - 系统性的基准测试 (多场景、多条件)
   - 通信约束下的鲁棒性分析

3. **工程贡献**:
   - 开源的多智能体搜索框架
   - 标准化的评估指标体系
   - 可复现的实验基准

---

## 🚀 快速开始

如果你想立即开始,我建议:

1. **先实现前沿探测驱动** (最核心的创新)
2. **运行初步实验** (与基线对比)
3. **根据结果调整** (决定是否添加其他功能)

我可以帮你:
- 实现前沿检测算法
- 集成到现有reward系统
- 创建实验脚本
- 设计对比实验

你想从哪个开始? 😊
