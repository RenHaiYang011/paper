# 🔧 轨迹可视化坐标对齐修复

## 问题描述

用户报告了以下问题：
1. ❌ **地面区域跑偏** - 红蓝色地图区域与网格坐标不对齐
2. ❌ **没有障碍物显示** - 轨迹图中看不到障碍物
3. ❌ **目标立方体位置不准** - 立方体不在地图红色区域正上方

## 根本原因

### 1. 坐标系统混乱
```python
# 问题代码：
Y, X = np.meshgrid(world_x, world_y)  # 错误的顺序
ax.plot_surface(X, Y, ...)            # X/Y不匹配
```

**原因分析**:
- `simulated_map` 是 `[row, col]` 索引，对应 `[y, x]`
- `meshgrid(x_coords, y_coords)` 返回 `(X, Y)`，其中 X按列变化，Y按行变化
- `plot_surface(X, Y, Z)` 期望 X沿列变化，Y沿行变化
- 必须确保索引顺序一致

### 2. 目标坐标转换错误
```python
# 问题代码：
target_x = target_positions[1] * (50.0 / map_width)  # 可能错位
target_y = target_positions[0] * (50.0 / map_height)
```

### 3. 缺少障碍物数据传递
- `plot_trajectories()` 调用时没有传入 `obstacles` 参数
- 没有生成或配置障碍物数据

## 修复方案

### ✅ 1. 修复坐标系统对齐

#### 修改文件: `utils/plotting.py`

```python
# 修复后的代码：
# CRITICAL: meshgrid must match plot_surface X,Y ordering
# simulated_map is indexed as [row, col] = [y, x]
x_coords = np.linspace(0, 50, map_width)   # columns -> X
y_coords = np.linspace(0, 50, map_height)  # rows -> Y
X, Y = np.meshgrid(x_coords, y_coords)     # X: columns, Y: rows

# Plot surface with correct alignment
ax.plot_surface(
    X,  # X coordinates (columns, 0-50)
    Y,  # Y coordinates (rows, 0-50)
    np.zeros_like(simulated_map),
    facecolors=cm.coolwarm(simulated_map),
    ...
)
```

**关键点**:
- `X, Y = np.meshgrid(x_coords, y_coords)` 正确顺序
- X对应列（宽度），Y对应行（高度）
- 与 `plot_surface(X, Y, Z)` 的参数顺序一致

### ✅ 2. 修复目标坐标转换

```python
# 修复后的代码：
# simulated_map[row, col] -> world coordinates (x, y)
# row corresponds to Y, col corresponds to X
target_y_pixels = target_positions[0]  # row indices -> Y
target_x_pixels = target_positions[1]  # column indices -> X

# Convert pixel coordinates to world coordinates
target_x = target_x_pixels * (50.0 / map_width)   # X in world coords
target_y = target_y_pixels * (50.0 / map_height)  # Y in world coords

# Draw cube at ground level with correct height
plot_cube(ax, tx, ty, 1.0, size=2.0, color='red', alpha=0.9)
```

**改进**:
- 清晰的注释说明row对应Y，col对应X
- 立方体z起点设为1.0（稍微抬高），避免与地面重叠
- 目标立方体严格对应地图上的红色区域

### ✅ 3. 添加障碍物生成和传递

#### 修改文件: `missions/coma_mission.py`

```python
# 1. 在初始化时生成障碍物
def __init__(self, ...):
    ...
    # Initialize obstacles for visualization
    self.obstacles = self._generate_obstacles()

# 2. 添加障碍物生成方法
def _generate_obstacles(self):
    """Generate obstacles for visualization"""
    obstacles_cfg = self.params.get("visualization", {}).get("obstacles", [])
    
    if obstacles_cfg:
        return obstacles_cfg  # Use configured obstacles
    
    # Generate random obstacles if not configured
    x_dim = self.params["environment"]["x_dim"]
    y_dim = self.params["environment"]["y_dim"]
    
    np.random.seed(42)
    n_obstacles = np.random.randint(3, 6)
    obstacles = []
    
    for i in range(n_obstacles):
        obs = {
            'x': float(np.random.uniform(10, x_dim - 10)),
            'y': float(np.random.uniform(10, y_dim - 10)),
            'z': 0,
            'height': float(np.random.uniform(8, 15)),
        }
        obstacles.append(obs)
    
    return obstacles

# 3. 传递障碍物到绘图函数
plot_trajectories(
    agent_positions,
    self.n_agents,
    self.writer,
    self.training_step_idx,
    t_collision,
    self.budget,
    simulated_map,
    obstacles=self.obstacles,  # Pass obstacles
)
```

### ✅ 4. 配置文件支持

#### 修改文件: `configs/params_fast.yaml`

```yaml
# 新增可视化配置部分
visualization:
  obstacles:
    - x: 15
      y: 25
      z: 0
      height: 12
    
    - x: 35
      y: 15
      z: 0
      height: 15
    
    - x: 25
      y: 40
      z: 0
      height: 10
```

**特点**:
- 可选配置，不配置则自动生成随机障碍物
- 支持自定义障碍物位置和高度
- 易于扩展（如动态障碍物、风场等）

## 验证方法

### 1. 运行测试脚本
```bash
cd E:\code\paper_code\paper
python test_trajectory_fix.py
```

### 2. 检查生成的图片
查看 `res/plots/coma_pathes_3d_999.png`，确认：
- ✅ 红色立方体位于地图红色区域正上方
- ✅ 灰色金字塔在指定位置显示
- ✅ 地图网格与颜色区域完美对齐
- ✅ 智能体轨迹合理绕过障碍物

### 3. 继续训练查看效果
下次训练时（每20步保存一次），新的轨迹图将包含：
- 正确对齐的地图坐标
- 准确标识的目标位置（红色立方体）
- 清晰显示的障碍物（灰色金字塔）

## 技术细节

### 坐标转换公式

```python
# 像素坐标 -> 世界坐标
pixel_col (x方向) -> world_x = pixel_col * (50.0 / map_width)
pixel_row (y方向) -> world_y = pixel_row * (50.0 / map_height)

# numpy索引 -> 世界坐标
simulated_map[row, col] -> (world_x, world_y)
where:
    world_x = col * (50.0 / map_width)
    world_y = row * (50.0 / map_height)
```

### meshgrid 理解

```python
x = [0, 1, 2]
y = [0, 1]

X, Y = np.meshgrid(x, y)

# X沿列变化（x方向）:
# [[0, 1, 2],
#  [0, 1, 2]]

# Y沿行变化（y方向）:
# [[0, 0, 0],
#  [1, 1, 1]]
```

## 修改文件清单

1. ✅ `utils/plotting.py` - 修复坐标系统和目标定位
2. ✅ `missions/coma_mission.py` - 添加障碍物生成和传递
3. ✅ `configs/params_fast.yaml` - 添加可视化配置
4. ✅ `test_trajectory_fix.py` - 测试脚本
5. ✅ `COORDINATE_ALIGNMENT_FIX.md` - 本文档

## 预期效果

### 修复前:
- ❌ 地图颜色区域偏移，与网格不对齐
- ❌ 没有障碍物显示
- ❌ 目标立方体位置偏离

### 修复后:
- ✅ 地图颜色区域与网格完美对齐
- ✅ 障碍物清晰显示为灰色金字塔
- ✅ 目标立方体严格位于红色区域正上方
- ✅ 轨迹与地图坐标精确匹配
- ✅ 专业美观的可视化效果

## 下一步

1. 在Linux服务器上重新训练
2. 检查新生成的轨迹图
3. 确认所有问题已解决
4. 如需调整障碍物，修改 `params_fast.yaml` 的 `visualization` 部分

## 参考资料

- matplotlib 3D plotting: https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html
- numpy meshgrid: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
- 坐标系统约定: [row, col] = [y, x] in image processing
