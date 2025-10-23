# GPU利用率问题诊断和解决方案

## 🔍 问题诊断

### 观察到的现象
```
GPU 0: 3158MB显存占用，但GPU利用率 0%
训练速度: 76秒/步 (预期应该<10秒)
```

### 根本原因
**CPU数据预处理瓶颈** - GPU在等待CPU处理数据

---

## 🎯 问题根源

### 1. 大量CPU numpy操作
在 `actor/transformations.py` 中，每一步都要执行：

```python
# 这些都在CPU上运行
cv2.resize()                    # OpenCV CPU处理
np.dstack()                     # numpy数组拼接
np.ones_like()                  # numpy数组创建
get_w_entropy_map()             # 复杂的numpy计算
```

### 2. 数据传输延迟
```python
# 数据流程
CPU numpy数组 → torch.tensor() → .to(GPU)
            ↑                    ↓
        每步重复               训练
```

每个episode要执行`budget × n_agents`次这样的转换！

---

## 🚀 解决方案

### 方案1: 减少CPU操作（立即可行）

#### 选项A: 降低地图分辨率

当前transformations中的resize可能很慢：

```bash
# 临时修改 - 在params.yaml中
environment:
  x_dim: 50  # 可以尝试降低到30
  y_dim: 50  # 可以尝试降低到30
```

这会减少计算量但可能影响精度。

#### 选项B: 简化特征图

注释掉一些非关键的特征：

```python
# 在 transformations.py 中，可以临时禁用一些特征
# observation_map = torch.tensor(
#     np.dstack([
#         budget_map,
#         # agent_id_map,  # 可能不太重要
#         position_map,
#         w_entropy_map,
#         # local_w_entropy_map,  # 重复信息
#         prob_map,
#         # footprint_map,  # 计算密集
#     ])
# )
```

---

### 方案2: 优化配置（推荐，立即尝试）

当前训练步数过多是主要问题。优化配置：

```yaml
# params.yaml
networks:
  data_passes: 2      # 从3减到2
  batch_size: 48      # 从64减到48  
  batch_number: 2     # 从3减到2

experiment:
  missions:
    n_episodes: 800   # 从1500减到800
  constraints:
    budget: 10        # 从14减到10（关键！）
```

**效果**:
- 总步数: ~2,667步（从4,800减少45%)
- 预计时间: 5-15小时
- 每个episode更短，减少CPU开销

---

### 方案3: 使用更小的模型（备选）

如果速度还是慢，可以减少网络大小：

```yaml
networks:
  actor:
    hidden_dim: 64    # 从128减到64
  critic:
    fc1_dim: 32       # 从64减到32
```

---

## ⚡ 立即行动

### 快速修复（在Linux服务器）

```bash
# 1. 停止当前训练
# 按 Ctrl+C

# 2. 拉取优化配置
cd ~/paper_v2/paper
git pull

# 3. 应用快速配置
cd marl_framework
cat > params_fast.yaml << 'EOF'
# 复制params.yaml内容，然后修改:
networks:
  data_passes: 2
  batch_size: 32      # 小batch减少CPU压力
  batch_number: 2

experiment:
  missions:
    n_episodes: 500   # 快速测试
  constraints:
    budget: 8         # 大幅减少每个episode的步数
EOF

# 4. 使用快速配置
export CONFIG_FILE_PATH=params_fast.yaml
./train_with_backup.sh fast_test

# 5. 监控GPU
# 在另一个终端
watch -n 1 nvidia-smi
```

---

## 📊 预期效果对比

| 配置 | 总步数 | CPU操作次数 | 预计时间 |
|------|--------|-------------|----------|
| 当前 | 4,800 | ~288,000 | 40-80小时 |
| 优化1 (budget=10) | 2,667 | ~106,680 | 15-30小时 |
| 优化2 (budget=8) | 2,000 | ~64,000 | 10-20小时 |
| 激进 (budget=6) | 1,500 | ~36,000 | 5-10小时 |

*CPU操作次数 = 总步数 × budget × n_agents × 特征数

---

## 🔧 诊断命令

### 检查GPU实际使用情况

```bash
# 1. 实时GPU计算利用率
nvidia-smi dmon -s u

# 2. 进程级监控
nvidia-smi pmon -c 1

# 3. 详细GPU使用
nvtop  # 需要安装: sudo apt install nvtop

# 4. Python进程GPU使用
py-spy top --pid $(pgrep -f "python.*main.py")
```

### 性能分析

```bash
# 使用Python profiler
python -m cProfile -o profile.stats main.py

# 分析结果
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

---

## 💡 长期优化建议

### 1. 异步数据加载
创建DataLoader进行预处理：

```python
# 需要代码重构
class ObservationDataLoader:
    def __init__(self):
        self.pool = multiprocessing.Pool(4)
    
    def preprocess_batch(self, observations):
        # 并行CPU预处理
        return self.pool.map(self.process_one, observations)
```

### 2. GPU上的图像操作
使用kornia替代OpenCV：

```python
import kornia
# GPU上的resize
img_gpu = kornia.geometry.transform.resize(
    img_tensor.cuda(), 
    (new_h, new_w)
)
```

### 3. 缓存常用数据
```python
# 缓存不变的特征图
self.cached_maps = {
    'budget': self.compute_budget_maps(),
    'agent_id': self.compute_agent_id_maps(),
}
```

---

## ⚠️ 当前配置问题总结

### 计算瓶颈分析

当前配置每步需要：
```
1. CPU预处理: ~60-70秒
   - 4个智能体 × 多个特征图
   - OpenCV resize
   - numpy计算
   
2. GPU训练: ~5-10秒
   - 网络前向传播
   - 反向传播
   - 参数更新

总计: ~76秒/步
```

### 优化后预期
```
1. 减少budget: 8 (从14)
   → 每个episode少43%的计算
   
2. 减少batch_size: 32 (从64)
   → 每步少50%的数据处理
   
3. 减少episodes: 500 (从1500)
   → 总训练量减少67%

结果: 预计20-30秒/步
```

---

## 🎯 推荐方案

### 立即执行（最高优先级）

```yaml
# params.yaml - 快速测试配置
experiment:
  constraints:
    budget: 8           # ← 关键！减少CPU计算
  missions:
    n_episodes: 500

networks:
  batch_size: 32
  batch_number: 2
  data_passes: 2
```

**预计效果**:
- 训练步数: ~2,083步
- 时间: 10-15小时
- 足够验证模型性能

---

## 📞 下一步

1. **立即**: 修改配置，减少budget到8
2. **短期**: 观察GPU利用率是否提升
3. **中期**: 如果仍然慢，考虑代码重构
4. **长期**: 实现GPU上的数据预处理

**关键**: budget参数直接影响CPU计算量，是最容易调整且效果最明显的参数！
