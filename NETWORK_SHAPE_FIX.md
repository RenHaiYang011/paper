# 🔧 网络形状问题修复总结

## 📋 问题诊断

### 问题1: 动作掩码维度不匹配
```
ValueError: operands could not be broadcast together with shapes (63,) (27,)
```

**原因：**
- `get_action_mask()` 在某些位置（如 [25, 25, 15]）返回了 63 维的掩码
- 但期望的是 27 维（对应 27 个动作）
- 这是因为代码将 3D 掩码 (3, 3, 7) 展平后得到 63 维

### 问题2: 全连接层维度不匹配
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x12544 and 565504x256)
```

**原因：**
- Actor 和 Critic 网络的 fc1 层输入维度硬编码为 256
- 但实际卷积层输出的展平维度是 12544
- 导致矩阵乘法维度不匹配

---

## ✅ 修复方案

### 修复1: `action_space.py` - 动作掩码生成

**修改位置：** `marl_framework/agent/action_space.py` 的 `get_action_mask()` 方法

**关键改动：**
```python
# 旧代码（错误）：
elif self.num_actions == 9:
    mask = np.squeeze(np.ones((self.space_x_dim, self.space_y_dim, self.space_z_dim)))
    # ... 边界检查 ...
    mask_flatten = mask.flatten()  # 可能产生 63 维

# 新代码（正确）：
elif self.num_actions == 9:
    mask = np.ones(9)  # 直接创建一维掩码
    # 对每个动作检查是否越界
    for action_idx in range(9):
        next_pos = self.action_to_position(position.copy(), action_idx)
        if not self._is_position_valid(next_pos):
            mask[action_idx] = 0
    mask[4] = 0  # 中心点（不移动）
```

**效果：**
- ✅ 始终返回长度为 `num_actions` 的一维掩码
- ✅ 适用于所有动作空间（4, 6, 9, 27）
- ✅ 正确处理边界情况

### 修复2: `actor/network.py` - 动态计算 fc1 维度

**修改位置：** `marl_framework/actor/network.py` 的 `__init__()` 方法

**关键改动：**
```python
# 旧代码（错误）：
self.fc1 = nn.Linear(256, self.hidden_dim)  # 硬编码输入维度

# 新代码（正确）：
# 动态计算卷积输出维度
with torch.no_grad():
    dummy = torch.zeros(1, self.input_channels, pix_y, pix_x)
    dummy = self.activation(self.conv1(dummy))
    dummy = self.activation(self.conv2(dummy))
    dummy = self.activation(self.conv3(dummy))
    conv_out_dim = int(self.flatten(dummy).shape[1])

self.fc1 = nn.Linear(conv_out_dim, self.hidden_dim)  # 使用实际维度
```

**效果：**
- ✅ 自动适配不同的输入尺寸（57x57）
- ✅ 自动适配不同的输入通道数（9-13）
- ✅ 避免硬编码导致的维度错误

### 修复3: `critic/network.py` - 动态计算 fc1 维度

**修改位置：** `marl_framework/critic/network.py` 的 `__init__()` 方法

**关键改动：** 与 Actor 相同
```python
# 动态计算卷积输出维度
with torch.no_grad():
    dummy = torch.zeros(1, self.input_channels, pix_y, pix_x)
    dummy = self.activation(self.conv1(dummy))
    dummy = self.activation(self.conv2(dummy))
    dummy = self.activation(self.conv3(dummy))
    conv_out_dim = int(self.flatten(dummy).shape[1])

self.fc1 = nn.Linear(conv_out_dim, 256)
```

---

## 🧪 验证方法

### 方法1: 运行测试脚本（Linux）

```bash
cd ~/paper_v2/paper
python test_action_mask_fix.py
```

**期望输出：**
```
=== 测试27动作空间掩码 ===
测试位置 1: [ 0  0 10]
  ✓ 掩码长度正确 (27)
测试位置 2: [25 25 15]
  ✓ 掩码长度正确 (27)  # 之前是63，现在修复了
测试位置 3: [ 0 25 10]
  ✓ 掩码长度正确 (27)
测试位置 4: [50 50 20]
  ✓ 掩码长度正确 (27)
```

### 方法2: 运行网络测试脚本（Windows）

```powershell
cd E:\code\paper_code\paper
python test_network_shapes.py
```

**期望输出：**
```
测试 Actor Network
✓ Actor 初始化成功
  - 输入通道数: 13
  - 输出动作数正确 (27)

测试 Critic Network
✓ Critic 初始化成功
  - 输入通道数: 18
  - 输出动作数正确 (27)

✓ 所有测试通过！
```

### 方法3: 启动训练验证（最终验证）

```bash
# Linux
cd ~/paper_v2/paper/marl_framework/scripts
CONFIG_FILE_PATH=configs/params.yaml ./train_with_backup.sh test_fix

# Windows
cd E:\code\paper_code\paper\running
.\start_training.ps1
```

**期望结果：**
- ✅ 训练正常启动，无形状错误
- ✅ 第一个 episode 能够成功完成
- ✅ TensorBoard 正常记录指标
- ✅ 生成的轨迹图正常显示

---

## 📊 技术细节

### 为什么需要动态计算维度？

卷积层输出的尺寸计算公式：
```
output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

对于我们的网络（无 padding，stride=1）：
- **输入**: 57x57
- **Conv1 (5x5)**: 53x53
- **Conv2 (4x4)**: 50x50
- **Conv3 (4x4)**: 47x47
- **通道数**: 256
- **展平后**: 47 × 47 × 256 = **564,224 维** （不是 256！）

但是：
- 如果输入通道数变化（9 vs 13）
- 如果输入尺寸变化（57x57 vs 其他）
- 卷积层数量变化
- 卷积核大小变化

输出维度都会改变！所以必须动态计算。

### 为什么掩码会生成错误维度？

原代码的问题：
```python
# 错误示例（num_actions=27）
mask = np.ones((3, 3, 7))  # 空间维度 3x3，高度维度 7
# ... 边界处理 ...
mask_flatten = mask.flatten()  # 3*3*7 = 63 维！
```

实际上应该：
```python
# 正确做法
mask = np.ones(27)  # 直接创建 27 维一维数组
# 对每个动作单独检查
```

---

## 🎯 测试清单

在确认修复前，请检查：

- [ ] `test_action_mask_fix.py` 所有测试通过
- [ ] `test_network_shapes.py` 所有测试通过
- [ ] 训练能够正常启动（无形状错误）
- [ ] 至少完成 1-2 个完整 episode
- [ ] TensorBoard 显示正常指标
- [ ] 生成的轨迹图无异常

---

## 🔄 如果还有问题

### 症状：仍然有维度不匹配错误

**检查：**
1. 确认 git pull 获取了最新代码
2. 重启 Python 进程（清除缓存的模块）
3. 检查配置文件中的 `num_actions` 值

### 症状：训练过慢或内存不足

**调整：**
```yaml
# params.yaml
experiment:
  constraints:
    budget: 30  # 减少步数
  missions:
    n_episodes: 300  # 减少轮数

networks:
  batch_size: 64  # 减小 batch size
```

### 症状：碰撞率过高

**调整：**
```yaml
experiment:
  obstacles:
    collision_penalty: 15.0  # 提高惩罚
    safety_margin: 2.0  # 增大安全边界
```

---

## 📝 修改文件列表

1. ✅ `marl_framework/agent/action_space.py`
   - 修复 `get_action_mask()` 方法
   - 添加 `_is_position_valid()` 辅助方法
   - 修复 `apply_obstacle_mask()` 方法

2. ✅ `marl_framework/actor/network.py`
   - 动态计算 `fc1` 输入维度

3. ✅ `marl_framework/critic/network.py`
   - 动态计算 `fc1` 输入维度

4. ✅ `test_action_mask_fix.py`（新增）
   - 测试动作掩码生成

5. ✅ `test_network_shapes.py`（新增）
   - 测试网络形状

---

## 🚀 下一步

修复验证通过后：

1. **提交代码**
   ```bash
   git add .
   git commit -m "Fix action mask and network shape issues for 27-action space"
   git push
   ```

2. **开始完整训练**
   ```bash
   # 预计时间：48-72小时
   CONFIG_FILE_PATH=configs/params.yaml ./train_with_backup.sh reg_search_v1
   ```

3. **监控训练**
   - 定期查看 TensorBoard
   - 观察奖励曲线是否上升
   - 检查生成的轨迹图

4. **分析结果**
   - 轨迹是否曲折
   - 是否在目标区域聚集
   - 是否成功避障
   - 多机是否协同

---

## 📞 技术支持

如果遇到其他问题，请提供：
1. 完整的错误信息
2. 相关配置文件（params.yaml）
3. 测试脚本的输出
4. 系统环境信息（Python版本、PyTorch版本）

祝训练顺利！🎉
