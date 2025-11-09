"""
测试27动作空间的掩码修复
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'marl_framework'))

import numpy as np
from agent.action_space import AgentActionSpace

# 创建测试参数
test_params = {
    "experiment": {
        "constraints": {
            "spacing": 3,
            "min_altitude": 5,
            "max_altitude": 25,
            "num_actions": 27
        }
    },
    "environment": {
        "x_dim": 50,
        "y_dim": 50
    }
}

# 创建动作空间
action_space = AgentActionSpace(test_params, obstacle_manager=None)

print("=== 测试27动作空间掩码 ===\n")

# 测试不同位置的掩码
test_positions = [
    np.array([0, 0, 10]),      # 角落
    np.array([25, 25, 15]),    # 中心
    np.array([0, 25, 10]),     # 边界
    np.array([50, 50, 20]),    # 另一个角落
]

for i, pos in enumerate(test_positions):
    print(f"测试位置 {i+1}: {pos}")
    
    # 获取动作掩码
    mask_1d, mask_nd = action_space.get_action_mask(pos)
    
    print(f"  - mask_1d shape: {mask_1d.shape}")
    print(f"  - mask_nd shape: {mask_nd.shape if hasattr(mask_nd, 'shape') else 'N/A'}")
    print(f"  - mask_1d length: {len(mask_1d)}")
    print(f"  - 有效动作数: {np.sum(mask_1d)}")
    print(f"  - num_actions: {action_space.num_actions}")
    
    # 验证长度匹配
    if len(mask_1d) == action_space.num_actions:
        print("  ✓ 掩码长度正确\n")
    else:
        print(f"  ✗ 掩码长度错误！期望{action_space.num_actions}，实际{len(mask_1d)}\n")

print("=== 测试apply_obstacle_mask（无障碍物）===\n")

# 测试apply_obstacle_mask函数（无障碍物管理器）
for i, pos in enumerate(test_positions[:2]):
    print(f"测试位置 {i+1}: {pos}")
    
    mask_1d, _ = action_space.get_action_mask(pos)
    print(f"  - 输入掩码长度: {len(mask_1d)}")
    
    # 应用障碍物掩码（应该直接返回原掩码）
    result_mask = action_space.apply_obstacle_mask(pos, mask_1d, None)
    print(f"  - 输出掩码长度: {len(result_mask)}")
    
    if len(result_mask) == len(mask_1d):
        print("  ✓ apply_obstacle_mask 正常工作\n")
    else:
        print(f"  ✗ apply_obstacle_mask 异常！\n")

print("=== 测试不同动作空间 ===\n")

for num_actions in [4, 6, 9, 27]:
    test_params["experiment"]["constraints"]["num_actions"] = num_actions
    action_space = AgentActionSpace(test_params, obstacle_manager=None)
    
    pos = np.array([25, 25, 15])
    mask_1d, _ = action_space.get_action_mask(pos)
    
    print(f"num_actions={num_actions}: mask长度={len(mask_1d)}, 期望={num_actions}")
    if len(mask_1d) == num_actions:
        print("  ✓ 正确\n")
    else:
        print("  ✗ 错误\n")

print("测试完成！")
