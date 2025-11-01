#!/usr/bin/env python3
"""
测试可视化修复效果 - 验证坐标对齐修复后的绘图效果
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'marl_framework'))

from marl_framework.mapping.ground_truths_region_based import generate_region_based_map
from marl_framework.utils.plotting import plot_trajectories

def test_visualization_fix():
    """测试可视化修复效果"""
    
    print("=== 可视化修复测试 ===")
    
    # 加载配置
    config_path = "marl_framework/configs/params_fast.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    # 计算精确分辨率
    min_altitude = params["experiment"]["constraints"]["min_altitude"]
    angle_x = params["sensor"]["field_of_view"]["angle_x"] 
    number_x = params["sensor"]["pixel"]["number_x"]
    
    res_x = (2 * min_altitude * math.tan(math.radians(angle_x) * 0.5)) / number_x
    
    # 计算网格尺寸
    x_dim_m = params["environment"]["x_dim"]
    x_dim_pixels = int(x_dim_m / res_x)
    actual_coverage = x_dim_pixels * res_x
    
    print(f"分辨率: {res_x:.6f} m/pixel")
    print(f"网格尺寸: {x_dim_pixels}×{x_dim_pixels} 像素")
    print(f"实际覆盖: {actual_coverage:.4f}×{actual_coverage:.4f} 米")
    
    # 生成测试地图
    print(f"\n生成测试地图...")
    test_episode = 8888
    simulated_map = generate_region_based_map(params, x_dim_pixels, x_dim_pixels, test_episode)
    
    # 创建模拟轨迹数据 - 确保轨迹经过所有区域
    print(f"创建测试轨迹...")
    n_agents = 2
    n_steps = 10
    
    # Agent 0: 从 high_priority_zone 移动到 medium_priority_zone
    agent0_trajectory = []
    for step in range(n_steps):
        # 从 (12, 12, 10) 移动到 (35, 15, 15)
        progress = step / (n_steps - 1)
        x = 12 + progress * (35 - 12)
        y = 12 + progress * (15 - 12) 
        z = 10 + progress * (15 - 10)
        agent0_trajectory.append([x, y, z])
    
    # Agent 1: 从 medium_priority_zone 移动到 low_priority_zone
    agent1_trajectory = []
    for step in range(n_steps):
        # 从 (40, 20, 12) 移动到 (25, 35, 18)
        progress = step / (n_steps - 1)
        x = 40 + progress * (25 - 40)
        y = 20 + progress * (35 - 20)
        z = 12 + progress * (18 - 12)
        agent1_trajectory.append([x, y, z])
    
    # 组织轨迹数据 - plot_trajectories 期望的格式
    agent_positions = []
    for step in range(n_steps):
        step_positions = {
            0: agent0_trajectory[step],  # Agent 0 position
            1: agent1_trajectory[step]   # Agent 1 position
        }
        agent_positions.append(step_positions)
    
    # 创建简单的障碍物数据 - 使用正确的格式，确保在实际覆盖范围内
    obstacles = [
        {"x": 12, "y": 10, "z": 0, "height": 8},   # 在 high_priority_zone 内
        {"x": 26, "y": 5, "z": 0, "height": 12},  # 在 medium_priority_zone 内  
        {"x": 28, "y": 31, "z": 0, "height": 6}    # 在 low_priority_zone 内
    ]
    
    print(f"障碍物数据:")
    for i, obs in enumerate(obstacles):
        print(f"  - 障碍物 {i+1}: ({obs['x']}, {obs['y']}, {obs['z']}) 高度: {obs['height']}")
        # 验证障碍物位置在覆盖范围内
        if obs['x'] > actual_coverage or obs['y'] > actual_coverage:
            print(f"    ⚠️  警告: 障碍物超出覆盖范围 {actual_coverage:.1f}m")
        else:
            print(f"    ✅ 障碍物位置正常")
    
    print(f"轨迹数据:")
    print(f"  - Agent 0: ({agent0_trajectory[0][0]:.1f}, {agent0_trajectory[0][1]:.1f}) → ({agent0_trajectory[-1][0]:.1f}, {agent0_trajectory[-1][1]:.1f})")
    print(f"  - Agent 1: ({agent1_trajectory[0][0]:.1f}, {agent1_trajectory[0][1]:.1f}) → ({agent1_trajectory[-1][0]:.1f}, {agent1_trajectory[-1][1]:.1f})")
    
    # 调用修复后的绘图函数
    print(f"\n调用修复后的绘图函数...")
    plot_trajectories(
        agent_positions=agent_positions,
        n_agents=n_agents,
        writer=None,  # 不需要 tensorboard writer
        training_step_index=999,
        t_collision=None,
        budget=10,
        simulated_map=simulated_map,
        obstacles=obstacles
    )
    
    print(f"\n=== 可视化测试完成 ===")
    print(f"图像已保存到 marl_framework/log/plots/ 目录")
    print(f"查看 coma_pathes_3d_999.png 验证修复效果")
    
    # 验证坐标覆盖
    print(f"\n验证坐标系:")
    print(f"1. 地图坐标范围: [0, {actual_coverage:.4f}] × [0, {actual_coverage:.4f}] 米")
    print(f"2. Agent 0 轨迹: 应该在 high_priority_zone (红色) 开始")
    print(f"3. Agent 1 轨迹: 应该在 medium_priority_zone (紫色) 开始") 
    print(f"4. 如果轨迹与颜色区域对齐，说明坐标修复成功！")

if __name__ == "__main__":
    test_visualization_fix()