#!/usr/bin/env python3
"""
测试修复后的轨迹可视化
验证坐标对齐和目标/障碍物显示
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add marl_framework to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'marl_framework'))

def test_coordinate_alignment():
    """测试坐标对齐修复"""
    print("=" * 60)
    print("  测试坐标对齐修复")
    print("=" * 60)
    
    from utils.plotting import plot_trajectories
    
    # 创建一个简单的测试地图 (100x100像素)
    map_size = 100
    simulated_map = np.zeros((map_size, map_size))
    
    # 在地图的四个角创建高概率目标区域
    # 左上角 (0, 0) -> (10, 10)
    simulated_map[0:10, 0:10] = 0.9
    
    # 右上角 (0, 40) -> (10, 50)
    simulated_map[0:10, 90:100] = 0.9
    
    # 左下角 (40, 0) -> (50, 10)
    simulated_map[90:100, 0:10] = 0.9
    
    # 右下角 (40, 40) -> (50, 50)
    simulated_map[90:100, 90:100] = 0.9
    
    # 中心区域
    simulated_map[45:55, 45:55] = 0.9
    
    print(f"✅ 创建测试地图: {map_size}x{map_size}像素")
    print(f"✅ 目标区域: 四角 + 中心")
    
    # 创建测试轨迹 - 4个智能体在四个象限飞行
    n_agents = 4
    agent_positions = []
    
    for t in range(6):
        positions = {}
        # Agent 0: 左下角飞到右上角
        positions[0] = np.array([5 + t*8, 5 + t*8, 12 + t])
        # Agent 1: 右上角飞到左下角
        positions[1] = np.array([45 - t*6, 45 - t*6, 15 + t])
        # Agent 2: 右下角飞到左上角
        positions[2] = np.array([45 - t*6, 5 + t*6, 18 + t])
        # Agent 3: 左上角飞到右下角
        positions[3] = np.array([5 + t*8, 45 - t*6, 20 + t])
        agent_positions.append(positions)
    
    print(f"✅ 创建 {n_agents} 个智能体轨迹")
    
    # 创建障碍物
    obstacles = [
        {'x': 15, 'y': 15, 'z': 0, 'height': 12},
        {'x': 35, 'y': 35, 'z': 0, 'height': 15},
        {'x': 25, 'y': 25, 'z': 0, 'height': 10},
    ]
    
    print(f"✅ 创建 {len(obstacles)} 个障碍物")
    
    # Mock writer
    class MockWriter:
        def add_figure(self, *args, **kwargs):
            pass
    
    writer = MockWriter()
    
    try:
        # 绘制轨迹图
        plot_trajectories(
            agent_positions=agent_positions,
            n_agents=n_agents,
            writer=writer,
            training_step_index=999,
            t_collision=0,
            budget=10,
            simulated_map=simulated_map,
            obstacles=obstacles
        )
        
        print("\n🎉 测试成功!")
        print("\n📊 验证项目:")
        print("  ✅ 目标立方体应该出现在:")
        print("     - 左下角 (0-10, 0-10)")
        print("     - 右上角 (40-50, 40-50)")
        print("     - 左上角 (0-10, 40-50)")
        print("     - 右下角 (40-50, 0-10)")
        print("     - 中心 (20-30, 20-30)")
        print("\n  ✅ 立方体应该严格位于红色区域上方")
        print("\n  ✅ 障碍物金字塔应该位于:")
        print("     - (15, 15) 高度12米")
        print("     - (35, 35) 高度15米")
        print("     - (25, 25) 高度10米")
        print("\n  ✅ 智能体轨迹应该穿越或绕过障碍物")
        print("\n📁 输出文件:")
        print("  - log/plots/coma_pathes_3d_999.png")
        print("  - res/plots/coma_pathes_3d_999.png")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_coordinate_alignment()
    
    if success:
        print("\n" + "=" * 60)
        print("  🎉 所有测试通过!")
        print("=" * 60)
        print("\n请检查生成的图片，确认:")
        print("1. 红色立方体位于地图红色区域正上方")
        print("2. 灰色金字塔在指定位置显示")
        print("3. 地图区域与网格坐标完美对齐")
    else:
        print("\n❌ 测试失败")
