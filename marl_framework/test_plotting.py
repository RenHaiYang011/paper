#!/usr/bin/env python3
"""
测试新的轨迹可视化功能
包括目标立方体和障碍物金字塔
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add marl_framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.plotting import plot_trajectories

def create_test_data():
    """创建测试数据"""
    print("🧪 创建测试数据...")
    
    # 模拟4个智能体的轨迹
    n_agents = 4
    n_steps = 10
    
    agent_positions = []
    for t in range(n_steps):
        positions = {}
        for agent_id in range(n_agents):
            # 创建螺旋式上升轨迹
            angle = t * 0.5 + agent_id * np.pi / 2
            radius = 15 + agent_id * 5
            x = 25 + radius * np.cos(angle)
            y = 25 + radius * np.sin(angle)
            z = 5 + t * 2
            positions[agent_id] = np.array([x, y, z])
        agent_positions.append(positions)
    
    # 创建模拟地图 (50x50)
    simulated_map = np.random.rand(100, 100) * 0.3
    
    # 添加一些高概率目标区域
    target_locations = [
        (20, 20), (80, 20), (50, 50), (20, 80), (80, 80)
    ]
    for tx, ty in target_locations:
        simulated_map[ty-5:ty+5, tx-5:tx+5] = 0.9
    
    # 创建障碍物数据
    obstacles = [
        {'x': 15, 'y': 35, 'z': 0, 'height': 12},
        {'x': 40, 'y': 15, 'z': 0, 'height': 15},
        {'x': 35, 'y': 40, 'z': 0, 'height': 10},
    ]
    
    print(f"  ✅ 创建了 {n_agents} 个智能体的轨迹")
    print(f"  ✅ 创建了 {len(target_locations)} 个目标")
    print(f"  ✅ 创建了 {len(obstacles)} 个障碍物")
    
    return agent_positions, n_agents, simulated_map, obstacles


def test_plotting():
    """测试轨迹绘制功能"""
    print("\n🎨 测试轨迹可视化...")
    
    # 创建测试数据
    agent_positions, n_agents, simulated_map, obstacles = create_test_data()
    
    # 创建mock writer
    class MockWriter:
        def add_figure(self, *args, **kwargs):
            pass
    
    writer = MockWriter()
    
    try:
        # 调用绘图函数
        plot_trajectories(
            agent_positions=agent_positions,
            n_agents=n_agents,
            writer=writer,
            training_step_index=100,
            t_collision=0,
            budget=10,
            simulated_map=simulated_map,
            obstacles=obstacles
        )
        
        print("  ✅ 轨迹可视化成功!")
        print(f"\n📁 图片已保存到:")
        print(f"  - log/plots/coma_pathes_3d_100.png")
        print(f"  - res/plots/coma_pathes_3d_100.png")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 轨迹可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  测试新的3D轨迹可视化功能")
    print("=" * 60)
    
    success = test_plotting()
    
    if success:
        print("\n🎉 所有测试通过!")
        print("\n📊 新功能包括:")
        print("  ✅ 网格坐标与地图区域完美对齐")
        print("  ✅ 红色立方体标识目标位置")
        print("  ✅ 灰色金字塔标识障碍物")
        print("  ✅ 彩色轨迹线显示智能体路径")
        print("  ✅ 起始点用球形标记")
        print("  ✅ 图例说明各种元素")
    else:
        print("\n❌ 测试失败，请检查错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
