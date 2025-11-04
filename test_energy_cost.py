"""测试能耗惩罚功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np
from marl_framework.utils.reward import get_global_reward
from marl_framework.agent.state_space import AgentStateSpace

def test_energy_cost():
    """测试能耗惩罚是否正常工作"""
    print("="*60)
    print("测试能耗惩罚功能")
    print("="*60)
    
    # 加载配置
    with open('marl_framework/configs/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    energy_cost = params['experiment'].get('energy_cost_per_step', 0.0)
    print(f"\n✓ 配置中的能耗惩罚: {energy_cost}")
    
    # 创建简单的测试数据
    map_size = 50
    last_map = np.random.rand(map_size, map_size)
    next_map = np.random.rand(map_size, map_size)
    simulated_map = np.random.rand(map_size, map_size)
    
    # 创建state space
    state_space = AgentStateSpace(
        map_size=map_size,
        map_resolution=1.0
    )
    
    # 测试1: 无能耗惩罚
    print("\n" + "-"*60)
    print("测试1: 无能耗惩罚 (energy_cost_per_step=0.0)")
    print("-"*60)
    
    done1, rel_reward1, abs_reward1 = get_global_reward(
        last_map, next_map, "COMA", None, simulated_map,
        state_space, None, 0, 1, 14,
        coverage_weight=0.15,
        energy_cost_per_step=0.0
    )
    
    print(f"相对奖励: {rel_reward1:.4f}")
    print(f"绝对奖励: {abs_reward1:.4f}")
    
    # 测试2: 有能耗惩罚
    print("\n" + "-"*60)
    print(f"测试2: 有能耗惩罚 (energy_cost_per_step={energy_cost})")
    print("-"*60)
    
    done2, rel_reward2, abs_reward2 = get_global_reward(
        last_map, next_map, "COMA", None, simulated_map,
        state_space, None, 0, 1, 14,
        coverage_weight=0.15,
        energy_cost_per_step=energy_cost
    )
    
    print(f"相对奖励: {rel_reward2:.4f}")
    print(f"绝对奖励: {abs_reward2:.4f}")
    
    # 验证差异
    print("\n" + "-"*60)
    print("对比分析")
    print("-"*60)
    
    abs_diff = abs_reward1 - abs_reward2
    print(f"绝对奖励差值: {abs_diff:.4f}")
    print(f"预期差值: {energy_cost:.4f}")
    
    if abs(abs_diff - energy_cost) < 0.001:
        print("✓ 能耗惩罚正确应用!")
        print(f"✓ 每步扣除 {energy_cost} 的奖励")
    else:
        print("✗ 能耗惩罚可能有问题")
        print(f"  预期差值: {energy_cost}")
        print(f"  实际差值: {abs_diff}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    
    # 模拟14步任务的总能耗
    budget = 14
    total_energy_cost = energy_cost * budget
    print(f"\n估算: 完成14步任务的总能耗惩罚: {total_energy_cost:.2f}")
    print(f"这将激励Agent尽快完成任务,减少不必要的飞行步数")

if __name__ == "__main__":
    test_energy_cost()
