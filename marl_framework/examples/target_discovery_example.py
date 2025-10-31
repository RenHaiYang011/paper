"""
目标发现奖励机制使用示例

本示例展示如何使用新的目标发现奖励机制：
1. 目标发现奖励
2. 任务完成/失败奖励  
3. 协同发现奖励
4. 改进的状态表示
"""

import numpy as np
import yaml
from marl_framework.utils.reward import (
    get_global_reward,
    detect_new_target_discoveries,
    calculate_collaborative_discovery_reward
)
from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.actor.transformations import (
    get_network_input,
    get_discovery_history_map,
    get_exploration_intensity_map
)


def load_enhanced_config():
    """加载增强的配置文件"""
    config = {
        "environment": {
            "seed": 42,
            "x_dim": 100,
            "y_dim": 100
        },
        "experiment": {
            "constraints": {
                "spacing": 5,
                "min_altitude": 5,
                "max_altitude": 25,
                "budget": 50,
                "num_actions": 6
            },
            "missions": {
                "n_agents": 4,
                "class_weighting": [0, 1],
                "planning_uncertainty": "SE"
            },
            # 新增的目标发现奖励配置
            "target_discovery_reward": 50.0,
            "mission_success_reward": 100.0,
            "mission_failure_penalty": -50.0,
            "collaborative_discovery_weight": 25.0,
            "discovery_threshold": 0.8
        },
        # 新增的状态表示配置
        "state_representation": {
            "use_discovery_history": True,
            "use_exploration_intensity": True,
            "use_target_probability": True,
            "exploration_decay_factor": 0.9,
            "discovery_influence_radius": 2
        }
    }
    return config


def simulate_target_discovery_scenario():
    """模拟目标发现场景"""
    print("=== 目标发现奖励机制演示 ===\n")
    
    # 1. 初始化配置
    config = load_enhanced_config()
    agent_state_space = AgentStateSpace(config)
    
    # 2. 创建模拟地图和智能体状态
    map_size = (20, 20)  # 简化的地图尺寸
    
    # 创建包含3个目标的真实地图
    simulated_map = np.zeros(map_size)
    target_positions = [(5, 5), (10, 15), (15, 8)]
    for pos in target_positions:
        simulated_map[pos] = 1.0
    
    print(f"创建了包含 {len(target_positions)} 个目标的地图")
    print(f"目标位置: {target_positions}\n")
    
    # 3. 模拟发现过程
    discovered_targets = set()
    total_targets = len(target_positions)
    
    # 智能体位置
    agent_positions = [
        np.array([25, 25, 10]),  # Agent 0
        np.array([50, 25, 15]),  # Agent 1  
        np.array([25, 75, 20]),  # Agent 2
        np.array([75, 75, 25])   # Agent 3
    ]
    
    # 4. 模拟多个时间步的发现过程
    for t in range(10):
        print(f"\n--- 时间步 {t} ---")
        
        # 模拟地图更新（智能体逐渐发现目标）
        last_map = np.random.rand(*map_size) * 0.3  # 之前的观测
        next_map = np.random.rand(*map_size) * 0.3  # 当前观测
        
        # 模拟在某些时间步发现目标
        if t == 3:
            # 发现第一个目标
            next_map[5, 5] = 0.9  # 高置信度
            print("🎯 Agent 0 发现了第一个目标!")
            
        elif t == 7:
            # 发现第二个目标
            next_map[10, 15] = 0.85
            print("🎯 Agent 1 发现了第二个目标!")
            
        # 检测新发现
        new_discoveries = detect_new_target_discoveries(
            last_map, next_map, simulated_map, agent_state_space, 
            discovered_targets, discovery_threshold=0.8
        )
        
        if new_discoveries > 0:
            print(f"✨ 新发现 {new_discoveries} 个目标!")
            
            # 计算协同发现奖励
            discovering_agent = 0 if t == 3 else 1
            collab_rewards = calculate_collaborative_discovery_reward(
                discovering_agent, agent_positions, agent_positions[discovering_agent]
            )
            print(f"🤝 协同发现奖励: {collab_rewards}")
        
        # 5. 计算综合奖励
        for agent_id in range(len(agent_positions)):
            reward_result = get_global_reward(
                last_map=last_map,
                next_map=next_map,
                mission_type="COMA",
                footprints=None,
                simulated_map=simulated_map,
                agent_state_space=agent_state_space,
                actions=None,
                agent_id=agent_id,
                t=t,
                budget=config["experiment"]["constraints"]["budget"],
                next_positions=agent_positions,
                # 新的目标发现参数
                target_discovery_reward=config["experiment"]["target_discovery_reward"],
                mission_success_reward=config["experiment"]["mission_success_reward"],
                mission_failure_penalty=config["experiment"]["mission_failure_penalty"],
                discovered_targets=discovered_targets,
                total_targets=total_targets
            )
            
            done, relative_reward, absolute_reward = reward_result
            
            if new_discoveries > 0 and agent_id == (0 if t == 3 else 1):
                print(f"💰 Agent {agent_id} 奖励: {absolute_reward:.2f} (包含发现奖励)")
            
            if done:
                if len(discovered_targets) >= total_targets:
                    print(f"🏆 任务成功完成! Agent {agent_id} 获得成功奖励")
                else:
                    print(f"💔 任务失败! Agent {agent_id} 受到失败惩罚")
                break
    
    print(f"\n=== 最终结果 ===")
    print(f"发现目标数量: {len(discovered_targets)} / {total_targets}")
    print(f"发现率: {len(discovered_targets)/total_targets*100:.1f}%")


def demonstrate_enhanced_state_representation():
    """演示增强的状态表示功能"""
    print("\n=== 增强状态表示演示 ===\n")
    
    # 模拟已发现的目标
    discovered_targets = {(5, 5), (10, 15)}
    map_size = (20, 20)
    
    # 创建AgentStateSpace (简化)
    config = load_enhanced_config()
    agent_state_space = AgentStateSpace(config)
    position_map = np.ones(map_size)
    
    # 1. 目标发现历史图
    discovery_map = get_discovery_history_map(
        discovered_targets, agent_state_space, position_map
    )
    print("📍 生成目标发现历史图")
    print(f"   - 已发现目标数量: {len(discovered_targets)}")
    print(f"   - 历史图中非零位置: {np.sum(discovery_map > 0)}")
    
    # 2. 探索强度图
    # 模拟本地信息
    local_information = {
        0: {"map2communicate": np.random.rand(*map_size)},
        1: {"map2communicate": np.random.rand(*map_size)},
        2: {"map2communicate": np.random.rand(*map_size)},
        3: {"map2communicate": np.random.rand(*map_size)}
    }
    
    intensity_map = get_exploration_intensity_map(
        local_information, 0, agent_state_space, position_map, 5
    )
    print("🔍 生成探索强度图")
    print(f"   - 平均探索强度: {np.mean(intensity_map):.3f}")
    print(f"   - 最高探索强度: {np.max(intensity_map):.3f}")
    
    print("\n✅ 状态表示增强完成!")


def main():
    """主函数"""
    print("🚁 多智能体目标发现系统演示\n")
    
    # 1. 演示目标发现奖励机制
    simulate_target_discovery_scenario()
    
    # 2. 演示增强的状态表示
    demonstrate_enhanced_state_representation()
    
    print("\n🎉 演示完成!")
    print("\n主要改进点:")
    print("1. ✅ 目标发现奖励 - 解决稀疏奖励问题")
    print("2. ✅ 任务完成/失败奖励 - 明确的成功/失败信号") 
    print("3. ✅ 协同发现奖励 - COMA信用分配机制")
    print("4. ✅ 增强状态表示 - 发现历史和探索强度")
    print("5. ✅ 配置化参数 - 易于调节和实验")


if __name__ == "__main__":
    main()