#!/usr/bin/env python3
"""
测试通道数计算 - 调试网络输入维度不匹配问题
"""

import sys
import os

# 添加marl_framework到路径
sys.path.append('marl_framework')

def test_channel_calculation():
    """测试Actor和Critic的通道数计算"""
    print("🔧 测试网络通道数计算")
    print("=" * 50)
    
    try:
        # 读取配置
        import yaml
        with open('marl_framework/configs/params_fast.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        print("📋 配置加载成功")
        
        # 测试Actor网络通道数
        print("\n🎭 Actor网络通道数计算:")
        
        # 基础层：9层
        actor_channels = 9
        print(f"  基础层: {actor_channels}")
        
        # 区域搜索特征
        if "search_regions" in params:
            actor_channels += 3
            print(f"  + 区域搜索特征: 3层")
        
        # 前沿探测特征
        intrinsic_rewards_config = params.get("experiment", {}).get("intrinsic_rewards", {})
        if intrinsic_rewards_config.get("enable", False) and intrinsic_rewards_config.get("frontier_reward_weight", 0) > 0:
            state_repr_config = params.get("state_representation", {})
            if state_repr_config.get("use_frontier_map", False):
                actor_channels += 1
                print(f"  + 前沿探测特征: 1层")
        
        print(f"  Actor总通道数: {actor_channels}")
        
        # 测试Critic网络通道数
        print("\n🎯 Critic网络通道数计算:")
        print(f"  Actor输入: {actor_channels}层")
        
        # Critic额外特征
        critic_additional = 5  # position_map(1) + w_entropy_map(1) + prob_map(1) + footprint_map(1) + other_actions_map(1)
        print(f"  + Critic额外特征: {critic_additional}层")
        print(f"    - position_map: 1层")
        print(f"    - w_entropy_map: 1层")
        print(f"    - prob_map: 1层")
        print(f"    - footprint_map: 1层")
        print(f"    - other_actions_map: 1层")
        
        critic_channels = actor_channels + critic_additional
        print(f"  Critic总通道数: {critic_channels}")
        
        # 创建网络实例验证
        print("\n🧪 创建网络实例验证:")
        
        from actor.network import ActorNetwork
        from critic.network import CriticNetwork
        
        # 模拟logger
        import logging
        logging.basicConfig(level=logging.INFO)
        
        actor_net = ActorNetwork(params)
        print(f"  ✅ Actor网络创建成功，通道数: {actor_net.input_channels}")
        
        critic_net = CriticNetwork(params)
        print(f"  ✅ Critic网络创建成功，通道数: {critic_net.input_channels}")
        
        # 检查是否匹配
        expected_critic_channels = actor_net.input_channels + 5
        if critic_net.input_channels == expected_critic_channels:
            print(f"  ✅ 通道数匹配! Critic = Actor({actor_net.input_channels}) + 额外(5) = {critic_net.input_channels}")
        else:
            print(f"  ❌ 通道数不匹配! 期望: {expected_critic_channels}, 实际: {critic_net.input_channels}")
        
        return actor_net.input_channels, critic_net.input_channels
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    actor_channels, critic_channels = test_channel_calculation()
    
    if actor_channels and critic_channels:
        print(f"\n🎉 测试完成!")
        print(f"Actor通道数: {actor_channels}")
        print(f"Critic通道数: {critic_channels}")
        print(f"期望匹配: Actor + 5 = {actor_channels + 5}")
        
        if critic_channels == actor_channels + 5:
            print("✅ 通道数配置正确，可以开始训练！")
        else:
            print("❌ 通道数配置错误，需要进一步调试")
    else:
        print("❌ 测试失败，请检查配置")