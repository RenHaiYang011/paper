"""
调试卷积输出尺寸问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'marl_framework'))

import torch
import torch.nn as nn
import yaml

def test_conv_sizes():
    """测试卷积层输出尺寸"""
    
    print("=" * 60)
    print("测试卷积输出尺寸")
    print("=" * 60)
    
    # 加载配置
    config_path = "marl_framework/configs/params.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.Loader)
    
    # 获取参数
    pix_x = params["sensor"]["pixel"]["number_x"]
    pix_y = params["sensor"]["pixel"]["number_y"]
    
    print(f"\n输入尺寸: {pix_y} x {pix_x}")
    
    # 计算Actor的输入通道数
    input_channels = 9
    if "search_regions" in params:
        input_channels += 3
        print(f"添加区域搜索特征: +3")
    
    intrinsic_rewards_config = params.get("experiment", {}).get("intrinsic_rewards", {})
    if intrinsic_rewards_config.get("enable", False) and intrinsic_rewards_config.get("frontier_reward_weight", 0) > 0:
        state_repr_config = params.get("state_representation", {})
        if state_repr_config.get("use_frontier_map", False):
            input_channels += 1
            print(f"添加前沿地图: +1")
    
    print(f"总输入通道数: {input_channels}")
    
    # 创建卷积层
    print(f"\n{'='*60}")
    print("创建卷积层")
    print('='*60)
    
    conv1 = nn.Conv2d(input_channels, 256, (5, 5))
    conv2 = nn.Conv2d(256, 256, (4, 4))
    conv3 = nn.Conv2d(256, 256, (4, 4))
    activation = nn.ReLU()
    flatten = nn.Flatten()
    
    # 测试前向传播
    print(f"\n前向传播:")
    dummy = torch.zeros(1, input_channels, pix_y, pix_x)
    print(f"  输入形状: {dummy.shape}")
    
    dummy = activation(conv1(dummy))
    print(f"  Conv1 输出: {dummy.shape}")
    
    dummy = activation(conv2(dummy))
    print(f"  Conv2 输出: {dummy.shape}")
    
    dummy = activation(conv3(dummy))
    print(f"  Conv3 输出: {dummy.shape}")
    
    flattened = flatten(dummy)
    conv_out_dim = flattened.shape[1]
    print(f"  Flatten 输出: {flattened.shape}")
    print(f"  展平维度: {conv_out_dim}")
    
    # 测试不同批次大小
    print(f"\n{'='*60}")
    print("测试不同批次大小")
    print('='*60)
    
    for batch_size in [1, 2, 4, 8]:
        dummy = torch.zeros(batch_size, input_channels, pix_y, pix_x)
        dummy = activation(conv1(dummy))
        dummy = activation(conv2(dummy))
        dummy = activation(conv3(dummy))
        flattened = flatten(dummy)
        print(f"  Batch {batch_size}: 输入 {(batch_size, input_channels, pix_y, pix_x)} -> 输出 {flattened.shape}")
    
    # 测试实际网络初始化
    print(f"\n{'='*60}")
    print("测试实际网络初始化")
    print('='*60)
    
    try:
        from actor.network import ActorNetwork
        
        print("正在初始化 ActorNetwork...")
        actor = ActorNetwork(params)
        print(f"✓ Actor 初始化成功")
        print(f"  fc1 权重形状: {actor.fc1.weight.shape}")
        print(f"  fc1 输入维度: {actor.fc1.in_features}")
        print(f"  fc1 输出维度: {actor.fc1.out_features}")
        
        # 测试前向传播
        test_input = torch.randn(1, pix_y, pix_x, input_channels)
        print(f"\n测试前向传播:")
        print(f"  输入形状: {test_input.shape}")
        
        with torch.no_grad():
            probs, hidden = actor.forward(test_input, eps=0.1)
        
        print(f"  输出概率形状: {probs.shape}")
        print(f"  隐藏层形状: {hidden.shape}")
        print(f"  ✓ 前向传播成功")
        
    except Exception as e:
        print(f"✗ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*60}")
    print("✓ 所有测试完成")
    print('='*60)
    return True


if __name__ == "__main__":
    success = test_conv_sizes()
    sys.exit(0 if success else 1)
