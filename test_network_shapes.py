"""
测试网络形状修复
验证 Actor 和 Critic 网络能够正确处理不同的输入尺寸
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'marl_framework'))

import torch
import yaml

def test_network_shapes():
    """测试 Actor 和 Critic 网络的形状"""
    
    print("=" * 50)
    print("测试网络形状修复")
    print("=" * 50)
    
    # 加载配置
    config_path = "marl_framework/configs/params.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.Loader)
    
    # 获取参数
    pix_x = params["sensor"]["pixel"]["number_x"]
    pix_y = params["sensor"]["pixel"]["number_y"]
    n_actions = params["experiment"]["constraints"]["num_actions"]
    
    print(f"\n配置参数:")
    print(f"  - 像素尺寸: {pix_x} x {pix_y}")
    print(f"  - 动作数量: {n_actions}")
    
    # 测试 Actor Network
    print(f"\n{'='*50}")
    print("测试 Actor Network")
    print('='*50)
    
    try:
        from actor.network import ActorNetwork
        
        actor = ActorNetwork(params)
        print(f"✓ Actor 初始化成功")
        print(f"  - 输入通道数: {actor.input_channels}")
        print(f"  - 隐藏层维度: {actor.hidden_dim}")
        print(f"  - 动作数: {actor.n_actions}")
        
        # 测试前向传播
        batch_size = 2
        dummy_input = torch.randn(batch_size, pix_y, pix_x, actor.input_channels)
        
        print(f"\n测试前向传播:")
        print(f"  - 输入形状: {dummy_input.shape}")
        
        with torch.no_grad():
            probs, hidden = actor.forward(dummy_input, eps=0.1)
        
        print(f"  - 输出概率形状: {probs.shape}")
        print(f"  - 隐藏层形状: {hidden.shape}")
        
        if probs.shape[1] == n_actions:
            print(f"  ✓ 输出动作数正确 ({n_actions})")
        else:
            print(f"  ✗ 输出动作数错误: 期望 {n_actions}, 实际 {probs.shape[1]}")
            return False
            
    except Exception as e:
        print(f"✗ Actor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 Critic Network
    print(f"\n{'='*50}")
    print("测试 Critic Network")
    print('='*50)
    
    try:
        from critic.network import CriticNetwork
        
        critic = CriticNetwork(params)
        print(f"✓ Critic 初始化成功")
        print(f"  - 输入通道数: {critic.input_channels}")
        print(f"  - 动作数: {critic.n_actions}")
        
        # 测试前向传播
        # Critic 的输入通道数更多（包含额外的全局特征）
        dummy_input = torch.randn(batch_size, pix_y, pix_x, critic.input_channels)
        
        print(f"\n测试前向传播:")
        print(f"  - 输入形状: {dummy_input.shape}")
        
        with torch.no_grad():
            q_values, log_probs = critic.forward(dummy_input)
        
        print(f"  - Q值形状: {q_values.shape}")
        print(f"  - Log概率形状: {log_probs.shape}")
        
        # 检查输出维度
        expected_shape = (batch_size, n_actions) if q_values.dim() == 2 else (n_actions,)
        actual_last_dim = q_values.shape[-1]
        
        if actual_last_dim == n_actions:
            print(f"  ✓ 输出动作数正确 ({n_actions})")
        else:
            print(f"  ✗ 输出动作数错误: 期望 {n_actions}, 实际 {actual_last_dim}")
            return False
            
    except Exception as e:
        print(f"✗ Critic 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试不同的批次大小
    print(f"\n{'='*50}")
    print("测试不同批次大小")
    print('='*50)
    
    for batch_size in [1, 4, 8, 16]:
        try:
            dummy_input = torch.randn(batch_size, pix_y, pix_x, actor.input_channels)
            with torch.no_grad():
                probs, hidden = actor.forward(dummy_input, eps=0.1)
            
            if probs.shape == (batch_size, n_actions):
                print(f"  ✓ Batch size {batch_size}: {probs.shape}")
            else:
                print(f"  ✗ Batch size {batch_size}: 期望 ({batch_size}, {n_actions}), 实际 {probs.shape}")
                return False
        except Exception as e:
            print(f"  ✗ Batch size {batch_size} 失败: {e}")
            return False
    
    print(f"\n{'='*50}")
    print("✓ 所有测试通过！")
    print('='*50)
    return True


if __name__ == "__main__":
    success = test_network_shapes()
    sys.exit(0 if success else 1)
