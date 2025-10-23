"""
GPU测试脚本 - 验证PyTorch CUDA配置和训练性能
"""
import torch
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'marl_framework'))

def test_cuda_availability():
    """测试CUDA是否可用"""
    print("=" * 60)
    print("CUDA可用性测试")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数量: {props.multi_processor_count}")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练（速度会很慢）")
    print()

def test_tensor_operations():
    """测试GPU张量操作"""
    print("=" * 60)
    print("GPU张量操作测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("跳过GPU测试（CUDA不可用）")
        return
    
    device = torch.device("cuda:0")
    
    # 测试张量创建和移动
    print("创建测试张量...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    print(f"张量设备: {x.device}")
    print(f"张量形状: {x.shape}")
    
    # 测试矩阵乘法性能
    print("\n测试矩阵乘法性能...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU耗时: {gpu_time:.4f}秒")
    
    # CPU对比
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start = time.time()
    for _ in range(100):
        z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU耗时: {cpu_time:.4f}秒")
    print(f"GPU加速比: {cpu_time/gpu_time:.2f}x")
    print()

def test_cnn_forward():
    """测试CNN前向传播"""
    print("=" * 60)
    print("CNN网络测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("跳过GPU测试（CUDA不可用）")
        return
    
    device = torch.device("cuda:0")
    
    # 创建简单的CNN
    model = torch.nn.Sequential(
        torch.nn.Conv2d(7, 256, 5),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, 4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, 4),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 6)
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 128
    input_tensor = torch.randn(batch_size, 7, 50, 50).to(device)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输入设备: {input_tensor.device}")
    
    # 预热
    for _ in range(10):
        _ = model(input_tensor)
    
    # 测试性能
    torch.cuda.synchronize()
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        output = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"输出形状: {output.shape}")
    print(f"前向传播{iterations}次耗时: {elapsed:.4f}秒")
    print(f"平均每次: {elapsed/iterations*1000:.2f}ms")
    print(f"吞吐量: {batch_size*iterations/elapsed:.1f} samples/sec")
    
    # 内存使用
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"\nGPU内存使用:")
    print(f"  已分配: {memory_allocated:.2f} MB")
    print(f"  已保留: {memory_reserved:.2f} MB")
    print()

def test_project_imports():
    """测试项目导入和网络初始化"""
    print("=" * 60)
    print("项目网络导入测试")
    print("=" * 60)
    
    try:
        from params import load_params
        import constants
        from actor.network import ActorNetwork
        from critic.network import CriticNetwork
        
        print("✓ 导入成功")
        
        # 加载配置
        params = load_params(constants.CONFIG_FILE_PATH)
        print(f"✓ 配置加载成功")
        print(f"  设备设置: {params['networks']['device']}")
        print(f"  批次大小: {params['networks']['batch_size']}")
        print(f"  智能体数量: {params['experiment']['missions']['n_agents']}")
        
        if torch.cuda.is_available() and params["networks"]["device"] == "cuda":
            device = torch.device("cuda:0")
            
            # 测试Actor网络
            print("\n测试Actor网络...")
            actor = ActorNetwork(params).to(device)
            print(f"✓ Actor网络创建成功")
            print(f"  参数量: {sum(p.numel() for p in actor.parameters()):,}")
            
            # 测试Critic网络
            print("\n测试Critic网络...")
            critic = CriticNetwork(params).to(device)
            print(f"✓ Critic网络创建成功")
            print(f"  参数量: {sum(p.numel() for p in critic.parameters()):,}")
            
            print("\n✓ 所有网络已成功移动到GPU")
        else:
            print("\n⚠️ 未配置GPU或CUDA不可用")
            
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
    print()

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "GPU训练环境测试" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    test_cuda_availability()
    test_tensor_operations()
    test_cnn_forward()
    test_project_imports()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✓ GPU已准备就绪，可以开始训练")
        print("\n建议的训练命令:")
        print("  cd marl_framework")
        print("  python main.py")
    else:
        print("⚠️ GPU不可用，训练将使用CPU（速度会很慢）")
    print()

if __name__ == "__main__":
    main()
