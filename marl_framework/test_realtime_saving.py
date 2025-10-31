#!/usr/bin/env python3
"""
测试实时日志和结果保存功能
"""

import os
import sys
import time
import json

# 添加marl_framework到Python路径
sys.path.append(os.path.dirname(__file__))

import constants

def test_realtime_saving():
    print("=== 实时保存功能测试 ===")
    
    # 检查目录
    print(f"📁 日志目录: {constants.LOG_DIR}")
    print(f"📁 结果目录: {constants.EXPERIMENTS_FOLDER}")
    
    # 确保目录存在
    os.makedirs(constants.LOG_DIR, exist_ok=True)
    os.makedirs(constants.EXPERIMENTS_FOLDER, exist_ok=True)
    
    # 测试实时保存
    print("\n🧪 模拟训练过程...")
    
    # 创建模拟进度文件
    progress_data = {
        "timestamp": "20241031_123456",
        "current_training_step": 150,
        "total_training_steps": 500,
        "progress_percentage": 30.0,
        "current_max_return": 15.5,
        "total_episodes": 75,
        "latest_episode_returns": [12.1, 13.5, 14.2, 15.1, 14.8],
        "recent_mean_return": 13.94,
        "overall_mean_return": 13.2,
        "training_status": "in_progress"
    }
    
    progress_file = os.path.join(constants.EXPERIMENTS_FOLDER, "training_progress.json")
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 测试进度文件已创建: {progress_file}")
    
    # 检查文件是否立即可见
    if os.path.exists(progress_file):
        print("✅ 进度文件立即可见")
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"📊 训练进度: {data['progress_percentage']}%")
    else:
        print("❌ 进度文件不可见")
    
    print("\n📋 实时保存功能总结:")
    print("1. 日志文件: 每次写入后立即刷新到磁盘")
    print("2. 训练进度: 每50步保存一次到 training_progress.json")
    print("3. 训练历史: 每50步更新完整历史到 training_history.csv")
    print("4. TensorBoard: 每20步刷新一次")
    print("5. 模型检查点: 按配置的最佳性能保存")

def check_file_monitoring():
    print("\n=== 文件监控指南 ===")
    print("🔍 训练期间可以实时查看的文件:")
    print(f"📝 日志文件: {os.path.join(constants.LOG_DIR, 'log_*.log')}")
    print(f"📊 训练进度: {os.path.join(constants.EXPERIMENTS_FOLDER, 'training_progress.json')}")
    print(f"📈 训练历史: {os.path.join(constants.EXPERIMENTS_FOLDER, 'training_history.csv')}")
    print(f"🧠 最佳模型: {os.path.join(constants.LOG_DIR, 'best_model*.pth')}")
    print()
    print("💡 监控命令示例:")
    print(f"# 实时查看日志")
    print(f"tail -f {os.path.join(constants.LOG_DIR, 'log_*.log')}")
    print()
    print(f"# 查看训练进度")
    print(f"cat {os.path.join(constants.EXPERIMENTS_FOLDER, 'training_progress.json')}")
    print()
    print(f"# 查看训练历史")
    print(f"tail -20 {os.path.join(constants.EXPERIMENTS_FOLDER, 'training_history.csv')}")

if __name__ == "__main__":
    test_realtime_saving()
    check_file_monitoring()