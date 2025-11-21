#!/usr/bin/env python3
"""
训练进度监控脚本
用于检查训练状态和模型保存情况
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime

def print_header(text):
    print("\n" + "="*50)
    print(f"📊 {text}")
    print("="*50 + "\n")

def print_section(text):
    print(f"\n{text}\n")

def main():
    print_header("训练状态监控")
    
    log_dir = Path("marl_framework/log")
    res_dir = Path("marl_framework/res")
    
    # 检查训练进度
    progress_file = res_dir / "training_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        
        print_section("📊 训练进度:")
        print(f"  当前步数: {progress['current_training_step']} / {progress['total_training_steps']}")
        print(f"  完成度: {progress['progress_percentage']}%")
        print(f"  当前回报: {progress['current_max_return']}")
        print(f"  状态: {progress['training_status']}")
    else:
        print_section("⚠️  未找到训练进度文件")
    
    # 检查模型文件
    print_section("📦 已保存的模型:")
    if log_dir.exists():
        models = list(log_dir.glob("*.pth"))
        if models:
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  - {model.name} ({size_mb:.2f} MB)")
        else:
            print("  未找到保存的模型文件")
    else:
        print("  日志目录不存在")
    
    # 检查训练历史
    history_file = res_dir / "training_history.csv"
    if history_file.exists():
        print_section("📈 训练历史:")
        with open(history_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            history = list(reader)
        
        print(f"  总episodes: {len(history)}")
        
        if len(history) >= 10:
            recent_returns = [float(row['episode_return']) for row in history[-10:]]
            recent_avg = sum(recent_returns) / len(recent_returns)
            print(f"  最近10个episode平均回报: {recent_avg:.2f}")
            
            # 检查改进趋势
            if len(history) >= 20:
                first_10 = [float(row['episode_return']) for row in history[:10]]
                last_10 = recent_returns
                first_avg = sum(first_10) / len(first_10)
                last_avg = recent_avg
                improvement = last_avg - first_avg
                
                if improvement > 0:
                    print(f"  ✅ 训练有改进! 提升: {improvement:.2f}")
                else:
                    print(f"  ⚠️  回报未见明显改进: {improvement:.2f}")
    else:
        print_section("⚠️  未找到训练历史文件")
    
    # 探索率配置验证
    print_section("🔍 探索率配置验证:")
    config_file = Path("marl_framework/configs/params.yaml")
    if config_file.exists():
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        eps_max = config['experiment']['missions']['eps_max']
        eps_min = config['experiment']['missions']['eps_min']
        eps_anneal = config['experiment']['missions']['eps_anneal_phase']
        
        print(f"  eps_max: {eps_max} {'✓' if eps_max <= 0.7 else '⚠️'}")
        print(f"  eps_min: {eps_min} {'✓' if eps_min <= 0.1 else '⚠️'}")
        print(f"  eps_anneal_phase: {eps_anneal} {'✓' if eps_anneal <= 1000 else '⚠️'}")
        
        if eps_anneal > 5000:
            print("\n  ⚠️  警告: eps_anneal_phase过大! 建议设置为100-200")
    
    # TensorBoard日志
    tensorboard_dir = log_dir / "tensorboard" if log_dir.exists() else None
    if tensorboard_dir and tensorboard_dir.exists():
        print_section("📊 TensorBoard日志:")
        tb_files = list(tensorboard_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in tb_files if f.is_file())
        print(f"  日志文件数: {len(tb_files)}")
        print(f"  总大小: {total_size / (1024*1024):.2f} MB")
        print(f"\n  查看训练曲线: tensorboard --logdir={tensorboard_dir}")
    
    # 使用建议
    print_header("💡 使用建议")
    print()
    print("1. 持续监控: 每隔30分钟运行此脚本检查进度")
    print("2. 查看详细日志: tail -f marl_framework/log/training.log")
    print("3. 启动TensorBoard: tensorboard --logdir=marl_framework/log")
    print("4. 对比轨迹: 在100、300、500步时保存并对比航线图")
    print()
    
    # 诊断建议
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        
        if progress['current_training_step'] > 100:
            if progress['current_max_return'] < 10:
                print("⚠️  训练可能存在问题:")
                print("   - 回报值过低")
                print("   - 检查奖励函数设置")
                print("   - 检查探索率是否过高")
            elif progress['progress_percentage'] > 50 and progress['current_max_return'] < 50:
                print("⚠️  训练进度过半但回报仍较低")
                print("   - 可能需要调整学习率")
                print("   - 可能需要更长的训练时间")
            else:
                print("✅ 训练看起来正常!")
    
    print("\n" + "="*50)
    print("脚本执行完成")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
