#!/usr/bin/env python3
"""
探索率诊断脚本
用于理解episode和training step的关系，以及探索率的实际值
"""

import os
import sys
import yaml
import csv
from pathlib import Path

def print_header(text):
    print("\n" + "="*50)
    print(f"🔍 {text}")
    print("="*50 + "\n")

def print_section(text):
    print(f"\n📊 {text}\n")

def main(current_training_step=300, current_episode=750):
    print_header("探索率诊断分析")
    
    # 读取配置
    config_file = Path("marl_framework/configs/params.yaml")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        eps_max = config['experiment']['missions']['eps_max']
        eps_min = config['experiment']['missions']['eps_min']
        eps_anneal = config['experiment']['missions']['eps_anneal_phase']
        
        print_section("当前配置")
        print(f"  eps_max: {eps_max}")
        print(f"  eps_min: {eps_min}")
        print(f"  eps_anneal_phase: {eps_anneal} episodes")
        
        # 计算关系
        print_section("Training Step vs Episode 关系")
        print(f"  Budget (时间步): 50")
        print(f"  智能体数量: 4")
        print(f"  每episode产生样本: 51 × 4 = 204")
        print(f"  Batch size: 128")
        print(f"  Batch number: 4")
        print(f"  每次训练需要样本: 128 × 4 = 512")
        print(f"  每次训练需要episodes: 512 / 204 ≈ 2.5")
        print()
        print("  → 所以1个training step ≈ 2.5个episodes")
        print("  → 100个training steps ≈ 250个episodes")
        print("  → 300个training steps ≈ 750个episodes")
        
        # 探索率变化
        print_section("探索率随时间变化")
        print()
        print(f"  {'Episode数':<10} | {'Training Step':<15} | {'探索率(eps)':<12}")
        print(f"  {'-'*10}|{'-'*16}|{'-'*12}")
        
        milestones = [1, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
        for ep in milestones:
            step = int(ep / 2.5)
            
            # 计算探索率
            if ep > eps_anneal:
                eps = eps_min
            else:
                eps = eps_max - (ep / eps_anneal) * (eps_max - eps_min)
            
            eps = max(eps, eps_min)
            
            highlight = " ← 当前位置" if ep == current_episode else ""
            
            # 颜色标记
            if eps < 0.1:
                color = "\033[92m"  # Green
            elif eps < 0.2:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            reset = "\033[0m"
            
            print(f"  {ep:<10} | {step:<15} | {color}{eps:<12.4f}{reset}{highlight}")
        
        # 解读
        print_section("解读")
        current_eps = eps_min if current_episode > eps_anneal else \
                     max(eps_max - (current_episode / eps_anneal) * (eps_max - eps_min), eps_min)
        
        print(f"  在第 {current_episode} 个episode (≈ 第 {current_training_step} 个training step):")
        print(f"  探索率 = {current_eps:.4f}")
        print()
        
        if current_eps > 0.15:
            print("  ⚠️  探索率过高 (>15%)!")
            print("  → 模型仍在大量随机探索，学习的策略未被充分使用")
            print("  → 轨迹看起来会很随机，与初期没有明显区别")
            print()
            print("  建议:")
            print("  1. 降低 eps_max 到 0.2-0.3")
            print("  2. 降低 eps_anneal_phase 到 50-100")
        elif current_eps > 0.08:
            print("  ⚠️  探索率偏高 (8-15%)")
            print("  → 模型正在从探索转向利用，但可能还不够快")
            print("  → 轨迹应该开始出现一些规律，但仍有随机性")
        else:
            print("  ✅ 探索率正常 (<8%)")
            print("  → 模型主要使用学习的策略，只有少量探索")
            print("  → 轨迹应该表现出明确的学习行为")
        
        # 修复建议
        print_section("修复建议")
        print()
        print("  当前配置 (修复后):")
        print(f"    eps_max: {eps_max}           ← 降低初始探索")
        print(f"    eps_min: {eps_min}          ← 保持最小探索")
        print(f"    eps_anneal_phase: {eps_anneal}  ← 在{eps_anneal}个episodes内完成衰减")
        print()
        print("  效果预测:")
        steps_to_min = int(eps_anneal / 2.5)
        print(f"    • 0-{steps_to_min} steps (0-{eps_anneal} episodes): 探索率从{eps_max*100:.0f}%降到{eps_min*100:.0f}%")
        print(f"    • {steps_to_min}+ steps ({eps_anneal}+ episodes): 保持{eps_min*100:.0f}%探索率")
        print(f"    • 300 steps时: 探索率应该是{eps_min*100:.0f}%")
        print()
        print("  如果还没效果，尝试:")
        print("    eps_max: 0.2           ← 更低的初始探索")
        print("    eps_min: 0.0           ← 完全不探索")
        print("    eps_anneal_phase: 50   ← 更快衰减")
    
    # 检查学习效果
    print_header("学习效果检查")
    
    history_file = Path("marl_framework/res/training_history.csv")
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            history = list(reader)
        
        if len(history) >= 20:
            first_10 = [float(row['episode_return']) for row in history[:10]]
            last_10 = [float(row['episode_return']) for row in history[-10:]]
            
            first_avg = sum(first_10) / len(first_10)
            last_avg = sum(last_10) / len(last_10)
            improvement = last_avg - first_avg
            improvement_pct = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
            
            print(f"  前10个episode平均回报: {first_avg:.2f}")
            print(f"  最近10个episode平均回报: {last_avg:.2f}")
            print(f"  改进: {improvement:.2f} ({improvement_pct:.1f}%)")
            print()
            
            if improvement > 0 and improvement_pct > 10:
                print("  ✅ 模型正在学习! 回报有明显改善")
            elif improvement > 0:
                print("  ⚠️  有改善但不明显，可能需要:")
                print("     • 提高学习率")
                print("     • 降低探索率")
                print("     • 检查奖励函数设计")
            else:
                print("  ❌ 回报未改善，可能问题:")
                print("     • 探索率过高，策略未被使用")
                print("     • 学习率过低，更新太慢")
                print("     • 奖励信号不清晰")
        else:
            print("  ⚠️  数据不足，无法评估学习效果")
    else:
        print("  ⚠️  未找到训练历史文件")
    
    print("\n" + "="*50)
    print("脚本执行完成")
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='诊断探索率设置')
    parser.add_argument('--step', type=int, default=300, help='当前training step')
    parser.add_argument('--episode', type=int, default=750, help='当前episode数')
    args = parser.parse_args()
    
    main(args.step, args.episode)
