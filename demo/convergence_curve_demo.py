import matplotlib.pyplot as plt
import matplotlib
import numpy as np
episodes = np.arange(1, 1501)
np.random.seed(42)
# HCS-COMA：收敛快，最终奖励高，波动小
rewards_hcs_coma = 120 + 380 * (1 - np.exp(-episodes/350)) + np.random.normal(0, 6, 1500)
# Standard COMA：收敛慢，最终奖励略低，波动略大
rewards_coma = 100 + 320 * (1 - np.exp(-episodes/700)) + np.random.normal(0, 10, 1500)
# QMIX：中等收敛速度，最终奖励中等，波动适中
rewards_qmix = 90 + 280 * (1 - np.exp(-episodes/500)) + np.random.normal(0, 8, 1500)
# Lawn Mower：确定性，奖励稳定但低
rewards_lawn = np.full(1500, 160)
# Random：奖励低且波动大
rewards_random = 60 + np.random.normal(0, 15, 1500)

# 滑动平均函数
def smooth(data, window=30):
    return np.convolve(data, np.ones(window)/window, mode='same')

plt.figure(figsize=(9,5))
plt.plot(episodes, smooth(rewards_hcs_coma), label='HCS-COMA', color='#D7263D', linewidth=2)
plt.plot(episodes, smooth(rewards_coma), label='Standard COMA', color='#1B98E0', linewidth=2)
plt.plot(episodes, smooth(rewards_qmix), label='QMIX', color='#2E933C', linewidth=2)
plt.plot(episodes, rewards_lawn, label='Lawn Mower', color='#F7B32B', linewidth=2)
plt.plot(episodes, rewards_random, label='Random', color='#6C6C6C', linewidth=2)

plt.xlabel('Training Episodes', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.title('Convergence Curve of Cumulative Reward', fontsize=15)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 550)
plt.tight_layout()
plt.savefig('./demo_png/convergence_curve.png', dpi=300)
plt.show()
"""
Reward Variance Bar Chart (Last 200 Episodes)
"""
alg_names = ['HCS-COMA', 'Standard COMA', 'QMIX', 'Lawn Mower', 'Random']
alg_rewards = [rewards_hcs_coma, rewards_coma, rewards_qmix, rewards_lawn, rewards_random]
var_list = [np.var(r[-200:]) for r in alg_rewards]

plt.figure(figsize=(6,4))
bars = plt.bar(alg_names, var_list, color=['#D7263D','#1B98E0','#2E933C','#F7B32B','#6C6C6C'], width=0.5)
plt.ylabel('Reward Variance', fontsize=12)
plt.title('Reward Variance Comparison (Last 200 Episodes)', fontsize=13)
plt.xticks(fontsize=9)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('./demo_png/reward_variance_bar.png', dpi=300)
plt.show()
