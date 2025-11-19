import matplotlib.pyplot as plt
import numpy as np

# 示例数据：五种算法的累计奖励曲线（可替换为实际数据）
episodes = np.arange(1, 1501)
rewards_hcs_coma = np.clip(500 - 500 * np.exp(-episodes/300) + np.random.normal(0, 15, 1500), 350, 520)
rewards_coma = np.clip(420 - 420 * np.exp(-episodes/600) + np.random.normal(0, 25, 1500), 250, 440)
rewards_qmix = np.clip(390 - 390 * np.exp(-episodes/400) + np.random.normal(0, 20, 1500), 200, 410)
rewards_lawn = np.full(1500, 180)
rewards_random = np.full(1500, -120)

plt.figure(figsize=(8,5))
plt.plot(episodes, rewards_hcs_coma, label='HCS-COMA', color='red', linewidth=2)
plt.plot(episodes, rewards_coma, label='Standard COMA', color='blue', linestyle='--')
plt.plot(episodes, rewards_qmix, label='QMIX', color='green', linestyle='-.')
plt.plot(episodes, rewards_lawn, label='Lawn Mower', color='orange', linestyle=':')
plt.plot(episodes, rewards_random, label='Random', color='gray', linestyle=':')
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Reward')
plt.title('Convergence Comparison of Algorithms')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('e:/code/paper_code/paper/demo/convergence_curve.png', dpi=300)
plt.show()
