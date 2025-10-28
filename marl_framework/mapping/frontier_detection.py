"""
前沿探测模块 - 用于静态搜索任务的内在奖励机制

核心思想:
- 前沿(Frontier): 已探索区域与未探索区域的边界
- 前沿驱动: 奖励智能体探索这些边界,逐步扩大已知区域
- 相比随机探索,前沿探索更高效,避免深入完全未知区域

实现方法:
1. 基于覆盖图检测前沿点
2. 计算智能体到最近前沿的距离奖励
3. 维护前沿图用于可视化和状态表示
"""

import numpy as np
from scipy.ndimage import binary_dilation, generic_filter
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FrontierDetector:
    """
    前沿检测器
    
    用于在覆盖图中检测已探索/未探索区域的边界
    """
    
    def __init__(
        self, 
        coverage_threshold: float = 0.3,
        kernel_size: int = 3,
        min_frontier_size: int = 1
    ):
        """
        Args:
            coverage_threshold: 判断已探索的覆盖率阈值 (0-1)
            kernel_size: 形态学操作的核大小
            min_frontier_size: 最小前沿点数量
        """
        self.coverage_threshold = coverage_threshold
        self.kernel_size = kernel_size
        self.min_frontier_size = min_frontier_size
        
        # 用于统计
        self.frontier_history = []
        
    def detect_frontiers(self, coverage_map: np.ndarray) -> np.ndarray:
        """
        检测覆盖图中的前沿点
        
        算法:
        1. 将覆盖图二值化为已探索/未探索
        2. 对已探索区域做膨胀操作
        3. 膨胀后与未探索区域的交集即为前沿
        
        Args:
            coverage_map: 覆盖图 (0-1之间的值), shape=(H, W)
        
        Returns:
            frontier_map: 前沿图 (0或1), shape=(H, W)
        """
        if coverage_map is None or coverage_map.size == 0:
            logger.warning("Empty coverage map provided to frontier detection")
            return np.zeros_like(coverage_map)
        
        # 1. 二值化
        explored = (coverage_map > self.coverage_threshold).astype(float)
        unexplored = (coverage_map <= self.coverage_threshold).astype(float)
        
        # 如果没有探索过的区域,返回空前沿
        if np.sum(explored) == 0:
            return np.zeros_like(coverage_map)
        
        # 2. 膨胀已探索区域
        struct = np.ones((self.kernel_size, self.kernel_size))
        explored_dilated = binary_dilation(explored, structure=struct)
        
        # 3. 计算前沿: 膨胀区域与未探索区域的交集
        frontier = explored_dilated.astype(float) * unexplored
        
        # 4. 过滤小前沿
        if np.sum(frontier) < self.min_frontier_size:
            return np.zeros_like(coverage_map)
        
        # 记录统计信息
        self.frontier_history.append(np.sum(frontier))
        
        return frontier
    
    def get_frontier_positions(self, frontier_map: np.ndarray) -> np.ndarray:
        """
        获取前沿点的坐标列表
        
        Args:
            frontier_map: 前沿图
        
        Returns:
            positions: 前沿点坐标数组, shape=(N, 2), 每行为[row, col]
        """
        return np.argwhere(frontier_map > 0.5)
    
    def get_statistics(self) -> Dict:
        """获取前沿检测统计信息"""
        if not self.frontier_history:
            return {
                'total_frontier_points': 0,
                'avg_frontier_points': 0,
                'max_frontier_points': 0
            }
        
        return {
            'total_frontier_points': int(np.sum(self.frontier_history)),
            'avg_frontier_points': float(np.mean(self.frontier_history)),
            'max_frontier_points': int(np.max(self.frontier_history))
        }


class FrontierRewardCalculator:
    """
    前沿奖励计算器
    
    基于智能体与最近前沿的距离计算内在奖励
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        decay_constant: float = 5.0,
        max_distance: float = 50.0
    ):
        """
        Args:
            reward_weight: 前沿奖励权重
            decay_constant: 指数衰减常数 (越大,衰减越慢)
            max_distance: 最大考虑距离 (超过此距离奖励为0)
        """
        self.reward_weight = reward_weight
        self.decay_constant = decay_constant
        self.max_distance = max_distance
        
        # 用于统计
        self.reward_history = []
        
    def calculate_reward(
        self,
        position: np.ndarray,
        frontier_map: np.ndarray,
        spacing: float = 1.0
    ) -> float:
        """
        计算智能体在当前位置的前沿奖励
        
        奖励公式: reward = weight * exp(-distance / decay_constant)
        
        Args:
            position: 智能体位置 [x, y]
            frontier_map: 前沿图
            spacing: 网格间距 (用于坐标转换)
        
        Returns:
            reward: 前沿奖励值
        """
        # 获取所有前沿点
        frontier_positions = np.argwhere(frontier_map > 0.5)
        
        if len(frontier_positions) == 0:
            # 无前沿点,返回0奖励
            return 0.0
        
        # 将智能体位置转换为网格索引
        pos_idx = self._position_to_index(position, spacing, frontier_map.shape)
        
        # 计算到所有前沿点的距离
        distances = np.linalg.norm(frontier_positions - pos_idx, axis=1)
        min_distance = np.min(distances)
        
        # 如果距离太远,返回0
        if min_distance > self.max_distance:
            reward = 0.0
        else:
            # 指数衰减奖励
            reward = self.reward_weight * np.exp(-min_distance / self.decay_constant)
        
        # 记录统计
        self.reward_history.append(reward)
        
        return reward
    
    def calculate_batch_rewards(
        self,
        positions: np.ndarray,
        frontier_map: np.ndarray,
        spacing: float = 1.0
    ) -> np.ndarray:
        """
        批量计算多个智能体的前沿奖励
        
        Args:
            positions: 智能体位置数组, shape=(N, 2)
            frontier_map: 前沿图
            spacing: 网格间距
        
        Returns:
            rewards: 奖励数组, shape=(N,)
        """
        rewards = np.zeros(len(positions))
        
        for i, pos in enumerate(positions):
            rewards[i] = self.calculate_reward(pos, frontier_map, spacing)
        
        return rewards
    
    def _position_to_index(
        self,
        position: np.ndarray,
        spacing: float,
        map_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        将实际坐标转换为网格索引
        
        Args:
            position: [x, y] 坐标
            spacing: 网格间距
            map_shape: 地图形状 (H, W)
        
        Returns:
            index: [row, col] 索引
        """
        # 假设地图原点在左下角,x向右,y向上
        col = int(position[0] / spacing)
        row = int(position[1] / spacing)
        
        # 边界检查
        row = np.clip(row, 0, map_shape[0] - 1)
        col = np.clip(col, 0, map_shape[1] - 1)
        
        return np.array([row, col])
    
    def get_statistics(self) -> Dict:
        """获取奖励统计信息"""
        if not self.reward_history:
            return {
                'total_frontier_reward': 0.0,
                'avg_frontier_reward': 0.0,
                'max_frontier_reward': 0.0
            }
        
        return {
            'total_frontier_reward': float(np.sum(self.reward_history)),
            'avg_frontier_reward': float(np.mean(self.reward_history)),
            'max_frontier_reward': float(np.max(self.reward_history))
        }


class FrontierManager:
    """
    前沿管理器 - 整合前沿检测和奖励计算
    
    用于COMA训练流程中的前沿探测功能
    """
    
    def __init__(self, params: dict):
        """
        Args:
            params: 配置参数字典
        """
        # 前沿检测参数
        frontier_threshold = params.get('experiment', {}).get(
            'intrinsic_rewards', {}
        ).get('frontier_detection_threshold', 0.3)
        
        frontier_kernel_size = params.get('state_representation', {}).get(
            'frontier_kernel_size', 3
        )
        
        # 前沿奖励参数
        frontier_weight = params.get('experiment', {}).get(
            'intrinsic_rewards', {}
        ).get('frontier_reward_weight', 1.0)
        
        # 初始化检测器和奖励计算器
        self.detector = FrontierDetector(
            coverage_threshold=frontier_threshold,
            kernel_size=frontier_kernel_size
        )
        
        self.reward_calculator = FrontierRewardCalculator(
            reward_weight=frontier_weight
        )
        
        # 缓存当前前沿图
        self.current_frontier_map = None
        
        # 启用标志
        self.enabled = params.get('experiment', {}).get(
            'intrinsic_rewards', {}
        ).get('enable', False)
        
        logger.info(f"FrontierManager initialized (enabled={self.enabled})")
        
    def update(self, coverage_map: np.ndarray):
        """
        更新前沿图
        
        Args:
            coverage_map: 当前覆盖图
        """
        if not self.enabled:
            return
        
        self.current_frontier_map = self.detector.detect_frontiers(coverage_map)
    
    def get_frontier_map(self) -> Optional[np.ndarray]:
        """获取当前前沿图"""
        return self.current_frontier_map
    
    def calculate_frontier_reward(
        self,
        position: np.ndarray,
        spacing: float = 1.0
    ) -> float:
        """
        计算智能体在当前位置的前沿奖励
        
        Args:
            position: 智能体位置 [x, y]
            spacing: 网格间距
        
        Returns:
            reward: 前沿奖励
        """
        if not self.enabled or self.current_frontier_map is None:
            return 0.0
        
        return self.reward_calculator.calculate_reward(
            position,
            self.current_frontier_map,
            spacing
        )
    
    def calculate_batch_frontier_rewards(
        self,
        positions: np.ndarray,
        spacing: float = 1.0
    ) -> np.ndarray:
        """
        批量计算多个智能体的前沿奖励
        
        Args:
            positions: 智能体位置数组, shape=(N, 2)
            spacing: 网格间距
        
        Returns:
            rewards: 奖励数组, shape=(N,)
        """
        if not self.enabled or self.current_frontier_map is None:
            return np.zeros(len(positions))
        
        return self.reward_calculator.calculate_batch_rewards(
            positions,
            self.current_frontier_map,
            spacing
        )
    
    def get_statistics(self) -> Dict:
        """获取前沿相关统计信息"""
        stats = {
            'frontier_detection': self.detector.get_statistics(),
            'frontier_reward': self.reward_calculator.get_statistics()
        }
        
        # 添加当前前沿点数量
        if self.current_frontier_map is not None:
            stats['current_frontier_points'] = int(np.sum(self.current_frontier_map))
        else:
            stats['current_frontier_points'] = 0
        
        return stats
    
    def reset(self):
        """重置前沿管理器"""
        self.current_frontier_map = None
        self.detector.frontier_history = []
        self.reward_calculator.reward_history = []


# ==================== 工具函数 ====================

def visualize_frontier(
    coverage_map: np.ndarray,
    frontier_map: np.ndarray,
    agent_positions: Optional[np.ndarray] = None,
    title: str = "Frontier Detection"
):
    """
    可视化前沿检测结果
    
    Args:
        coverage_map: 覆盖图
        frontier_map: 前沿图
        agent_positions: 智能体位置 (可选)
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 覆盖图
    im0 = axes[0].imshow(coverage_map, cmap='Greys', origin='lower')
    axes[0].set_title('Coverage Map')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0], label='Coverage')
    
    # 前沿图
    im1 = axes[1].imshow(frontier_map, cmap='Reds', origin='lower')
    axes[1].set_title('Frontier Map')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1], label='Frontier')
    
    # 叠加显示
    axes[2].imshow(coverage_map, cmap='Greys', alpha=0.5, origin='lower')
    axes[2].imshow(frontier_map, cmap='Reds', alpha=0.5, origin='lower')
    
    # 绘制智能体位置
    if agent_positions is not None:
        axes[2].scatter(
            agent_positions[:, 0],
            agent_positions[:, 1],
            c='blue',
            marker='o',
            s=100,
            label='Agents'
        )
        axes[2].legend()
    
    axes[2].set_title('Coverage + Frontier Overlay')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def test_frontier_detection():
    """测试前沿检测功能"""
    print("Testing Frontier Detection...")
    
    # 创建测试覆盖图
    coverage_map = np.zeros((50, 50))
    
    # 模拟已探索区域 (中心圆形)
    center = np.array([25, 25])
    for i in range(50):
        for j in range(50):
            distance = np.linalg.norm(np.array([i, j]) - center)
            if distance < 15:
                coverage_map[i, j] = 0.8
            elif distance < 18:
                coverage_map[i, j] = 0.2
    
    # 创建检测器
    detector = FrontierDetector(coverage_threshold=0.3, kernel_size=3)
    frontier_map = detector.detect_frontiers(coverage_map)
    
    print(f"Coverage points: {np.sum(coverage_map > 0.3)}")
    print(f"Frontier points: {np.sum(frontier_map)}")
    print(f"Statistics: {detector.get_statistics()}")
    
    # 测试奖励计算
    reward_calc = FrontierRewardCalculator(reward_weight=1.0, decay_constant=5.0)
    
    # 测试不同位置的奖励
    test_positions = [
        np.array([25, 25]),  # 中心 (远离前沿)
        np.array([40, 25]),  # 靠近前沿
        np.array([10, 10])   # 远离前沿
    ]
    
    for pos in test_positions:
        reward = reward_calc.calculate_reward(pos, frontier_map, spacing=1.0)
        print(f"Position {pos}: reward = {reward:.4f}")
    
    # 可视化
    visualize_frontier(coverage_map, frontier_map, np.array(test_positions))
    
    print("Test completed!")


if __name__ == "__main__":
    test_frontier_detection()
