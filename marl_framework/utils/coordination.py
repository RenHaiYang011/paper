"""
协同机制模块 - 用于多智能体协同搜索的信用分配和反重叠

核心思想:
1. 抗重叠惩罚: 避免多个智能体搜索相同区域,浪费资源
2. 区域分工奖励: 鼓励智能体分散到不同区域,提高覆盖效率
3. 协同发现奖励: 奖励多个智能体在高优先级区域的协同搜索

实现方法:
- 路径预测: 基于历史轨迹预测未来路径
- 重叠度量: 计算观测区域、路径的重叠程度
- 分工度量: 评估智能体在不同区域的分布均匀性
- 协同检测: 识别多智能体协同搜索行为
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PathOverlapDetector:
    """
    路径重叠检测器
    
    检测多个智能体的观测区域或路径重叠,用于计算抗重叠惩罚
    """
    
    def __init__(
        self,
        overlap_threshold: float = 0.3,
        history_length: int = 5
    ):
        """
        Args:
            overlap_threshold: 重叠度阈值 (0-1)
            history_length: 历史位置记录长度
        """
        self.overlap_threshold = overlap_threshold
        self.history_length = history_length
        
        # 记录每个智能体的历史位置
        self.position_history: Dict[int, List[np.ndarray]] = {}
        
    def update_position(self, agent_id: int, position: np.ndarray):
        """更新智能体位置历史"""
        if agent_id not in self.position_history:
            self.position_history[agent_id] = []
        
        self.position_history[agent_id].append(position.copy())
        
        # 保持历史长度限制
        if len(self.position_history[agent_id]) > self.history_length:
            self.position_history[agent_id].pop(0)
    
    def calculate_observation_overlap(
        self,
        agent_positions: List[np.ndarray],
        sensor_range: float
    ) -> float:
        """
        计算智能体观测区域的重叠度
        
        Args:
            agent_positions: 所有智能体的当前位置列表
            sensor_range: 传感器观测范围
        
        Returns:
            overlap_ratio: 重叠比例 (0-1)
        """
        if len(agent_positions) < 2:
            return 0.0
        
        overlaps = []
        
        # 两两计算重叠
        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                pos1 = np.array(agent_positions[i][:2])  # 只考虑x,y
                pos2 = np.array(agent_positions[j][:2])
                
                distance = np.linalg.norm(pos1 - pos2)
                
                # 如果距离小于2倍传感器范围,有重叠
                if distance < 2 * sensor_range:
                    # 计算重叠面积比例
                    if distance == 0:
                        overlap = 1.0
                    else:
                        # 简化计算: 使用距离比例估算重叠
                        overlap = 1.0 - (distance / (2 * sensor_range))
                    overlaps.append(overlap)
                else:
                    overlaps.append(0.0)
        
        if not overlaps:
            return 0.0
        
        return float(np.mean(overlaps))
    
    def calculate_path_overlap(
        self,
        agent_id: int,
        other_agent_ids: List[int],
        spacing: float = 5.0
    ) -> float:
        """
        计算智能体路径与其他智能体的重叠度
        
        Args:
            agent_id: 当前智能体ID
            other_agent_ids: 其他智能体ID列表
            spacing: 网格间距
        
        Returns:
            overlap_ratio: 路径重叠比例 (0-1)
        """
        if agent_id not in self.position_history:
            return 0.0
        
        agent_history = self.position_history[agent_id]
        if len(agent_history) < 2:
            return 0.0
        
        overlaps = []
        
        for other_id in other_agent_ids:
            if other_id not in self.position_history:
                continue
            
            other_history = self.position_history[other_id]
            if len(other_history) < 2:
                continue
            
            # 计算两条路径的最小距离
            min_distance = float('inf')
            for pos1 in agent_history:
                for pos2 in other_history:
                    dist = np.linalg.norm(pos1[:2] - pos2[:2])
                    min_distance = min(min_distance, dist)
            
            # 如果距离小于2个网格间距,认为有重叠
            if min_distance < 2 * spacing:
                overlap = 1.0 - (min_distance / (2 * spacing))
                overlaps.append(overlap)
            else:
                overlaps.append(0.0)
        
        if not overlaps:
            return 0.0
        
        return float(np.mean(overlaps))
    
    def reset(self):
        """重置历史记录"""
        self.position_history = {}


class DivisionOfLaborMetric:
    """
    区域分工度量
    
    评估智能体在不同区域的分布均匀性,鼓励分工协作
    """
    
    def __init__(self, num_agents: int):
        """
        Args:
            num_agents: 智能体数量
        """
        self.num_agents = num_agents
        
    def calculate_division_score(
        self,
        agent_positions: List[np.ndarray],
        regions: Optional[List] = None
    ) -> float:
        """
        计算区域分工得分
        
        完美分工 (每个智能体在不同区域) = 1.0
        完全重叠 (所有智能体在同一区域) = 0.0
        
        Args:
            agent_positions: 智能体位置列表
            regions: 搜索区域列表 (可选)
        
        Returns:
            division_score: 分工得分 (0-1)
        """
        if len(agent_positions) < 2:
            return 1.0
        
        if regions is not None:
            # 基于区域的分工度量
            return self._calculate_region_based_division(agent_positions, regions)
        else:
            # 基于位置分布的分工度量
            return self._calculate_position_based_division(agent_positions)
    
    def _calculate_region_based_division(
        self,
        agent_positions: List[np.ndarray],
        regions: List
    ) -> float:
        """基于区域的分工度量"""
        # 统计每个区域的智能体数量
        region_counts = {}
        
        for pos in agent_positions:
            # 找到当前位置所属的区域
            region_found = None
            for region in regions:
                if self._is_in_region(pos, region):
                    region_found = region.name
                    break
            
            if region_found:
                region_counts[region_found] = region_counts.get(region_found, 0) + 1
        
        if not region_counts:
            return 0.0
        
        # 计算分布的均匀性 (使用标准差)
        counts = list(region_counts.values())
        
        # 理想情况: 每个区域1个智能体
        # 使用变异系数 (CV) 衡量分布均匀性
        mean_count = np.mean(counts)
        if mean_count == 0:
            return 0.0
        
        std_count = np.std(counts)
        cv = std_count / mean_count
        
        # 归一化到0-1, CV越小越好
        # 最大CV = sqrt(n-1), 当所有智能体在一个区域时
        max_cv = np.sqrt(len(agent_positions) - 1)
        division_score = 1.0 - min(cv / max_cv, 1.0)
        
        return division_score
    
    def _calculate_position_based_division(
        self,
        agent_positions: List[np.ndarray]
    ) -> float:
        """基于位置分布的分工度量"""
        positions = np.array([pos[:2] for pos in agent_positions])
        
        # 计算智能体之间的平均距离
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        
        # 计算地图对角线长度作为归一化因子
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        diagonal = np.linalg.norm(max_coords - min_coords)
        
        if diagonal == 0:
            return 0.0
        
        # 归一化距离
        normalized_distance = min(avg_distance / diagonal, 1.0)
        
        return normalized_distance
    
    def _is_in_region(self, position: np.ndarray, region) -> bool:
        """检查位置是否在区域内"""
        x, y = position[0], position[1]
        
        if region.type == "rectangle":
            x_min, y_min, x_max, y_max = region.coordinates[0]
            return x_min <= x <= x_max and y_min <= y <= y_max
        
        return False


class CollaborationDetector:
    """
    协同发现检测器
    
    识别多个智能体在高优先级区域的协同搜索行为
    """
    
    def __init__(
        self,
        collaboration_distance: float = 15.0,
        min_agents: int = 2
    ):
        """
        Args:
            collaboration_distance: 协同距离阈值
            min_agents: 最小协同智能体数量
        """
        self.collaboration_distance = collaboration_distance
        self.min_agents = min_agents
        
    def detect_collaboration(
        self,
        agent_positions: List[np.ndarray],
        regions: Optional[List] = None
    ) -> Tuple[bool, float]:
        """
        检测是否存在协同搜索行为
        
        Args:
            agent_positions: 智能体位置列表
            regions: 搜索区域列表 (可选)
        
        Returns:
            is_collaborating: 是否协同
            collaboration_score: 协同得分 (0-1)
        """
        if len(agent_positions) < self.min_agents:
            return False, 0.0
        
        # 计算智能体之间的距离矩阵
        positions = np.array([pos[:2] for pos in agent_positions])
        n = len(positions)
        
        # 找到距离小于阈值的智能体对
        close_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= self.collaboration_distance:
                    close_pairs.append((i, j, dist))
        
        if not close_pairs:
            return False, 0.0
        
        # 如果提供了区域信息,检查是否在高优先级区域
        if regions is not None:
            high_priority_collab = self._check_high_priority_collaboration(
                agent_positions, close_pairs, regions
            )
            
            if high_priority_collab:
                # 计算协同得分 (基于距离的接近程度)
                avg_distance = np.mean([dist for _, _, dist in close_pairs])
                score = 1.0 - (avg_distance / self.collaboration_distance)
                return True, score
        else:
            # 没有区域信息,仅基于距离
            avg_distance = np.mean([dist for _, _, dist in close_pairs])
            score = 1.0 - (avg_distance / self.collaboration_distance)
            return True, score
        
        return False, 0.0
    
    def _check_high_priority_collaboration(
        self,
        agent_positions: List[np.ndarray],
        close_pairs: List[Tuple[int, int, float]],
        regions: List
    ) -> bool:
        """检查协同是否发生在高优先级区域"""
        for i, j, _ in close_pairs:
            pos1 = agent_positions[i]
            pos2 = agent_positions[j]
            
            # 检查两个智能体是否都在高优先级区域
            region1 = self._find_region(pos1, regions)
            region2 = self._find_region(pos2, regions)
            
            if region1 and region2:
                # 如果在同一个高优先级区域 (priority > 0.7)
                if region1.name == region2.name and region1.priority > 0.7:
                    return True
        
        return False
    
    def _find_region(self, position: np.ndarray, regions: List):
        """找到位置所属的区域"""
        x, y = position[0], position[1]
        
        for region in regions:
            if region.type == "rectangle":
                x_min, y_min, x_max, y_max = region.coordinates[0]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return region
        
        return None


class CoordinationManager:
    """
    协同管理器 - 整合所有协同机制
    
    统一接口,用于计算协同相关的奖励和惩罚
    """
    
    def __init__(self, params: dict, num_agents: int):
        """
        Args:
            params: 配置参数
            num_agents: 智能体数量
        """
        self.params = params
        self.num_agents = num_agents
        
        # 从配置中读取参数
        coord_config = params.get("experiment", {}).get("coordination", {})
        self.enabled = coord_config.get("enable", False)
        
        # 初始化各个组件
        self.overlap_detector = PathOverlapDetector(
            overlap_threshold=coord_config.get("overlap_threshold", 0.3)
        )
        
        self.division_metric = DivisionOfLaborMetric(num_agents)
        
        self.collaboration_detector = CollaborationDetector(
            collaboration_distance=coord_config.get("collaboration_distance", 15.0)
        )
        
        # 权重参数
        self.overlap_penalty_weight = coord_config.get("overlap_penalty_weight", 1.5)
        self.division_reward_weight = coord_config.get("division_reward_weight", 0.8)
        self.joint_discovery_weight = coord_config.get("joint_discovery_weight", 2.0)
        
        # 传感器范围 (从配置中读取或使用默认值)
        self.sensor_range = self._estimate_sensor_range(params)
        
        logger.info(f"CoordinationManager initialized (enabled={self.enabled})")
    
    def _estimate_sensor_range(self, params: dict) -> float:
        """估算传感器观测范围"""
        try:
            # 从传感器配置中估算
            fov_x = params.get("sensor", {}).get("field_of_view", {}).get("angle_x", 60)
            altitude = params.get("experiment", {}).get("constraints", {}).get("max_altitude", 25)
            
            # 简化估算: range ≈ altitude * tan(fov/2)
            import math
            range_estimate = altitude * math.tan(math.radians(fov_x / 2))
            
            return float(range_estimate)
        except Exception:
            # 默认值
            return 15.0
    
    def update_positions(self, agent_positions: List[np.ndarray]):
        """更新所有智能体的位置"""
        if not self.enabled:
            return
        
        for agent_id, pos in enumerate(agent_positions):
            self.overlap_detector.update_position(agent_id, pos)
    
    def calculate_coordination_rewards(
        self,
        agent_id: int,
        agent_positions: List[np.ndarray],
        regions: Optional[List] = None
    ) -> Dict[str, float]:
        """
        计算协同相关的奖励和惩罚
        
        Args:
            agent_id: 当前智能体ID
            agent_positions: 所有智能体位置列表
            regions: 搜索区域列表 (可选)
        
        Returns:
            rewards: 包含各项协同奖励的字典
        """
        rewards = {
            'overlap_penalty': 0.0,
            'division_reward': 0.0,
            'collaboration_reward': 0.0,
            'total_coordination': 0.0
        }
        
        if not self.enabled or len(agent_positions) < 2:
            return rewards
        
        try:
            # 1. 计算抗重叠惩罚
            observation_overlap = self.overlap_detector.calculate_observation_overlap(
                agent_positions, self.sensor_range
            )
            
            other_agent_ids = [i for i in range(len(agent_positions)) if i != agent_id]
            path_overlap = self.overlap_detector.calculate_path_overlap(
                agent_id, other_agent_ids
            )
            
            # 综合重叠度
            total_overlap = (observation_overlap + path_overlap) / 2.0
            
            if total_overlap > self.overlap_detector.overlap_threshold:
                rewards['overlap_penalty'] = -self.overlap_penalty_weight * total_overlap
            
            # 2. 计算区域分工奖励
            division_score = self.division_metric.calculate_division_score(
                agent_positions, regions
            )
            rewards['division_reward'] = self.division_reward_weight * division_score
            
            # 3. 检测协同发现
            is_collab, collab_score = self.collaboration_detector.detect_collaboration(
                agent_positions, regions
            )
            
            if is_collab:
                rewards['collaboration_reward'] = self.joint_discovery_weight * collab_score
            
            # 总协同奖励
            rewards['total_coordination'] = sum([
                rewards['overlap_penalty'],
                rewards['division_reward'],
                rewards['collaboration_reward']
            ])
            
        except Exception as e:
            logger.warning(f"Failed to calculate coordination rewards: {e}")
        
        return rewards
    
    def get_statistics(self) -> Dict:
        """获取协同统计信息"""
        return {
            'enabled': self.enabled,
            'num_agents': self.num_agents,
            'sensor_range': self.sensor_range,
        }
    
    def reset(self):
        """重置协同管理器"""
        self.overlap_detector.reset()


# ==================== 工具函数 ====================

def visualize_coordination(
    agent_positions: List[np.ndarray],
    regions: Optional[List] = None,
    collaboration_distance: float = 15.0,
    title: str = "Agent Coordination"
):
    """
    可视化智能体协同情况
    
    Args:
        agent_positions: 智能体位置列表
        regions: 搜索区域列表
        collaboration_distance: 协同距离阈值
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制区域
    if regions:
        for region in regions:
            if region.type == "rectangle":
                x_min, y_min, x_max, y_max = region.coordinates[0]
                rect = Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=True,
                    alpha=0.2,
                    color='blue' if region.priority > 0.7 else 'green',
                    label=f"{region.name} (p={region.priority})"
                )
                ax.add_patch(rect)
    
    # 绘制智能体
    positions = np.array([pos[:2] for pos in agent_positions])
    ax.scatter(positions[:, 0], positions[:, 1], c='red', s=200, marker='o', label='Agents', zorder=3)
    
    # 绘制智能体ID
    for i, pos in enumerate(positions):
        ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
    
    # 绘制协同关系 (距离小于阈值的连线)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= collaboration_distance:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    'r--', linewidth=2, alpha=0.5
                )
    
    # 绘制观测范围 (示意)
    for pos in positions:
        circle = Circle(pos, 10, fill=False, linestyle='--', color='red', alpha=0.3)
        ax.add_patch(circle)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig


if __name__ == "__main__":
    # 测试协同管理器
    print("Testing Coordination Manager...")
    
    # 创建测试配置
    test_params = {
        "experiment": {
            "coordination": {
                "enable": True,
                "overlap_penalty_weight": 1.5,
                "division_reward_weight": 0.8,
                "joint_discovery_weight": 2.0,
                "collaboration_distance": 15.0,
                "overlap_threshold": 0.3
            },
            "constraints": {
                "max_altitude": 25
            }
        },
        "sensor": {
            "field_of_view": {
                "angle_x": 60
            }
        }
    }
    
    # 初始化管理器
    manager = CoordinationManager(test_params, num_agents=4)
    
    # 测试场景1: 智能体重叠
    print("\n场景1: 智能体重叠")
    positions_overlap = [
        np.array([10.0, 10.0, 10.0]),
        np.array([12.0, 11.0, 10.0]),  # 非常接近
        np.array([30.0, 30.0, 15.0]),
        np.array([40.0, 40.0, 20.0])
    ]
    
    manager.update_positions(positions_overlap)
    rewards = manager.calculate_coordination_rewards(0, positions_overlap)
    print(f"协同奖励: {rewards}")
    
    # 测试场景2: 智能体分散
    print("\n场景2: 智能体分散")
    positions_dispersed = [
        np.array([10.0, 10.0, 10.0]),
        np.array([30.0, 10.0, 10.0]),
        np.array([10.0, 30.0, 15.0]),
        np.array([30.0, 30.0, 20.0])
    ]
    
    manager.reset()
    manager.update_positions(positions_dispersed)
    rewards = manager.calculate_coordination_rewards(0, positions_dispersed)
    print(f"协同奖励: {rewards}")
    
    print("\n测试完成!")
