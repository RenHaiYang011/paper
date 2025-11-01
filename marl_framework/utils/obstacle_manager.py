"""
障碍物管理器 - 用于管理3D空间中的障碍物并提供碰撞检测
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObstacleManager:
    """
    管理环境中的障碍物，提供碰撞检测和避障功能
    """
    
    def __init__(self, params: dict):
        """
        初始化障碍物管理器
        
        Args:
            params: 环境配置参数
        """
        self.params = params
        self.obstacles = []
        
        # 从配置中读取障碍物参数
        obstacle_config = params.get("experiment", {}).get("obstacles", {})
        self.enabled = obstacle_config.get("enable", False)
        self.safety_margin = obstacle_config.get("safety_margin", 2.0)  # 安全边界（米）
        self.collision_penalty = obstacle_config.get("collision_penalty", 50.0)
        
        # 环境尺寸
        self.x_dim = params["environment"]["x_dim"]
        self.y_dim = params["environment"]["y_dim"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        
        logger.info(f"ObstacleManager initialized: enabled={self.enabled}, "
                   f"safety_margin={self.safety_margin}m, "
                   f"collision_penalty={self.collision_penalty}")
    
    def set_obstacles(self, obstacles: List[Dict]):
        """
        设置障碍物列表
        
        Args:
            obstacles: 障碍物列表，每个障碍物包含 {'x', 'y', 'z', 'height', 'radius'}
        """
        self.obstacles = []
        for obs in obstacles:
            # 标准化障碍物数据
            normalized_obs = {
                'x': float(obs['x']),
                'y': float(obs['y']),
                'z': float(obs.get('z', 0)),
                'height': float(obs.get('height', 10)),
                'radius': float(obs.get('radius', 2.75))  # 默认半径 = base_size / 2
            }
            self.obstacles.append(normalized_obs)
        
        logger.info(f"Loaded {len(self.obstacles)} obstacles")
        for i, obs in enumerate(self.obstacles):
            logger.debug(f"  Obstacle {i+1}: pos=({obs['x']:.1f}, {obs['y']:.1f}), "
                        f"height={obs['height']:.1f}m, radius={obs['radius']:.1f}m")
    
    def is_position_in_obstacle(self, position: np.ndarray, include_margin: bool = True) -> Tuple[bool, Optional[int]]:
        """
        检查位置是否在障碍物内部
        
        Args:
            position: 位置 [x, y, z]
            include_margin: 是否包含安全边界
            
        Returns:
            (is_collision, obstacle_index): 是否碰撞及碰撞的障碍物索引
        """
        if not self.enabled or len(self.obstacles) == 0:
            return False, None
        
        x, y, z = position[0], position[1], position[2]
        
        for i, obs in enumerate(self.obstacles):
            # 计算水平距离
            dx = x - obs['x']
            dy = y - obs['y']
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            
            # 障碍物半径（考虑安全边界）
            effective_radius = obs['radius']
            if include_margin:
                effective_radius += self.safety_margin
            
            # 障碍物高度范围
            obs_z_min = obs['z']
            obs_z_max = obs['z'] + obs['height']
            
            # 检查是否在障碍物范围内
            if horizontal_dist <= effective_radius:
                if obs_z_min <= z <= obs_z_max:
                    return True, i
        
        return False, None
    
    def is_path_colliding(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                         num_samples: int = 10) -> Tuple[bool, Optional[int]]:
        """
        检查从起点到终点的路径是否与障碍物碰撞
        
        Args:
            start_pos: 起始位置 [x, y, z]
            end_pos: 结束位置 [x, y, z]
            num_samples: 采样点数量
            
        Returns:
            (is_collision, obstacle_index): 是否碰撞及碰撞的障碍物索引
        """
        if not self.enabled or len(self.obstacles) == 0:
            return False, None
        
        # 在路径上采样多个点进行检测
        for t in np.linspace(0, 1, num_samples):
            sample_pos = start_pos + t * (end_pos - start_pos)
            is_collision, obs_idx = self.is_position_in_obstacle(sample_pos, include_margin=True)
            if is_collision:
                return True, obs_idx
        
        return False, None
    
    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """
        获取位置到最近障碍物的距离
        
        Args:
            position: 位置 [x, y, z]
            
        Returns:
            到最近障碍物的距离（米），如果没有障碍物返回无穷大
        """
        if not self.enabled or len(self.obstacles) == 0:
            return float('inf')
        
        x, y, z = position[0], position[1], position[2]
        min_distance = float('inf')
        
        for obs in self.obstacles:
            # 计算水平距离
            dx = x - obs['x']
            dy = y - obs['y']
            horizontal_dist = np.sqrt(dx**2 + dy**2)
            
            # 障碍物高度范围
            obs_z_min = obs['z']
            obs_z_max = obs['z'] + obs['height']
            
            # 如果在障碍物高度范围内，使用水平距离减去半径
            if obs_z_min <= z <= obs_z_max:
                distance = max(0, horizontal_dist - obs['radius'])
            else:
                # 如果不在高度范围内，计算3D距离
                # 找到障碍物高度范围内最近的点
                closest_z = np.clip(z, obs_z_min, obs_z_max)
                dz = z - closest_z
                distance = np.sqrt(horizontal_dist**2 + dz**2) - obs['radius']
            
            min_distance = min(min_distance, distance)
        
        return max(0, min_distance)
    
    def get_collision_penalty(self, position: np.ndarray) -> float:
        """
        计算位置的碰撞惩罚（基于到障碍物的距离）
        
        Args:
            position: 位置 [x, y, z]
            
        Returns:
            惩罚值（越接近障碍物惩罚越大）
        """
        if not self.enabled or len(self.obstacles) == 0:
            return 0.0
        
        # 检查是否在障碍物内部
        is_collision, _ = self.is_position_in_obstacle(position, include_margin=False)
        if is_collision:
            return self.collision_penalty  # 完全碰撞，最大惩罚
        
        # 检查是否在安全边界内
        is_in_margin, _ = self.is_position_in_obstacle(position, include_margin=True)
        if is_in_margin:
            # 根据到障碍物的距离计算渐变惩罚
            distance = self.get_nearest_obstacle_distance(position)
            if distance < self.safety_margin:
                # 距离越近，惩罚越大（线性衰减）
                penalty_ratio = 1.0 - (distance / self.safety_margin)
                return self.collision_penalty * penalty_ratio * 0.5  # 边界惩罚是完全碰撞的一半
        
        return 0.0
    
    def get_safe_actions_mask(self, current_pos: np.ndarray, 
                             possible_next_positions: List[np.ndarray]) -> np.ndarray:
        """
        获取安全动作掩码（屏蔽会导致碰撞的动作）
        
        Args:
            current_pos: 当前位置 [x, y, z]
            possible_next_positions: 所有可能的下一步位置列表
            
        Returns:
            动作掩码数组，1表示安全，0表示会碰撞
        """
        if not self.enabled or len(self.obstacles) == 0:
            return np.ones(len(possible_next_positions))
        
        mask = np.ones(len(possible_next_positions))
        
        for i, next_pos in enumerate(possible_next_positions):
            # 检查目标位置是否在障碍物内
            is_collision, _ = self.is_position_in_obstacle(next_pos, include_margin=True)
            if is_collision:
                mask[i] = 0
                continue
            
            # 检查路径是否穿过障碍物
            is_path_collision, _ = self.is_path_colliding(current_pos, next_pos)
            if is_path_collision:
                mask[i] = 0
        
        return mask
    
    def visualize_obstacles(self) -> List[Dict]:
        """
        返回用于可视化的障碍物列表
        
        Returns:
            障碍物列表，格式与plotting.py兼容
        """
        return self.obstacles
    
    def get_statistics(self) -> Dict:
        """
        获取障碍物统计信息
        
        Returns:
            统计信息字典
        """
        if not self.enabled or len(self.obstacles) == 0:
            return {
                'enabled': False,
                'num_obstacles': 0
            }
        
        heights = [obs['height'] for obs in self.obstacles]
        radii = [obs['radius'] for obs in self.obstacles]
        
        return {
            'enabled': True,
            'num_obstacles': len(self.obstacles),
            'avg_height': np.mean(heights),
            'max_height': np.max(heights),
            'avg_radius': np.mean(radii),
            'safety_margin': self.safety_margin,
            'collision_penalty': self.collision_penalty
        }


def test_obstacle_manager():
    """测试障碍物管理器"""
    print("=== 测试障碍物管理器 ===\n")
    
    # 测试参数
    test_params = {
        "environment": {
            "x_dim": 50,
            "y_dim": 50
        },
        "experiment": {
            "constraints": {
                "min_altitude": 5,
                "max_altitude": 25
            },
            "obstacles": {
                "enable": True,
                "safety_margin": 2.0,
                "collision_penalty": 50.0
            }
        }
    }
    
    # 创建管理器
    manager = ObstacleManager(test_params)
    
    # 添加测试障碍物
    obstacles = [
        {'x': 10, 'y': 10, 'z': 0, 'height': 15, 'radius': 3},
        {'x': 30, 'y': 20, 'z': 0, 'height': 12, 'radius': 2.5},
        {'x': 25, 'y': 35, 'z': 0, 'height': 10, 'radius': 3.5}
    ]
    manager.set_obstacles(obstacles)
    
    # 测试1: 位置碰撞检测
    print("测试1: 位置碰撞检测")
    test_positions = [
        (10, 10, 8, "障碍物中心"),
        (12, 12, 8, "障碍物边缘"),
        (20, 20, 8, "安全区域"),
        (10, 10, 20, "障碍物上方（超出高度）")
    ]
    
    for x, y, z, desc in test_positions:
        pos = np.array([x, y, z])
        is_collision, obs_idx = manager.is_position_in_obstacle(pos, include_margin=True)
        distance = manager.get_nearest_obstacle_distance(pos)
        penalty = manager.get_collision_penalty(pos)
        print(f"  位置 ({x}, {y}, {z}) - {desc}")
        print(f"    碰撞: {is_collision}, 最近距离: {distance:.2f}m, 惩罚: {penalty:.2f}")
    
    # 测试2: 路径碰撞检测
    print("\n测试2: 路径碰撞检测")
    test_paths = [
        ((5, 5, 8), (15, 15, 8), "穿过障碍物"),
        ((5, 5, 8), (5, 15, 8), "绕过障碍物"),
    ]
    
    for start, end, desc in test_paths:
        start_pos = np.array(start)
        end_pos = np.array(end)
        is_collision, obs_idx = manager.is_path_colliding(start_pos, end_pos)
        print(f"  路径 {start} -> {end} - {desc}")
        print(f"    碰撞: {is_collision}")
    
    # 测试3: 统计信息
    print("\n测试3: 统计信息")
    stats = manager.get_statistics()
    print(f"  {stats}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_obstacle_manager()
