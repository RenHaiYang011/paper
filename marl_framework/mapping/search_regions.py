"""
区域搜索管理模块
用于定义、管理和追踪多个感兴趣区域 (ROI) 的搜索任务
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SearchRegion:
    """搜索区域定义"""
    name: str
    region_type: str  # 'rectangle', 'circle', 'polygon'
    priority: float   # 优先级权重 0.0-1.0
    min_coverage: float  # 最小覆盖要求
    coordinates: List  # 区域坐标定义
    search_density: str  # 'high', 'medium', 'low'
    required_visits: int  # 需要的访问次数
    
    def __post_init__(self):
        """初始化后处理"""
        self.current_coverage = 0.0  # 当前覆盖率
        self.visit_count_map = None  # 访问次数地图
        self.last_visit_time = {}    # 最后访问时间


class SearchRegionManager:
    """搜索区域管理器"""
    
    def __init__(self, config: dict, map_shape: Tuple[int, int], spacing: float):
        """
        初始化搜索区域管理器
        
        Args:
            config: 搜索区域配置字典
            map_shape: 地图形状 (height, width)
            spacing: 网格间距
        """
        self.config = config
        self.map_shape = map_shape
        self.spacing = spacing
        self.regions: List[SearchRegion] = []
        self.global_visit_map = np.zeros(map_shape, dtype=np.int32)
        self.global_coverage_map = np.zeros(map_shape, dtype=np.float32)
        
        # 搜索策略配置
        strategy_config = config.get('strategy', {})
        self.search_mode = strategy_config.get('mode', 'priority_based')
        self.allow_overlap = strategy_config.get('allow_overlap', False)
        self.revisit_threshold = strategy_config.get('revisit_threshold', 0.1)
        self.completion_threshold = strategy_config.get('completion_threshold', 0.9)
        
        # 搜索密度要求
        density_req = config.get('density_requirements', {})
        self.density_visits = {
            'high': density_req.get('high', 3),
            'medium': density_req.get('medium', 2),
            'low': density_req.get('low', 1)
        }
        
        # 初始化所有搜索区域
        self._initialize_regions()
    
    def _initialize_regions(self):
        """从配置初始化所有搜索区域"""
        regions_config = self.config.get('regions', [])
        
        for region_cfg in regions_config:
            # 创建搜索区域对象
            region = SearchRegion(
                name=region_cfg['name'],
                region_type=region_cfg['type'],
                priority=region_cfg['priority'],
                min_coverage=region_cfg['min_coverage'],
                coordinates=region_cfg['coordinates'],
                search_density=region_cfg.get('search_density', 'medium'),
                required_visits=self.density_visits[region_cfg.get('search_density', 'medium')]
            )
            
            # 初始化该区域的访问次数地图
            region.visit_count_map = np.zeros(self.map_shape, dtype=np.int32)
            
            self.regions.append(region)
        
        print(f"✓ 初始化了 {len(self.regions)} 个搜索区域")
        for region in self.regions:
            print(f"  - {region.name}: 优先级={region.priority}, "
                  f"密度={region.search_density}, 需要访问={region.required_visits}次")
    
    def is_in_region(self, position: np.ndarray, region: SearchRegion) -> bool:
        """
        判断位置是否在指定区域内
        
        Args:
            position: [x, y, z] 位置
            region: 搜索区域对象
            
        Returns:
            是否在区域内
        """
        x, y = position[0], position[1]
        
        if region.region_type == 'rectangle':
            # 矩形区域: [x_min, y_min, x_max, y_max]
            coords = region.coordinates[0]
            return (coords[0] <= x <= coords[2] and 
                    coords[1] <= y <= coords[3])
        
        elif region.region_type == 'circle':
            # 圆形区域: [center_x, center_y, radius]
            coords = region.coordinates[0]
            center_x, center_y, radius = coords[0], coords[1], coords[2]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance <= radius
        
        elif region.region_type == 'polygon':
            # 多边形区域: [[x1,y1], [x2,y2], ...]
            # 使用射线法判断点是否在多边形内
            return self._point_in_polygon((x, y), region.coordinates)
        
        return False
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[List[float]]) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def update_visit(self, position: np.ndarray, agent_id: int, 
                    current_step: int, sensor_footprint: np.ndarray):
        """
        更新位置的访问记录
        
        Args:
            position: 智能体位置 [x, y, z]
            agent_id: 智能体ID
            current_step: 当前步数
            sensor_footprint: 传感器足迹(覆盖的网格)
        """
        # 更新全局访问地图
        for (i, j) in sensor_footprint:
            if 0 <= i < self.map_shape[0] and 0 <= j < self.map_shape[1]:
                self.global_visit_map[i, j] += 1
                self.global_coverage_map[i, j] = 1.0
        
        # 更新各个区域的访问记录
        for region in self.regions:
            if self.is_in_region(position, region):
                # 更新该区域的访问地图
                for (i, j) in sensor_footprint:
                    if 0 <= i < self.map_shape[0] and 0 <= j < self.map_shape[1]:
                        region.visit_count_map[i, j] += 1
                
                # 更新最后访问时间
                region.last_visit_time[agent_id] = current_step
                
                # 更新区域覆盖率
                region.current_coverage = self._calculate_region_coverage(region)
    
    def _calculate_region_coverage(self, region: SearchRegion) -> float:
        """
        计算区域的覆盖率
        
        Args:
            region: 搜索区域
            
        Returns:
            覆盖率 (0.0-1.0)
        """
        # 创建区域掩码
        region_mask = self._create_region_mask(region)
        
        # 计算满足搜索密度要求的网格数
        satisfied_cells = np.sum(region.visit_count_map >= region.required_visits)
        total_cells = np.sum(region_mask)
        
        if total_cells == 0:
            return 0.0
        
        return satisfied_cells / total_cells
    
    def _create_region_mask(self, region: SearchRegion) -> np.ndarray:
        """创建区域掩码"""
        mask = np.zeros(self.map_shape, dtype=bool)
        
        if region.region_type == 'rectangle':
            coords = region.coordinates[0]
            # 转换为网格索引
            i_min = int(coords[1] // self.spacing)
            i_max = int(coords[3] // self.spacing)
            j_min = int(coords[0] // self.spacing)
            j_max = int(coords[2] // self.spacing)
            
            i_min = max(0, min(i_min, self.map_shape[0]))
            i_max = max(0, min(i_max, self.map_shape[0]))
            j_min = max(0, min(j_min, self.map_shape[1]))
            j_max = max(0, min(j_max, self.map_shape[1]))
            
            mask[i_min:i_max, j_min:j_max] = True
        
        # TODO: 实现圆形和多边形的掩码创建
        
        return mask
    
    def get_region_at_position(self, position: np.ndarray) -> Optional[SearchRegion]:
        """获取位置所在的搜索区域"""
        for region in self.regions:
            if self.is_in_region(position, region):
                return region
        return None
    
    def get_nearest_unsearched_region(self, position: np.ndarray) -> Optional[SearchRegion]:
        """
        获取最近的未完成搜索区域
        
        Args:
            position: 当前位置
            
        Returns:
            最近的未完成搜索区域
        """
        unsearched_regions = [r for r in self.regions 
                             if r.current_coverage < r.min_coverage]
        
        if not unsearched_regions:
            return None
        
        # 按优先级和距离排序
        def score_region(region):
            center = self._get_region_center(region)
            distance = np.linalg.norm(position[:2] - center)
            # 分数 = 优先级 / (距离 + 1)
            return region.priority / (distance + 1.0)
        
        return max(unsearched_regions, key=score_region)
    
    def _get_region_center(self, region: SearchRegion) -> np.ndarray:
        """获取区域中心点"""
        if region.region_type == 'rectangle':
            coords = region.coordinates[0]
            center_x = (coords[0] + coords[2]) / 2
            center_y = (coords[1] + coords[3]) / 2
            return np.array([center_x, center_y])
        
        # TODO: 实现其他类型区域的中心计算
        return np.array([0.0, 0.0])
    
    def calculate_search_reward(self, position: np.ndarray, 
                               prev_position: np.ndarray,
                               sensor_footprint: np.ndarray) -> Dict[str, float]:
        """
        计算区域搜索相关的奖励
        
        Returns:
            包含各种奖励分量的字典
        """
        rewards = {
            'region_coverage': 0.0,
            'region_priority': 0.0,
            'search_density': 0.0,
            'redundant_search': 0.0,
            'region_transition': 0.0
        }
        
        current_region = self.get_region_at_position(position)
        prev_region = self.get_region_at_position(prev_position)
        
        if current_region is not None:
            # 1. 区域覆盖奖励
            newly_covered = 0
            for (i, j) in sensor_footprint:
                if (0 <= i < self.map_shape[0] and 0 <= j < self.map_shape[1]):
                    # 如果是首次达到所需访问次数,给予奖励
                    if (current_region.visit_count_map[i, j] == current_region.required_visits - 1):
                        newly_covered += 1
            
            rewards['region_coverage'] = newly_covered
            
            # 2. 优先级奖励
            rewards['region_priority'] = current_region.priority * newly_covered
            
            # 3. 搜索密度奖励
            under_searched = 0
            over_searched = 0
            for (i, j) in sensor_footprint:
                if (0 <= i < self.map_shape[0] and 0 <= j < self.map_shape[1]):
                    visits = current_region.visit_count_map[i, j]
                    if visits < current_region.required_visits:
                        under_searched += 1
                    elif visits > current_region.required_visits:
                        over_searched += 1
            
            # 搜索未达标区域有奖励,过度搜索有惩罚
            rewards['search_density'] = under_searched * 0.5
            rewards['redundant_search'] = -over_searched * 0.2
            
            # 4. 区域切换惩罚
            if prev_region is not None and prev_region.name != current_region.name:
                # 如果前一个区域还没搜索完就切换,给予惩罚
                if prev_region.current_coverage < prev_region.min_coverage:
                    rewards['region_transition'] = -0.5
        
        return rewards
    
    def get_search_completion(self) -> float:
        """
        计算全局搜索完成度
        
        Returns:
            完成度 (0.0-1.0)
        """
        if len(self.regions) == 0:
            return 1.0
        
        # 加权平均完成度
        total_weight = sum(r.priority for r in self.regions)
        if total_weight == 0:
            return 0.0
        
        weighted_completion = sum(
            r.current_coverage * r.priority for r in self.regions
        )
        
        return weighted_completion / total_weight
    
    def is_search_complete(self) -> bool:
        """判断搜索是否完成"""
        completion = self.get_search_completion()
        return completion >= self.completion_threshold
    
    def get_search_statistics(self) -> Dict:
        """获取搜索统计信息"""
        stats = {
            'global_completion': self.get_search_completion(),
            'total_visits': np.sum(self.global_visit_map),
            'covered_cells': np.sum(self.global_coverage_map > 0),
            'regions': []
        }
        
        for region in self.regions:
            region_stats = {
                'name': region.name,
                'priority': region.priority,
                'coverage': region.current_coverage,
                'required_coverage': region.min_coverage,
                'status': 'completed' if region.current_coverage >= region.min_coverage else 'in_progress',
                'avg_visits': np.mean(region.visit_count_map[region.visit_count_map > 0]) if np.any(region.visit_count_map > 0) else 0
            }
            stats['regions'].append(region_stats)
        
        return stats
    
    def reset(self):
        """重置所有搜索记录"""
        self.global_visit_map = np.zeros(self.map_shape, dtype=np.int32)
        self.global_coverage_map = np.zeros(self.map_shape, dtype=np.float32)
        
        for region in self.regions:
            region.current_coverage = 0.0
            region.visit_count_map = np.zeros(self.map_shape, dtype=np.int32)
            region.last_visit_time = {}
