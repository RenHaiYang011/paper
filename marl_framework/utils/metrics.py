"""
搜索任务评估指标模块

用于静态搜索任务的性能评估,包括:
1. 搜索核心指标: 发现时间、完成率
2. 搜索效率指标: 覆盖曲线、路径重复度
3. 协同效能指标: 协同效率、负载均衡
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SearchMetrics:
    """
    搜索任务核心指标计算器
    
    跟踪目标发现时间、任务完成情况等关键指标
    """
    
    def __init__(self):
        self.target_discovery_times = []  # 目标发现时间列表
        self.targets_discovered = 0  # 已发现目标数量
        self.total_targets = 0  # 总目标数量
        self.task_start_time = 0
        self.task_completion_time = None
        
    def record_target_discovery(self, target_id: int, discovery_time: int):
        """记录目标发现"""
        self.target_discovery_times.append((target_id, discovery_time))
        self.targets_discovered += 1
    
    def set_task_completion(self, completion_time: int):
        """设置任务完成时间"""
        self.task_completion_time = completion_time
    
    def get_first_discovery_time(self) -> Optional[int]:
        """首次发现目标的时间"""
        if not self.target_discovery_times:
            return None
        return min(t for _, t in self.target_discovery_times)
    
    def get_average_discovery_time(self) -> Optional[float]:
        """平均目标发现时间"""
        if not self.target_discovery_times:
            return None
        times = [t for _, t in self.target_discovery_times]
        return float(np.mean(times))
    
    def get_discovery_rate(self) -> float:
        """目标发现率"""
        if self.total_targets == 0:
            return 0.0
        return self.targets_discovered / self.total_targets
    
    def get_metrics(self) -> Dict:
        """获取所有核心指标"""
        return {
            'first_discovery_time': self.get_first_discovery_time(),
            'average_discovery_time': self.get_average_discovery_time(),
            'task_completion_time': self.task_completion_time,
            'discovery_rate': self.get_discovery_rate(),
            'targets_discovered': self.targets_discovered,
            'total_targets': self.total_targets
        }


class EfficiencyMetrics:
    """
    搜索效率指标计算器
    
    评估搜索效率、路径质量等
    """
    
    def __init__(self):
        self.coverage_history = []  # 覆盖率历史
        self.time_steps = []  # 时间步历史
        self.path_lengths = []  # 各智能体路径长度
        self.visited_cells = []  # 访问过的网格
        
    def record_coverage(self, time_step: int, coverage: float):
        """记录覆盖率"""
        self.time_steps.append(time_step)
        self.coverage_history.append(coverage)
    
    def record_path(self, agent_id: int, path: List[np.ndarray]):
        """记录智能体路径"""
        if agent_id >= len(self.path_lengths):
            self.path_lengths.extend([0] * (agent_id - len(self.path_lengths) + 1))
        
        # 计算路径长度
        path_length = 0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(path[i][:2] - path[i+1][:2])
        
        self.path_lengths[agent_id] = path_length
    
    def record_visited_cell(self, cell: Tuple[int, int], agent_id: int):
        """记录访问的网格"""
        self.visited_cells.append((cell, agent_id))
    
    def calculate_coverage_curve(self) -> Tuple[List[int], List[float]]:
        """
        计算覆盖率曲线
        
        Returns:
            time_steps: 时间步列表
            coverage: 覆盖率列表
        """
        return self.time_steps, self.coverage_history
    
    def calculate_path_redundancy(self) -> float:
        """
        计算路径重复度
        
        重复度 = 重复访问的网格数 / 总访问次数
        """
        if not self.visited_cells:
            return 0.0
        
        unique_cells = set(cell for cell, _ in self.visited_cells)
        total_visits = len(self.visited_cells)
        redundant_visits = total_visits - len(unique_cells)
        
        return redundant_visits / total_visits if total_visits > 0 else 0.0
    
    def calculate_overlap_degree(self) -> float:
        """
        计算重叠度
        
        重叠度 = 多智能体访问的网格数 / 总访问网格数
        """
        if not self.visited_cells:
            return 0.0
        
        # 统计每个网格被多少个智能体访问
        cell_agents = {}
        for cell, agent_id in self.visited_cells:
            if cell not in cell_agents:
                cell_agents[cell] = set()
            cell_agents[cell].add(agent_id)
        
        # 计算被多个智能体访问的网格比例
        overlapped_cells = sum(1 for agents in cell_agents.values() if len(agents) > 1)
        total_cells = len(cell_agents)
        
        return overlapped_cells / total_cells if total_cells > 0 else 0.0
    
    def calculate_search_efficiency(self) -> float:
        """
        计算搜索效率
        
        效率 = 最终覆盖率 / 总时间步数
        """
        if not self.coverage_history or not self.time_steps:
            return 0.0
        
        final_coverage = self.coverage_history[-1]
        total_time = self.time_steps[-1] if self.time_steps else 1
        
        return final_coverage / total_time
    
    def get_metrics(self) -> Dict:
        """获取所有效率指标"""
        return {
            'path_redundancy': self.calculate_path_redundancy(),
            'overlap_degree': self.calculate_overlap_degree(),
            'search_efficiency': self.calculate_search_efficiency(),
            'total_path_length': sum(self.path_lengths),
            'avg_path_length': np.mean(self.path_lengths) if self.path_lengths else 0.0,
            'coverage_curve': self.calculate_coverage_curve()
        }


class CoordinationMetrics:
    """
    协同效能指标计算器
    
    评估多智能体协同的效果
    """
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.single_agent_times = []  # 单智能体完成时间
        self.multi_agent_time = None  # 多智能体完成时间
        self.agent_workloads = [0] * num_agents  # 各智能体工作量
        self.communication_count = 0  # 通信次数
        self.communication_overhead = 0.0  # 通信开销
        
    def set_single_agent_baseline(self, times: List[int]):
        """设置单智能体基线时间"""
        self.single_agent_times = times
    
    def set_multi_agent_time(self, time: int):
        """设置多智能体完成时间"""
        self.multi_agent_time = time
    
    def record_agent_workload(self, agent_id: int, workload: float):
        """记录智能体工作量(如访问的网格数)"""
        if agent_id < len(self.agent_workloads):
            self.agent_workloads[agent_id] = workload
    
    def record_communication(self, cost: float = 1.0):
        """记录通信事件"""
        self.communication_count += 1
        self.communication_overhead += cost
    
    def calculate_coordination_efficiency(self) -> Optional[float]:
        """
        计算协同效率
        
        协同效率 = (单智能体总时间) / (多智能体实际时间)
        理想情况下应接近智能体数量
        """
        if not self.single_agent_times or self.multi_agent_time is None:
            return None
        
        total_single_time = sum(self.single_agent_times)
        
        if self.multi_agent_time == 0:
            return None
        
        efficiency = total_single_time / self.multi_agent_time
        
        return efficiency
    
    def calculate_load_balance(self) -> float:
        """
        计算负载均衡度
        
        使用变异系数衡量: 标准差/均值
        越接近0表示越均衡
        """
        if not self.agent_workloads:
            return 0.0
        
        workloads = [w for w in self.agent_workloads if w > 0]
        
        if not workloads:
            return 0.0
        
        mean_workload = np.mean(workloads)
        std_workload = np.std(workloads)
        
        if mean_workload == 0:
            return 0.0
        
        # 返回1 - CV,使得1表示完全均衡
        cv = std_workload / mean_workload
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score
    
    def calculate_speedup(self) -> Optional[float]:
        """
        计算加速比
        
        加速比 = 单智能体平均时间 / 多智能体时间
        """
        if not self.single_agent_times or self.multi_agent_time is None:
            return None
        
        avg_single_time = np.mean(self.single_agent_times)
        
        if self.multi_agent_time == 0:
            return None
        
        return avg_single_time / self.multi_agent_time
    
    def get_metrics(self) -> Dict:
        """获取所有协同指标"""
        return {
            'coordination_efficiency': self.calculate_coordination_efficiency(),
            'load_balance': self.calculate_load_balance(),
            'speedup': self.calculate_speedup(),
            'communication_count': self.communication_count,
            'communication_overhead': self.communication_overhead,
            'avg_communication_per_agent': self.communication_count / self.num_agents if self.num_agents > 0 else 0
        }


class EpisodeMetricsTracker:
    """
    单次Episode的完整指标跟踪器
    
    整合所有指标计算器,提供统一接口
    """
    
    def __init__(self, num_agents: int, total_targets: int = 0):
        self.search_metrics = SearchMetrics()
        self.search_metrics.total_targets = total_targets
        
        self.efficiency_metrics = EfficiencyMetrics()
        self.coordination_metrics = CoordinationMetrics(num_agents)
        
        self.episode_start_time = 0
        self.episode_end_time = None
        
    def start_episode(self):
        """开始Episode"""
        self.episode_start_time = 0
    
    def end_episode(self, end_time: int):
        """结束Episode"""
        self.episode_end_time = end_time
        self.search_metrics.set_task_completion(end_time)
        self.coordination_metrics.set_multi_agent_time(end_time)
    
    def record_step(
        self,
        time_step: int,
        coverage: float,
        agent_positions: List[np.ndarray],
        visited_cells: Optional[List[Tuple[Tuple[int, int], int]]] = None
    ):
        """
        记录单个时间步的信息
        
        Args:
            time_step: 当前时间步
            coverage: 当前覆盖率
            agent_positions: 智能体位置列表
            visited_cells: 本步访问的网格列表 [(cell, agent_id), ...]
        """
        # 记录覆盖率
        self.efficiency_metrics.record_coverage(time_step, coverage)
        
        # 记录访问的网格
        if visited_cells:
            for cell, agent_id in visited_cells:
                self.efficiency_metrics.record_visited_cell(cell, agent_id)
    
    def get_all_metrics(self) -> Dict:
        """获取所有指标"""
        metrics = {
            'episode_time': self.episode_end_time,
        }
        
        # 搜索核心指标
        metrics.update({
            f'search/{k}': v 
            for k, v in self.search_metrics.get_metrics().items()
        })
        
        # 效率指标
        efficiency = self.efficiency_metrics.get_metrics()
        # 排除coverage_curve (太大,单独处理)
        coverage_curve = efficiency.pop('coverage_curve', None)
        metrics.update({
            f'efficiency/{k}': v 
            for k, v in efficiency.items()
        })
        
        # 协同指标
        metrics.update({
            f'coordination/{k}': v 
            for k, v in self.coordination_metrics.get_metrics().items()
        })
        
        return metrics


class MetricsAggregator:
    """
    多Episode指标聚合器
    
    跨多个episode计算平均指标和统计信息
    """
    
    def __init__(self):
        self.episode_metrics = []
        
    def add_episode(self, metrics: Dict):
        """添加一个episode的指标"""
        self.episode_metrics.append(metrics)
    
    def get_aggregated_metrics(self) -> Dict:
        """
        获取聚合后的指标
        
        计算均值、标准差、最大值、最小值
        """
        if not self.episode_metrics:
            return {}
        
        aggregated = {}
        
        # 获取所有指标键
        all_keys = set()
        for ep_metrics in self.episode_metrics:
            all_keys.update(ep_metrics.keys())
        
        # 对每个指标计算统计量
        for key in all_keys:
            values = []
            for ep_metrics in self.episode_metrics:
                val = ep_metrics.get(key)
                if val is not None and isinstance(val, (int, float)):
                    values.append(float(val))
            
            if values:
                aggregated[f'{key}/mean'] = np.mean(values)
                aggregated[f'{key}/std'] = np.std(values)
                aggregated[f'{key}/max'] = np.max(values)
                aggregated[f'{key}/min'] = np.min(values)
        
        return aggregated
    
    def get_summary(self) -> str:
        """获取可读的摘要"""
        agg = self.get_aggregated_metrics()
        
        summary = "=== Metrics Summary ===\n"
        
        # 搜索核心指标
        summary += "\n[Search Metrics]\n"
        for key in sorted(k for k in agg.keys() if k.startswith('search/')):
            summary += f"  {key}: {agg[key]:.4f}\n"
        
        # 效率指标
        summary += "\n[Efficiency Metrics]\n"
        for key in sorted(k for k in agg.keys() if k.startswith('efficiency/')):
            summary += f"  {key}: {agg[key]:.4f}\n"
        
        # 协同指标
        summary += "\n[Coordination Metrics]\n"
        for key in sorted(k for k in agg.keys() if k.startswith('coordination/')):
            summary += f"  {key}: {agg[key]:.4f}\n"
        
        return summary


# ==================== 工具函数 ====================

def compute_coverage_from_map(coverage_map: np.ndarray, threshold: float = 0.5) -> float:
    """
    从覆盖图计算覆盖率
    
    Args:
        coverage_map: 覆盖图 (0-1之间的值)
        threshold: 判断已覆盖的阈值
    
    Returns:
        coverage: 覆盖率 (0-1)
    """
    if coverage_map is None or coverage_map.size == 0:
        return 0.0
    
    covered_cells = np.sum(coverage_map > threshold)
    total_cells = coverage_map.size
    
    return covered_cells / total_cells


def position_to_cell(position: np.ndarray, spacing: float) -> Tuple[int, int]:
    """
    将实际位置转换为网格坐标
    
    Args:
        position: [x, y, z] 位置
        spacing: 网格间距
    
    Returns:
        cell: (row, col) 网格坐标
    """
    col = int(position[0] / spacing)
    row = int(position[1] / spacing)
    return (row, col)


if __name__ == "__main__":
    # 测试指标计算
    print("Testing Metrics Calculation...")
    
    # 创建tracker
    tracker = EpisodeMetricsTracker(num_agents=4, total_targets=10)
    tracker.start_episode()
    
    # 模拟搜索过程
    for t in range(50):
        coverage = min(t / 50, 1.0)
        
        agent_positions = [
            np.array([t * 0.5, 10, 10]),
            np.array([10, t * 0.5, 15]),
            np.array([t * 0.3, t * 0.3, 20]),
            np.array([20 - t * 0.2, 20, 10])
        ]
        
        # 记录步骤
        visited = [
            ((int(t * 0.5), 10), 0),
            ((10, int(t * 0.5)), 1),
        ]
        
        tracker.record_step(t, coverage, agent_positions, visited)
        
        # 模拟目标发现
        if t == 10:
            tracker.search_metrics.record_target_discovery(0, t)
        if t == 25:
            tracker.search_metrics.record_target_discovery(1, t)
    
    tracker.end_episode(50)
    
    # 获取指标
    metrics = tracker.get_all_metrics()
    
    print("\n=== Episode Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    # 测试聚合器
    aggregator = MetricsAggregator()
    aggregator.add_episode(metrics)
    
    # 添加更多episode
    for _ in range(5):
        tracker2 = EpisodeMetricsTracker(num_agents=4, total_targets=10)
        tracker2.start_episode()
        for t in range(45):
            tracker2.efficiency_metrics.record_coverage(t, t / 45)
        tracker2.end_episode(45)
        aggregator.add_episode(tracker2.get_all_metrics())
    
    print("\n" + aggregator.get_summary())
    
    print("\nTest completed!")
