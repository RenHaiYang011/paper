import logging
from typing import Dict, List, Optional

import numpy as np
import cv2
import torch

from agent.state_space import AgentStateSpace
from utils.state import get_w_entropy_map
from mapping.search_regions import SearchRegionManager
from mapping.frontier_detection import FrontierManager

logger = logging.getLogger(__name__)


def get_network_input(
    local_information,
    fused_local_map,
    simulated_map,
    agent_id,
    t,
    params,
    batch_memory,
    agent_state_space,
    search_region_manager: Optional[SearchRegionManager] = None,
    frontier_manager: Optional[FrontierManager] = None,
    discovered_targets: Optional[set] = None,
):
    total_budget = params["experiment"]["constraints"]["budget"]
    spacing = params["experiment"]["constraints"]["spacing"]
    class_weighting = params["experiment"]["missions"]["class_weighting"]

    position_map, position = get_position_feature_map(
        local_information, agent_id, agent_state_space, params
    )

    w_entropy_map, weightings_map, entropy_map, local_w_entropy_map, prob_map = get_w_entropy_map(
        local_information[agent_id]["footprint_img"],
        fused_local_map,
        simulated_map,
        "actor",
        agent_state_space,
        class_weighting,
    )

    budget_map = get_budget_feature_map(total_budget - t, position_map, params)
    altitude_map = get_altitude_map(
        local_information, agent_id, agent_state_space, position_map, spacing
    )
    agent_id_map = get_agent_id_map(agent_id, position_map, params)
    footprint_map = get_footprint_map(local_information, agent_id, agent_state_space, t)
    
    # Target discovery history map (new feature)
    discovery_history_map = get_discovery_history_map(
        discovered_targets, agent_state_space, position_map
    )
    
    # Exploration intensity map (improved exploration guidance)
    exploration_intensity_map = get_exploration_intensity_map(
        local_information, agent_id, agent_state_space, position_map, t
    )
    
    # Base observation layers (now 9 layers instead of 7)
    base_layers = [
        budget_map,
        agent_id_map,
        position_map,
        w_entropy_map,
        local_w_entropy_map,
        prob_map,
        footprint_map,
        discovery_history_map,
        exploration_intensity_map,
    ]
    
    # Add region search features if available (3 additional layers)
    if search_region_manager is not None:
        region_features = get_region_search_features(
            local_information, agent_id, agent_state_space, position_map, search_region_manager
        )
        base_layers.extend([
            region_features['region_priority_map'],
            region_features['region_distance_map'],
            region_features['search_completion_map'],
        ])
    
    # Add frontier map if available (1 additional layer)
    if frontier_manager is not None and frontier_manager.enabled:
        frontier_feature = get_frontier_feature_map(
            frontier_manager, agent_state_space, position_map
        )
        base_layers.append(frontier_feature)
    
    observation_map = torch.tensor(np.dstack(base_layers))

    return observation_map


def get_footprint_map(local_information, agent_id, agent_state_space, t):
    footprint_map = local_information[agent_id]["map2communicate"].copy()
    footprint_map[footprint_map < 0.49] = 1
    footprint_map[footprint_map > 0.51] = 1
    for agent in local_information:
        if agent == agent_id:
            pass
        else:
            new_map = local_information[agent]["map2communicate"].copy()
            footprint_map[new_map < 0.49] = 0
            footprint_map[new_map > 0.51] = 0
    new_map = local_information[agent_id]["map2communicate"].copy()
    footprint_map[new_map < 0.49] = 1
    footprint_map[new_map > 0.51] = 1

    footprint_map = cv2.resize(
        footprint_map,
        (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
        interpolation=cv2.INTER_AREA,
    )

    return footprint_map


def get_agent_id_map(agent_id, position_map, params: Dict) -> np.array:
    n_agents = params["experiment"]["missions"]["n_agents"]
    return np.ones_like(position_map) * ((agent_id + 1) / n_agents)


def get_budget_feature_map(remaining_budget, position_map, params: Dict):
    total_budget = params["experiment"]["constraints"]["budget"]
    return np.ones_like(position_map) * (remaining_budget / total_budget)


def get_altitude_map(
    local_information, agent_id, agent_state_space, position_map, spacing
):
    altitude = None
    for idx in local_information:
        if idx == agent_id:
            own_position = local_information[idx]["position"]
            altitude = own_position[2]
            break
    return np.ones_like(position_map) * (
        (altitude // spacing) / agent_state_space.space_z_dim
    )


def get_position_feature_map(
    local_information: Dict,
    agent_id: int,
    agent_state_space: AgentStateSpace,
    params: Dict,
) -> np.array:
    n_agents = params["experiment"]["missions"]["n_agents"]

    own_position = None
    relative_indexes = None
    other_positions = []
    position_map = np.ones(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim)
    )

    for idx in local_information:
        if idx == agent_id:
            own_position = agent_state_space.position_to_index(
                local_information[idx]["position"]
            )
            relative_indexes = [
                [
                    idx,
                    [5, 5, (own_position[2] + 1) / (agent_state_space.space_z_dim + 1)],
                ]
            ]
            if own_position[0] < 5:
                position_map[0 : 5 - own_position[0], :] = 0
            if own_position[1] < 5:
                position_map[:, 0 : 5 - own_position[1]] = 0
            if own_position[0] > 5:
                position_map[
                    agent_state_space.space_x_dim - 1 - (own_position[0] - 6) :, :
                ] = 0
            if own_position[1] > 5:
                position_map[
                    :, agent_state_space.space_y_dim - 1 - (own_position[1] - 6) :
                ] = 0
        else:
            other_position_idx = agent_state_space.position_to_index(
                local_information[idx]["position"]
            )
            other_positions.append([idx, other_position_idx])

    for other_position in other_positions:
        relative_indexes.append(
            [
                other_position[0],
                [
                    other_position[1][0] - own_position[0] + 5,
                    other_position[1][1] - own_position[1] + 5,
                    (other_position[1][2] + 1) / (agent_state_space.space_z_dim + 1),
                ],
            ]
        )
    for relative_index in relative_indexes:
        if (
            relative_index[1][0] >= 0
            and relative_index[1][0] < agent_state_space.space_x_dim
            and relative_index[1][1] >= 0
            and relative_index[1][1] < agent_state_space.space_x_dim
        ):
            position_map[
                int(relative_index[1][0]), int(relative_index[1][1])
            ] = relative_index[1][2]

    return position_map, own_position


def get_previous_action_map(agent_id, batch_memory, t, params: Dict):
    n_agents = params["experiment"]["missions"]["n_agents"]
    n_actions = params["experiment"]["constraints"]["num_actions"]

    if t == 0:
        action = 0
    else:
        for agent in range(n_agents):
            if agent_id == agent_id:
                action = np.float64(batch_memory.get(-1, agent_id, "action").item()) / (
                    n_actions - 1
                )
    return action


def get_region_search_features(
    local_information: Dict,
    agent_id: int,
    agent_state_space: AgentStateSpace,
    position_map: np.array,
    search_region_manager: Optional[SearchRegionManager] = None,
) -> Dict[str, np.array]:
    """
    Get region search related feature maps
    
    Returns:
        Dict containing:
        - region_priority_map: Current region priority (0-1, or 0 if not in region)
        - region_distance_map: Normalized distance to nearest unsearched region
        - search_completion_map: Global search completion percentage
    """
    features = {
        'region_priority_map': np.zeros_like(position_map),
        'region_distance_map': np.zeros_like(position_map),
        'search_completion_map': np.zeros_like(position_map),
    }
    
    if search_region_manager is None:
        # Return zero maps if no region search
        return features
    
    try:
        # Get agent position
        own_position = local_information[agent_id]["position"]
        position_array = np.array(own_position)
        
        # 1. Current region priority
        current_region = search_region_manager.get_region_at_position(position_array)
        if current_region is not None:
            priority = current_region.priority
        else:
            priority = 0.0
        features['region_priority_map'] = np.ones_like(position_map) * priority
        
        # 2. Distance to nearest unsearched region
        nearest_region = search_region_manager.get_nearest_unsearched_region(position_array)
        if nearest_region is not None:
            region_center = search_region_manager._get_region_center(nearest_region)
            # Calculate distance
            distance = np.linalg.norm(position_array[:2] - region_center)
            # Normalize by map diagonal
            map_diagonal = np.sqrt(
                (agent_state_space.space_x_dim * agent_state_space.spacing) ** 2 +
                (agent_state_space.space_y_dim * agent_state_space.spacing) ** 2
            )
            normalized_distance = min(distance / map_diagonal, 1.0)
        else:
            # All regions searched
            normalized_distance = 0.0
        features['region_distance_map'] = np.ones_like(position_map) * normalized_distance
        
        # 3. Global search completion
        completion = search_region_manager.get_search_completion()
        features['search_completion_map'] = np.ones_like(position_map) * completion
        
    except Exception as e:
        logger.warning(f"Failed to calculate region search features: {e}")
    
    return features


def get_frontier_feature_map(
    frontier_manager: FrontierManager,
    agent_state_space: AgentStateSpace,
    position_map: np.array,
) -> np.array:
    """
    Get frontier map as a feature layer for the agent observation
    
    The frontier map shows the boundary between explored and unexplored regions,
    helping the agent decide where to explore next.
    
    Args:
        frontier_manager: FrontierManager instance with current frontier map
        agent_state_space: Agent state space for map dimensions
        position_map: Reference map for shape
    
    Returns:
        frontier_feature_map: Resized frontier map (0-1), same shape as position_map
    """
    try:
        # Get current frontier map from manager
        frontier_map = frontier_manager.get_frontier_map()
        
        if frontier_map is None or frontier_map.size == 0:
            # Return zero map if no frontier available
            return np.zeros_like(position_map)
        
        # Resize frontier map to match agent observation space
        frontier_resized = cv2.resize(
            frontier_map,
            (position_map.shape[1], position_map.shape[0]),
            interpolation=cv2.INTER_AREA
        )
        
        return frontier_resized
        
    except Exception as e:
        logger.warning(f"Failed to get frontier feature map: {e}")
        return np.zeros_like(position_map)


def get_discovery_history_map(
    discovered_targets: Optional[set],
    agent_state_space: AgentStateSpace,
    position_map: np.array
) -> np.array:
    """
    生成目标发现历史特征图
    
    显示已发现目标的位置，帮助智能体了解搜索进度并避免重复搜索已发现区域
    
    Args:
        discovered_targets: 已发现目标位置的集合 (网格坐标)
        agent_state_space: 智能体状态空间
        position_map: 参考位置图
    
    Returns:
        discovery_map: 发现历史图 (0-1值)
    """
    discovery_map = np.zeros_like(position_map)
    
    if discovered_targets is None or len(discovered_targets) == 0:
        return discovery_map
    
    try:
        # 在已发现目标位置标记为1
        for target_coord in discovered_targets:
            # target_coord 是 (row, col) 格式的网格坐标
            row, col = target_coord
            
            # 确保坐标在有效范围内
            if (0 <= row < discovery_map.shape[0] and 
                0 <= col < discovery_map.shape[1]):
                discovery_map[row, col] = 1.0
                
                # 在目标周围添加衰减的影响区域
                influence_radius = 2  # 影响半径
                for dr in range(-influence_radius, influence_radius + 1):
                    for dc in range(-influence_radius, influence_radius + 1):
                        r, c = row + dr, col + dc
                        if (0 <= r < discovery_map.shape[0] and 
                            0 <= c < discovery_map.shape[1]):
                            distance = np.sqrt(dr*dr + dc*dc)
                            if distance <= influence_radius and distance > 0:
                                # 距离越远影响越小
                                influence = max(0, 1.0 - distance / influence_radius) * 0.5
                                discovery_map[r, c] = max(discovery_map[r, c], influence)
    
    except Exception as e:
        logger.warning(f"Failed to generate discovery history map: {e}")
    
    return discovery_map


def get_exploration_intensity_map(
    local_information: Dict,
    agent_id: int,
    agent_state_space: AgentStateSpace,
    position_map: np.array,
    current_time: int,
    decay_factor: float = 0.9
) -> np.array:
    """
    生成探索强度特征图
    
    显示各区域的探索强度，结合时间衰减，鼓励探索长时间未访问的区域
    
    Args:
        local_information: 本地信息字典
        agent_id: 当前智能体ID
        agent_state_space: 智能体状态空间
        position_map: 参考位置图
        current_time: 当前时间步
        decay_factor: 时间衰减因子
    
    Returns:
        intensity_map: 探索强度图 (0-1值，越高表示最近探索越频繁)
    """
    intensity_map = np.zeros_like(position_map)
    
    try:
        # 基于所有智能体的足迹图计算探索强度
        for agent_idx in local_information:
            agent_info = local_information[agent_idx]
            
            # 获取智能体的足迹或地图信息
            if "map2communicate" in agent_info:
                agent_map = agent_info["map2communicate"]
                
                # 将足迹图调整到状态空间大小
                agent_footprint = cv2.resize(
                    agent_map,
                    (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
                    interpolation=cv2.INTER_AREA,
                )
                
                # 计算探索强度 (基于覆盖度)
                explored_mask = agent_footprint < 0.49  # 已探索区域
                intensity_map += explored_mask.astype(float)
        
        # 归一化到0-1范围
        if np.max(intensity_map) > 0:
            intensity_map = intensity_map / np.max(intensity_map)
        
        # 反转强度：高强度区域（已频繁探索）-> 低值，低强度区域（少探索）-> 高值
        # 这样网络会被引导到探索强度低的区域
        exploration_guidance_map = 1.0 - intensity_map
        
        return exploration_guidance_map
        
    except Exception as e:
        logger.warning(f"Failed to generate exploration intensity map: {e}")
        return np.ones_like(position_map) * 0.5  # 返回中性值

