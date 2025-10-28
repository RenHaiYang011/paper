import numpy as np
import matplotlib.pyplot as plt
import torch

from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.utils.state import get_w_entropy_map
from marl_framework.mapping.search_regions import SearchRegionManager
from marl_framework.mapping.frontier_detection import FrontierManager
from marl_framework.utils.coordination import CoordinationManager

from utils.utils import compute_euclidean_distance, compute_coverage


def get_global_reward(
    last_map,
    next_map,
    mission_type,
    footprints,
    simulated_map: np.array,
    agent_state_space: AgentStateSpace,
    actions,
    agent_id,
    t,
    budget,
    coverage_weight: float = None,
    distance_weight: float = 0.0,
    footprint_weight: float = 0.0,
    collision_weight: float = 0.0,
    prev_positions=None,
    next_positions=None,
    writer=None,
    global_step=None,
    collision_distance=1.0,
    class_weighting: list = None,
    altitude_diversity_weight: float = 0.0,
    # Region search parameters
    search_region_manager: SearchRegionManager = None,
    region_coverage_weight: float = 0.0,
    region_priority_weight: float = 0.0,
    search_density_weight: float = 0.0,
    search_completion_weight: float = 0.0,
    redundant_search_penalty: float = 0.0,
    region_transition_penalty: float = 0.0,
    sensor_footprint: np.ndarray = None,
    # Frontier-based intrinsic reward parameters
    frontier_manager: FrontierManager = None,
    spacing: float = 5.0,
    # Coordination parameters
    coordination_manager: CoordinationManager = None,
):
    done = False
    reward = 0
    scale = 10  # 40
    offset = 0.17  # 1.05

    o_min = 0
    o_max = 0.02
    p_max = 1
    fp_factor = 1

    absolute_utility_reward, relative_utility_reward = get_utility_reward(
        last_map, next_map, simulated_map, agent_state_space, coverage_weight, class_weighting
    )

    # distance penalty (average over agents) — applied on utility level before scaling
    mean_dist = 0.0
    try:
        if distance_weight is not None and float(distance_weight) != 0.0 and prev_positions is not None and next_positions is not None:
            dists = []
            # handle cases where positions are lists of tuples/arrays
            for i in range(min(len(prev_positions), len(next_positions))):
                p = np.array(prev_positions[i], dtype=np.float64)
                q = np.array(next_positions[i], dtype=np.float64)
                d = compute_euclidean_distance(p, q)
                dists.append(d)
            if len(dists) > 0:
                mean_dist = float(np.mean(dists))
                # subtract distance contribution from utility reward
                absolute_utility_reward -= float(distance_weight) * mean_dist
    except Exception:
        # Do not fail reward computation due to distance penalty calculation
        mean_dist = 0.0

    # print(f"absolute_utility_reward: {absolute_utility_reward}")
    absolute_reward = scale * absolute_utility_reward - offset
    relative_reward = 22 * relative_utility_reward - 0.5

    # footprint overlap penalty
    fp_pen = 0.0
    try:
        if footprints is not None and float(footprint_weight) != 0.0 and agent_id is not None:
            fp_pen = get_footprint_penalty(footprints, agent_id, simulated_map, o_min, o_max, p_max)
            absolute_reward -= float(footprint_weight) * float(fp_pen)
    except Exception:
        fp_pen = 0.0

    # collision penalty: if next_positions indicates collision, apply penalty
    coll_pen = 0.0
    try:
        if float(collision_weight) != 0.0 and next_positions is not None:
            # detect any collisions in next_positions
            done_flag, coll_reward = get_collision_reward(next_positions, False, collision_distance=collision_distance)
            if done_flag:
                coll_pen = float(abs(coll_reward))
                absolute_reward -= float(collision_weight) * coll_pen
    except Exception:
        coll_pen = 0.0

    # altitude diversity bonus: encourage agents to explore different altitudes
    altitude_bonus = 0.0
    try:
        if float(altitude_diversity_weight) != 0.0 and next_positions is not None and len(next_positions) > 1:
            # Calculate altitude variance across all agents
            altitudes = [pos[2] for pos in next_positions if len(pos) > 2]
            if len(altitudes) > 1:
                altitude_variance = float(np.var(altitudes))
                # Also reward individual altitude changes
                if prev_positions is not None and len(prev_positions) == len(next_positions):
                    altitude_changes = []
                    for i in range(len(prev_positions)):
                        if len(prev_positions[i]) > 2 and len(next_positions[i]) > 2:
                            alt_change = abs(next_positions[i][2] - prev_positions[i][2])
                            altitude_changes.append(alt_change)
                    if altitude_changes:
                        mean_altitude_change = float(np.mean(altitude_changes))
                        # Bonus combines variance (diversity across agents) and change (temporal variation)
                        altitude_bonus = float(altitude_diversity_weight) * (altitude_variance * 0.01 + mean_altitude_change * 0.1)
                        absolute_reward += altitude_bonus
    except Exception as e:
        altitude_bonus = 0.0

    # ==================== Region Search Rewards ====================
    region_rewards = {
        'region_coverage': 0.0,
        'region_priority': 0.0,
        'search_density': 0.0,
        'search_completion': 0.0,
        'redundant_search': 0.0,
        'region_transition': 0.0,
        'total_region_reward': 0.0
    }
    
    try:
        if search_region_manager is not None and agent_id is not None:
            # Get current and previous positions for this agent
            if next_positions is not None and len(next_positions) > agent_id:
                current_pos = np.array(next_positions[agent_id])
                prev_pos = np.array(prev_positions[agent_id]) if prev_positions is not None and len(prev_positions) > agent_id else current_pos
                
                # Calculate region-specific rewards
                region_rewards_raw = search_region_manager.calculate_search_reward(
                    current_pos, prev_pos, sensor_footprint if sensor_footprint is not None else []
                )
                
                # Apply configured weights
                region_rewards['region_coverage'] = float(region_coverage_weight) * region_rewards_raw['region_coverage']
                region_rewards['region_priority'] = float(region_priority_weight) * region_rewards_raw['region_priority']
                region_rewards['search_density'] = float(search_density_weight) * region_rewards_raw['search_density']
                region_rewards['redundant_search'] = float(redundant_search_penalty) * region_rewards_raw['redundant_search']
                region_rewards['region_transition'] = float(region_transition_penalty) * region_rewards_raw['region_transition']
                
                # Search completion bonus (only when reaching completion threshold)
                if search_region_manager.get_search_completion() >= search_region_manager.completion_threshold:
                    region_rewards['search_completion'] = float(search_completion_weight)
                
                # Sum all region rewards
                region_rewards['total_region_reward'] = sum([
                    region_rewards['region_coverage'],
                    region_rewards['region_priority'],
                    region_rewards['search_density'],
                    region_rewards['search_completion'],
                    region_rewards['redundant_search'],
                    region_rewards['region_transition']
                ])
                
                # Add to absolute reward
                absolute_reward += region_rewards['total_region_reward']
    except Exception as e:
        # Don't fail reward computation if region search fails
        pass

    # ==================== Frontier-Based Intrinsic Reward ====================
    frontier_reward = 0.0
    try:
        if frontier_manager is not None and agent_id is not None and frontier_manager.enabled:
            # Get current position for this agent
            if next_positions is not None and len(next_positions) > agent_id:
                current_pos = np.array(next_positions[agent_id])
                
                # Calculate frontier reward based on distance to nearest frontier
                frontier_reward = frontier_manager.calculate_frontier_reward(
                    current_pos, spacing
                )
                
                # Add to absolute reward
                absolute_reward += frontier_reward
    except Exception as e:
        # Don't fail reward computation if frontier reward fails
        frontier_reward = 0.0

    # ==================== Coordination Rewards ====================
    coordination_rewards = {
        'overlap_penalty': 0.0,
        'division_reward': 0.0,
        'collaboration_reward': 0.0,
        'total_coordination': 0.0
    }
    
    try:
        if coordination_manager is not None and agent_id is not None and coordination_manager.enabled:
            # Get all agent positions
            if next_positions is not None and len(next_positions) > 1:
                # Get search regions if available
                regions = None
                if search_region_manager is not None:
                    regions = search_region_manager.regions
                
                # Calculate coordination rewards
                coordination_rewards = coordination_manager.calculate_coordination_rewards(
                    agent_id, next_positions, regions
                )
                
                # Add to absolute reward
                absolute_reward += coordination_rewards['total_coordination']
    except Exception as e:
        # Don't fail reward computation if coordination fails
        coordination_rewards['total_coordination'] = 0.0

    # Log individual penalty components to TensorBoard if writer provided
    try:
        if writer is not None:
            # write zeros if not active
            writer.add_scalar('Penalties/Distance', float(mean_dist), global_step)
            writer.add_scalar('Penalties/Footprint', float(fp_pen), global_step)
            writer.add_scalar('Penalties/Collision', float(coll_pen), global_step)
            writer.add_scalar('Bonuses/Altitude_Diversity', float(altitude_bonus), global_step)
            
            # Region search rewards
            writer.add_scalar('RegionSearch/Coverage_Reward', region_rewards['region_coverage'], global_step)
            writer.add_scalar('RegionSearch/Priority_Reward', region_rewards['region_priority'], global_step)
            writer.add_scalar('RegionSearch/Density_Reward', region_rewards['search_density'], global_step)
            writer.add_scalar('RegionSearch/Completion_Reward', region_rewards['search_completion'], global_step)
            writer.add_scalar('RegionSearch/Redundant_Penalty', region_rewards['redundant_search'], global_step)
            writer.add_scalar('RegionSearch/Transition_Penalty', region_rewards['region_transition'], global_step)
            writer.add_scalar('RegionSearch/Total_Region_Reward', region_rewards['total_region_reward'], global_step)
            
            # Log search completion percentage
            if search_region_manager is not None:
                completion = search_region_manager.get_search_completion()
                writer.add_scalar('RegionSearch/Completion_Percentage', completion * 100, global_step)
            
            # Frontier-based intrinsic reward
            writer.add_scalar('IntrinsicRewards/Frontier_Reward', float(frontier_reward), global_step)
            
            # Frontier statistics (if available)
            if frontier_manager is not None and frontier_manager.enabled:
                frontier_stats = frontier_manager.get_statistics()
                writer.add_scalar('Frontier/Current_Points', frontier_stats['current_frontier_points'], global_step)
                if 'frontier_reward' in frontier_stats:
                    writer.add_scalar('Frontier/Avg_Reward', frontier_stats['frontier_reward']['avg_frontier_reward'], global_step)
                    writer.add_scalar('Frontier/Total_Reward', frontier_stats['frontier_reward']['total_frontier_reward'], global_step)
            
            # Coordination rewards
            writer.add_scalar('Coordination/Overlap_Penalty', coordination_rewards['overlap_penalty'], global_step)
            writer.add_scalar('Coordination/Division_Reward', coordination_rewards['division_reward'], global_step)
            writer.add_scalar('Coordination/Collaboration_Reward', coordination_rewards['collaboration_reward'], global_step)
            writer.add_scalar('Coordination/Total_Coordination_Reward', coordination_rewards['total_coordination'], global_step)
    except Exception:
        # do not raise from logging
        pass

    return (
        done,
        relative_reward,
        absolute_reward,
    )  # done, relative_reward, absolute_reward


def get_collision_reward(next_positions, done, collision_distance=1.0):
    """Return (done_flag, penalty_flag) where done_flag True if any pair closer than collision_distance."""
    for agent1 in range(len(next_positions)):
        for agent2 in range(agent1):
            done = is_collided(next_positions[agent1], next_positions[agent2], collision_distance)
            if done:
                break
        if done:
            break

    return done, -1 if done else 0


def get_utility_reward(
    state: np.array,
    state_: np.array,
    simulated_map: np.array,
    agent_state_space: AgentStateSpace,
    coverage_weight: float = None,
    class_weighting: list = None,
):
    entropy_before = get_w_entropy_map(
        None, state, simulated_map, "reward", agent_state_space, class_weighting
    )[2]
    output = get_w_entropy_map(None, state_, simulated_map, "reward", agent_state_space, class_weighting)
    entropy_after = output[2]
    entropy_reduction = entropy_before - entropy_after

    absolute_reward = np.mean(output[1] * entropy_reduction)
    relative_reward = absolute_reward / (np.mean(output[1] * entropy_before))

    # coverage-based reward: reward positive change in coverage
    try:
        coverage_before = compute_coverage(state)
        coverage_after = compute_coverage(state_)
        coverage_delta = coverage_after - coverage_before
    except Exception:
        coverage_delta = 0.0

    # decide weight: prefer provided coverage_weight, fallback to module constant
    weight = COVERAGE_WEIGHT if coverage_weight is None else float(coverage_weight)
    # add coverage contribution (small weight)
    absolute_reward += weight * coverage_delta
    # avoid divide by zero
    denom = np.mean(output[1] * entropy_before)
    if denom != 0:
        relative_reward = absolute_reward / denom
    else:
        relative_reward = 0.0

    # plt.imshow(state)
    # plt.title(f"state before")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # from marl_framework.constants import REPO_DIR
    # import os
    # os.makedirs(os.path.join(REPO_DIR, "res", "plots"), exist_ok=True)
    # plt.savefig(os.path.join(REPO_DIR, "res", "plots", "state_before.png"))
    #
    # plt.imshow(state_)
    # plt.title(f"state after")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/state_after.png")
    # plt.savefig(os.path.join(REPO_DIR, "res", "plots", "state_after.png"))
    #
    # plt.imshow(entropy_before)
    # plt.title(f"entropy_before")
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_before.png")
    #
    # plt.imshow(entropy_after)
    # plt.title(f"entropy_after")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_after.png")
    #
    # plt.imshow(entropy_reduction)
    # plt.title(f"entropy_reduction")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_reduction.png")
    #
    # plt.imshow(output[1])
    # plt.title(f"weightings")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/weightings.png")
    #
    # plt.imshow(output[1] * entropy_reduction)
    # plt.title(f"weighted reduction")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(os.path.join(REPO_DIR, "res", "plots", "weighted_reduction.png"))

    # print(f"absolute_reward: {absolute_reward}")
    # print(f"relative_reward: {relative_reward}")

    return absolute_reward, relative_reward


def is_collided(next_position_1, next_position_2, collision_distance=1.0):
    try:
        p1 = np.array(next_position_1, dtype=np.float64)
        p2 = np.array(next_position_2, dtype=np.float64)
        # use 2D distance (x,y) to determine collision; include altitude if shapes require
        if p1.size >= 2 and p2.size >= 2:
            d = np.linalg.norm(p1[:2] - p2[:2])
        else:
            d = np.linalg.norm(p1 - p2)
        return d <= float(collision_distance)
    except Exception:
        # fallback to strict equality
        try:
            return np.array_equal(next_position_1, next_position_2)
        except Exception:
            return False


def get_footprint_penalty(footprints, agent_id, simulated_map, o_min, o_max, p_max):
    own_footprint = footprints[agent_id]
    overlap = []
    for fp in range(len(footprints)):
        if fp == agent_id:
            pass
        else:
            overlap.append(
                compute_overlap(own_footprint, footprints[fp], simulated_map)
            )
    mean_overlap = sum(overlap) / len(overlap)
    if mean_overlap > o_max:
        return 0
    elif mean_overlap < o_min:
        return p_max
    else:
        return p_max - ((mean_overlap - o_min) / (o_max - o_min)) / p_max


def compute_overlap(footprint1, footprint2, simulated_map):
    yu = max(footprint1[0], footprint2[0])
    yd = min(footprint1[1], footprint2[1])
    xl = max(footprint1[2], footprint2[2])
    xr = min(footprint1[3], footprint2[3])

    if yu > yd:
        return 0
    if xl > xr:
        return 0
    return ((yd - yu + 1) * (xr - xl + 1)) / np.size(simulated_map)
