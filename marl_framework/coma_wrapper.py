import copy
import logging
from typing import Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter

from marl_framework.actor.learner import ActorLearner
from marl_framework.actor.network import ActorNetwork
from marl_framework.agent.agent import Agent
from marl_framework.agent.communication_log import CommunicationLog
from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.critic.learner import CriticLearner
from marl_framework.critic.network import CriticNetwork
from marl_framework.mapping.search_regions import SearchRegionManager
from marl_framework.mapping.frontier_detection import FrontierManager
from marl_framework.utils.coordination import CoordinationManager

from actor.transformations import get_network_input as get_actor_input
from critic.transformations import get_network_input as get_critic_input
from utils.reward import get_global_reward
from utils.utils import compute_coverage


logger = logging.getLogger(__name__)


class COMAWrapper:
    def __init__(self, params: Dict, writer: SummaryWriter):
        self.params = params
        # keep writer to log additional metrics (coverage delta)
        self.writer = writer
        self.mission_type = self.params["experiment"]["missions"]["type"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.class_weighting = params["experiment"]["missions"]["class_weighting"]
        self.altitude_diversity_weight = params["experiment"].get("altitude_diversity_weight", 0.0)
        self.agent_state_space = AgentStateSpace(self.params)
        
        # Region search parameters
        self.search_region_manager: Optional[SearchRegionManager] = None
        self.region_coverage_weight = params["experiment"].get("region_coverage_weight", 0.0)
        self.region_priority_weight = params["experiment"].get("region_priority_weight", 0.0)
        self.search_density_weight = params["experiment"].get("search_density_weight", 0.0)
        self.search_completion_weight = params["experiment"].get("search_completion_weight", 0.0)
        self.redundant_search_penalty = params["experiment"].get("redundant_search_penalty", 0.0)
        self.region_transition_penalty = params["experiment"].get("region_transition_penalty", 0.0)
        
        # Frontier-based intrinsic reward parameters
        self.frontier_manager: Optional[FrontierManager] = None
        intrinsic_rewards = params.get("experiment", {}).get("intrinsic_rewards", {})
        if intrinsic_rewards.get("enable", False):
            self.frontier_manager = FrontierManager(params)
            logger.info("FrontierManager initialized for intrinsic rewards")
        
        # Coordination mechanism parameters
        self.coordination_manager: Optional[CoordinationManager] = None
        coordination_config = params.get("experiment", {}).get("coordination", {})
        if coordination_config.get("enable", False):
            self.coordination_manager = CoordinationManager(params, self.n_agents)
            logger.info("CoordinationManager initialized for multi-agent coordination")
        
        # Initialize networks and move to GPU
        import constants
        self.actor_network = ActorNetwork(self.params).to(constants.DEVICE)
        self.actor_learner = ActorLearner(
            self.params, writer, self.actor_network, self.agent_state_space
        )
        self.critic_network = CriticNetwork(self.params).to(constants.DEVICE)
        self.target_critic_network = copy.deepcopy(self.critic_network).to(constants.DEVICE)
        self.critic_learner = CriticLearner(self.params, writer, self.critic_network)
        
        logger.info(f"Actor network moved to {constants.DEVICE}")
        logger.info(f"Critic network moved to {constants.DEVICE}")
    
    def set_search_region_manager(self, manager: SearchRegionManager):
        """Set the search region manager (called from COMAMission)"""
        self.search_region_manager = manager
        logger.info(f"SearchRegionManager attached to COMAWrapper")


    def build_observations(
        self, mapping, agents, num_episode, t, params, batch_memory, mode
    ):
        communication_log = CommunicationLog(self.params, num_episode)
        local_maps = []
        positions = []
        global_information = {}
        for agent_id in range(self.n_agents):
            global_information, local_map, position = agents[agent_id].communicate(
                t, num_episode, communication_log, mode
            )
            local_maps.append(local_map)
            positions.append(position)

        observations = []
        for agent_id in range(self.n_agents):
            local_information, fused_local_map = agents[agent_id].receive_messages(
                communication_log, agent_id, t
            )

            observation = get_actor_input(
                local_information,
                fused_local_map,
                mapping.simulated_map,
                agent_id,
                t,
                params,
                batch_memory,
                self.agent_state_space,
                self.search_region_manager,  # Pass search_region_manager
                self.frontier_manager,  # Pass frontier_manager
            )

            batch_memory.add(agent_id, observation=observation)
            observations.append(observation)

        return global_information, positions, observations

    def steps(
        self,
        mapping,
        t: int,
        agents: List[Agent],
        accumulated_map_knowledge,
        num_episode,
        batch_memory,
        global_information,
        simulated_map,
        params,
        mode,
        global_step: int = None,
        prev_positions: List = None,
    ):
        next_maps = []
        next_positions = []
        actions = []
        footprints = []
        altitudes = []
        maps2communicate = []

        critic_map_knowledge = mapping.fuse_map(
            accumulated_map_knowledge, global_information, mode, "global"
        )
        
        # Update frontier map based on current coverage
        if self.frontier_manager is not None and self.frontier_manager.enabled:
            try:
                # Use the fused map as coverage map
                self.frontier_manager.update(critic_map_knowledge)
            except Exception as e:
                logger.warning(f"Failed to update frontier manager: {e}")
        
        # Update coordination manager with agent positions
        if self.coordination_manager is not None and self.coordination_manager.enabled:
            try:
                self.coordination_manager.update_positions(next_positions)
            except Exception as e:
                logger.warning(f"Failed to update coordination manager: {e}")

        for agent_id in range(self.n_agents):
            next_map, next_position, eps, action, footprint_idx, map2communicate = agents[
                agent_id
            ].step(
                agent_id, t, num_episode, batch_memory, mode, next_positions
            )
            next_maps.append(next_map)
            next_positions.append(next_position)
            try:
        
                actions.append(action.tolist()[0])
            except:
                actions.append(action)
            footprints.append(footprint_idx)
            altitudes.append(next_position[2])
            maps2communicate.append(map2communicate)

        if self.mission_type == "DeepQ":
            for agent_id in range(self.n_agents):
                update_simulation = mapping.fuse_map(
                    critic_map_knowledge.copy(),
                    [maps2communicate[agent_id]],
                    agent_id,
                    "global",
                )
                # correct argument ordering and pass coverage weight from params
                done, relative_reward, absolute_reward = get_global_reward(
                    critic_map_knowledge,
                    update_simulation,
                    self.mission_type,
                    footprints,
                    simulated_map,
                    self.agent_state_space,
                    actions[agent_id],
                    agent_id,
                    t,
                    self.budget,
                    self.params["experiment"].get("coverage_weight", None),
                    distance_weight=self.params["experiment"].get("distance_weight", 0.0),
                    footprint_weight=self.params["experiment"].get("footprint_weight", 0.0),
                    collision_weight=self.params["experiment"].get("collision_weight", 0.0),
                    collision_distance=self.params["experiment"].get("collision_distance", 1.0),
                    writer=self.writer,
                    global_step=global_step,
                    class_weighting=self.class_weighting,
                    altitude_diversity_weight=self.altitude_diversity_weight,
                )
                batch_memory.insert(-1, agent_id, reward=relative_reward)

        for agent_id in range(self.n_agents):
            states = get_critic_input(
                t,
                global_information,
                critic_map_knowledge,
                batch_memory,
                agent_id,
                simulated_map,
                params,
            )
        next_global_map = mapping.fuse_map(
            accumulated_map_knowledge, global_information, mode, "global"
        )

        if self.mission_type == "COMA":
            done, relative_reward, absolute_reward = get_global_reward(
                accumulated_map_knowledge,  #
                next_global_map,
                self.mission_type,
                None,
                simulated_map,
                self.agent_state_space,
                actions,
                None,
                t,
                self.budget,
                self.params["experiment"].get("coverage_weight", None),
                distance_weight=self.params["experiment"].get("distance_weight", 0.0),
                footprint_weight=self.params["experiment"].get("footprint_weight", 0.0),
                collision_weight=self.params["experiment"].get("collision_weight", 0.0),
                collision_distance=self.params["experiment"].get("collision_distance", 1.0),
                prev_positions=prev_positions,
                next_positions=next_positions,
                writer=self.writer,
                global_step=global_step,
                class_weighting=self.class_weighting,
                altitude_diversity_weight=self.altitude_diversity_weight,
                # Region search parameters
                search_region_manager=self.search_region_manager,
                region_coverage_weight=self.region_coverage_weight,
                region_priority_weight=self.region_priority_weight,
                search_density_weight=self.search_density_weight,
                search_completion_weight=self.search_completion_weight,
                redundant_search_penalty=self.redundant_search_penalty,
                region_transition_penalty=self.region_transition_penalty,
                sensor_footprint=None,  # TODO: get actual sensor footprint
                # Frontier-based intrinsic reward parameters
                frontier_manager=self.frontier_manager,
                spacing=self.params["experiment"]["constraints"]["spacing"],
                # Coordination parameters
                coordination_manager=self.coordination_manager,
            )

            # log coverage delta and absolute coverage to TensorBoard if writer available
            try:
                coverage_before = compute_coverage(accumulated_map_knowledge)
                coverage_after = compute_coverage(next_global_map)
                coverage_delta = coverage_after - coverage_before
                if hasattr(self, "writer") and self.writer is not None:
                    try:
                        # write absolute coverage and delta with global_step if provided
                        if global_step is not None:
                            self.writer.add_scalar(
                                "Metrics/Coverage", float(coverage_after), global_step
                            )
                            self.writer.add_scalar(
                                "Metrics/CoverageDelta", float(coverage_delta), global_step
                            )
                        else:
                            self.writer.add_scalar("Metrics/Coverage", float(coverage_after))
                            self.writer.add_scalar("Metrics/CoverageDelta", float(coverage_delta))
                    except Exception:
                        pass
            except Exception:
                pass

        if t == self.budget:
            done = True

        if self.mission_type == "COMA":
            for agent_id in range(self.n_agents):
                batch_memory.insert(-1, agent_id, reward=relative_reward, done=done)
        if self.mission_type == "DeepQ":
            for agent_id in range(self.n_agents):
                batch_memory.insert(-1, agent_id, done=done)

        return (
            batch_memory,
            relative_reward,
            absolute_reward,
            done,
            next_positions,
            eps,
            actions,
            altitudes,
            next_global_map,
        )
