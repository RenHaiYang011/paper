import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import marl_framework.constants as constants
from marl_framework.missions.episode_generator import EpisodeGenerator
from marl_framework.missions.missions import Mission
from marl_framework.mapping.search_regions import SearchRegionManager

from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
from mapping.grid_maps import GridMap
from sensors import Sensor
from sensors.models import SensorModel
from utils.plotting import plot_trajectories

# Get the logger that was set up in main
logger = logging.getLogger("marl_framework")


class COMAMission(Mission):
    def __init__(
        self, params: Dict, writer: SummaryWriter, max_mean_episode_return: float
    ):
        super().__init__(params, writer, max_mean_episode_return)
        
        logger.info("🎯 Initializing COMAMission...")
        logger.info(f"📊 TensorBoard writer log dir: {writer.log_dir}")

        self.grid_map = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid_map)
        self.num_episodes = self.params["experiment"]["missions"]["n_episodes"]
        self.batch_size = self.params["networks"]["batch_size"]
        self.batch_number = self.params["networks"]["batch_number"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.n_actions = self.params["experiment"]["constraints"]["num_actions"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.data_passes = self.params["networks"]["data_passes"]
        
        logger.info(f"📈 Training configuration:")
        logger.info(f"  - Episodes: {self.num_episodes}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Agents: {self.n_agents}")
        logger.info(f"  - Budget: {self.budget}")
        
        self.coma_wrapper = COMAWrapper(self.params, self.writer)
        self.training_step_idx = 0
        self.environment_step_idx = 0
        self.episode_returns = []
        self.collision_returns = []
        self.utility_returns = []
        self.mode = "train"
        
        # Initialize SearchRegionManager if region search config exists
        self.search_region_manager: Optional[SearchRegionManager] = None
        if "search_regions" in self.params:
            try:
                # Get map shape from grid_map
                map_shape = (
                    int(self.params["environment"]["y_dim"] // self.params["experiment"]["constraints"]["spacing"]),
                    int(self.params["environment"]["x_dim"] // self.params["experiment"]["constraints"]["spacing"])
                )
                spacing = self.params["experiment"]["constraints"]["spacing"]
                
                self.search_region_manager = SearchRegionManager(
                    config=self.params["search_regions"],
                    map_shape=map_shape,
                    spacing=spacing
                )
                # Attach to coma_wrapper
                self.coma_wrapper.set_search_region_manager(self.search_region_manager)
                logger.info(f"✓ SearchRegionManager initialized with {len(self.search_region_manager.regions)} regions")
            except Exception as e:
                logger.warning(f"Failed to initialize SearchRegionManager: {e}")
                self.search_region_manager = None
        
        # Logging/plotting frequency (throttle heavy ops)
        logging_cfg = self.params.get("logging", {})
        self.figure_interval = int(logging_cfg.get("figure_interval", 20))
        self.histogram_interval = int(logging_cfg.get("histogram_interval", 200))
        
        # For ETA calculation
        self.total_training_steps = int(
                self.num_episodes
                * (self.batch_size * self.batch_number)
                / ((self.budget + 1) * self.n_agents)
            )
        self.start_time = time.time()
        self.last_time = self.start_time

    def execute(self):
        logger.info("🚀 Starting COMA mission execution...")
        logger.info(f"📊 Total training steps planned: {self.total_training_steps}")
        logger.info(f"💾 TensorBoard logs will be saved to: {self.writer.log_dir}")

        batch_memory = BatchMemory(self.params, self.coma_wrapper)
        episode_returns = []
        episode_reward_list = []
        absolute_returns = []
        chosen_actions = []
        chosen_altitudes = []

        logger.info("🔄 Starting training loop...")
        for episode_idx in range(
            1,
            self.total_training_steps + 1,
        ):

            episode = EpisodeGenerator(
                self.params, self.writer, self.grid_map, self.sensor
            )
            (
                episode_return,
                episode_rewards,
                absolute_return,
                simulated_map,
                batch_memory,
                _,
                _,
                eps,
                agent_actions,
                agent_altitudes,
            ) = episode.execute(
                episode_idx, batch_memory, self.coma_wrapper, self.mode, self.training_step_idx
            )

            episode_returns.append(episode_return)
            episode_reward_list.append(episode_rewards)
            absolute_returns.append(absolute_return)
            chosen_actions.append(agent_actions)
            chosen_altitudes.append(agent_altitudes)

            if batch_memory.size() >= self.batch_size * self.batch_number:
                batch_memory.build_td_targets(self.coma_wrapper.target_critic_network)
                for data_pass in range(self.data_passes):
                    batches = batch_memory.build_batches()
                    q_values, critic_metrics = self.coma_wrapper.critic_learner.learn(
                        self.training_step_idx, batches, data_pass
                    )
                    actor_network, actor_metrics = self.coma_wrapper.actor_learner.learn(
                        batches, q_values, eps
                    )
                    if data_pass == 0:
                        self.training_step_idx += 1
                        self.environment_step_idx += batch_memory.size()
                        
                        current_time = time.time()
                        step_time = current_time - self.last_time
                        self.last_time = current_time
                        
                        elapsed_time = current_time - self.start_time
                        avg_step_time = elapsed_time / self.training_step_idx
                        remaining_steps = self.total_training_steps - self.training_step_idx
                        eta_seconds = remaining_steps * avg_step_time
                        
                        eta_h = int(eta_seconds // 3600)
                        eta_m = int((eta_seconds % 3600) // 60)
                        eta_s = int(eta_seconds % 60)
                        
                        print(f"Training step: {self.training_step_idx}/{self.total_training_steps}, Step Time: {step_time:.2f}s, ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}")
                        
                        logger.info(f"Training step: {self.training_step_idx}")
                        logger.info(f"Environment step: {self.environment_step_idx}")
                        
                        # Log first TensorBoard write
                        if self.training_step_idx == 1:
                            logger.info("📊 Writing first data to TensorBoard...")
                            
                        self.add_to_tensorboard(
                            chosen_actions,
                            chosen_altitudes,
                            episode_returns,
                            absolute_returns,
                            episode_reward_list,
                            critic_metrics,
                            actor_metrics,
                        )
                        
                        if self.training_step_idx == 1:
                            logger.info("✅ First TensorBoard write completed")
                            # Immediately flush first write to create files
                            self.writer.flush()
                            logger.info("💾 First TensorBoard flush completed")

                batch_memory.clear()
                self.episode_returns.append(episode_return)
                self.save_best_model(actor_network)
                
                # Periodically flush TensorBoard data to ensure it's saved
                if self.training_step_idx % 20 == 0:
                    self.writer.flush()
                    
                episode_returns = []
                episode_reward_list = []
                absolute_returns = []
                chosen_actions = []
                chosen_altitudes = []

                if self.training_step_idx % 50 == 0:
                    self.mode = "eval"
                    for i in range(50):
                        (
                            episode_return,
                            episode_rewards,
                            absolute_return,
                            simulated_map,
                            batch_memory,
                            agent_positions,
                            t_collision,
                            _,
                            agent_actions,
                            agent_altitudes,
                        ) = episode.execute(
                            episode_idx + i,
                            batch_memory,
                            self.coma_wrapper,
                            self.mode,
                            self.training_step_idx,
                        )
                        if i == 0:
                            plot_trajectories(
                                agent_positions,
                                self.n_agents,
                                self.writer,
                                self.training_step_idx,
                                t_collision,
                                self.budget,
                                simulated_map,
                            )

                        episode_returns.append(episode_return)
                        episode_reward_list.append(episode_rewards)
                        absolute_returns.append(absolute_return)
                        chosen_actions.append(agent_actions)
                        chosen_altitudes.append(agent_altitudes)

                    batch_memory.clear()
                    self.add_to_tensorboard(
                        chosen_actions,
                        chosen_altitudes,
                        episode_returns,
                        absolute_returns,
                        episode_reward_list,
                    )
                    self.mode = "train"
                    episode_returns = []
                    episode_reward_list = []
                    collision_returns = []
                    chosen_actions = []
                    chosen_altitudes = []

        # Ensure TensorBoard data is flushed to disk
        logger.info("Training completed. Flushing TensorBoard data...")
        self.writer.flush()
        self.writer.close()
        logger.info("TensorBoard data saved successfully.")
        
        # Save training results to res/ folder
        self.save_training_results()
        
        return self.max_mean_episode_return

    def _safe_add_histogram(self, tag: str, values, step: int):
        """Safely add histogram to TensorBoard, ensuring numeric dtype and robustness.
        - Flattens and casts to float64
        - Filters non-finite values
        - Skips empty arrays
        - Falls back gracefully on TensorBoard TypeErrors
        """
        # First attempt: fast path using numpy conversion
        try:
            vals = np.asarray(values, dtype=np.float64).reshape(-1)
            # Filter out non-finite values
            vals = vals[np.isfinite(vals)]
        except Exception:
            # Fallback: iterative per-element conversion to float
            vals_list = []
            dropped = 0
            try:
                for i, v in enumerate(np.ravel(values)):
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            vals_list.append(fv)
                    except Exception:
                        dropped += 1
                        if dropped <= 5:
                            logger.debug(f"Dropping non-numeric histogram element for {tag}: index={i}, value={repr(v)[:200]}")
                            # also append to a persistent log for offline inspection
                            try:
                                from marl_framework import constants

                                os.makedirs(constants.LOG_DIR, exist_ok=True)
                                bad_log = os.path.join(constants.LOG_DIR, 'bad_hist_elems.log')
                                with open(bad_log, 'a') as bf:
                                    bf.write(f"{tag}\tindex={i}\tvalue={repr(v)[:400]}\n")
                            except Exception:
                                pass
                if len(vals_list) == 0:
                    return
                vals = np.asarray(vals_list, dtype=np.float64)
            except Exception as e:
                logger.warning(f"Skip histogram '{tag}' due to preprocessing error: {e}")
                return

        if vals.size == 0:
            return

        # Prefer a fixed small bin count to avoid heavy work
        try:
            self.writer.add_histogram(tag, vals, step, bins=50)
        except TypeError as e:
            # Some torch TB + numpy versions have a bug path; try alternative bins
            try:
                self.writer.add_histogram(tag, vals, step, bins='tensorflow')
            except Exception:
                logger.warning(f"Skip histogram '{tag}' due to writer error: {e}")

    def add_to_tensorboard(
        self,
        chosen_actions,
        chosen_altitudes,
        episode_returns,
        absolute_returns,
        episode_rewards,
        critic_metrics=None,
        actor_metrics=None,
    ):

        episode_rewards = [item for sublist in episode_rewards for item in sublist]
        chosen_actions = [item for sublist in chosen_actions for item in sublist]
        chosen_actions = [item for sublist in chosen_actions for item in sublist]
        chosen_altitudes = [item for sublist in chosen_altitudes for item in sublist]
        chosen_altitudes = [item for sublist in chosen_altitudes for item in sublist]

        # Throttle heavy plotting: only every figure_interval steps
        if self.training_step_idx % self.figure_interval == 0:
            action_counts = [chosen_actions.count(i) for i in range(self.n_actions)]
            altitude_counts = [chosen_altitudes.count(i) for i in [5, 10, 15]]

            plt.figure()
            fig_ = sns.barplot(
                x=list(range(self.n_actions)), y=action_counts, color="blue"
            ).get_figure()
            self.writer.add_figure(
                f"Sampled_actions_{self.mode}", fig_, self.training_step_idx, close=True
            )

            plt.figure()
            fig_ = sns.barplot(
                x=[float(i) for i in [5, 10, 15]], y=altitude_counts, color="blue"
            ).get_figure()
            self.writer.add_figure(
                f"Altitudes_{self.mode}", fig_, self.training_step_idx, close=True
            )

        self.writer.add_scalar(
            f"{self.mode}Return/Episode/mean",
            np.mean(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/std",
            np.std(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/max",
            np.max(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/min",
            np.min(absolute_returns),
            self.training_step_idx,
        )

        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/mean",
            np.mean(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/std",
            np.std(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/max",
            np.max(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/min",
            np.min(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/mean",
            np.mean(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/std",
            np.std(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/max",
            np.max(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/min",
            np.min(episode_returns),
            self.training_step_idx,
        )

        if self.mode == "train":
            self.writer.add_scalar(
                "Critic/Loss", critic_metrics[0], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/TD-Targets mean", critic_metrics[1], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/TD-Targets std", critic_metrics[2], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q chosen mean", critic_metrics[3], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values mean", critic_metrics[4], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values min", critic_metrics[5], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values std", critic_metrics[6], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Explained variance", critic_metrics[7], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Discounted returns mean",
                np.array(critic_metrics[8]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Discounted_returns std",
                np.array(critic_metrics[9]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Abs deviation Q-value <-> Return mean",
                np.array(critic_metrics[10]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Abs deviation Q-value <-> Return std",
                np.array(critic_metrics[11]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Log probs according to critic",
                np.array(critic_metrics[12]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/Loss", actor_metrics[0], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Advantages mean", actor_metrics[1], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Advantages std", actor_metrics[2], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Log probs chosen mean", actor_metrics[3], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Policy entropy",
                np.array(actor_metrics[4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/KL divergence policy",
                np.array(actor_metrics[5]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/Hidden state entropy",
                np.array(actor_metrics[6]),
                self.training_step_idx,
            )

            # Throttle parameter histograms as they're expensive
            if self.training_step_idx % self.histogram_interval == 0:
                for tag, params in self.coma_wrapper.critic_network.named_parameters():
                    if params.grad is not None:
                        vals = params.data.detach().flatten().cpu().numpy()
                        self._safe_add_histogram(
                            f"Critic/Parameters/{tag}", vals, self.training_step_idx
                        )
                for tag, params in self.coma_wrapper.actor_network.named_parameters():
                    if params.grad is not None:
                        vals = params.data.detach().flatten().cpu().numpy()
                        self._safe_add_histogram(
                            f"Actor/Parameters/{tag}", vals, self.training_step_idx
                        )

            self.writer.add_scalar(
                "Parameters/Actor/Conv1 gradients",
                np.array(actor_metrics[-1][0]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/Conv2 gradients",
                np.array(actor_metrics[-1][1]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/Conv3 gradients",
                np.array(actor_metrics[-1][2]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC1 gradients",
                np.array(actor_metrics[-1][3]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC2 gradients",
                np.array(actor_metrics[-1][4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC3 gradients",
                np.array(actor_metrics[-1][5]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv1 gradients",
                np.array(critic_metrics[-1][0]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv2 gradients",
                np.array(critic_metrics[-1][1]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv3 gradients",
                np.array(critic_metrics[-1][2]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC1 gradients",
                np.array(critic_metrics[-1][3]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC2 gradients",
                np.array(critic_metrics[-1][4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC3 gradients",
                np.array(critic_metrics[-1][5]),
                self.training_step_idx,
            )
        
        # ==================== Region Search Statistics ====================
        if self.search_region_manager is not None:
            try:
                stats = self.search_region_manager.get_search_statistics()
                
                # Global metrics
                self.writer.add_scalar(
                    f"{self.mode}RegionSearch/Global_Completion",
                    stats['global_completion'],
                    self.training_step_idx,
                )
                self.writer.add_scalar(
                    f"{self.mode}RegionSearch/Total_Visits",
                    stats['total_visits'],
                    self.training_step_idx,
                )
                self.writer.add_scalar(
                    f"{self.mode}RegionSearch/Covered_Cells",
                    stats['covered_cells'],
                    self.training_step_idx,
                )
                
                # Per-region metrics
                for region_stat in stats['regions']:
                    region_name = region_stat['name']
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/{region_name}/Coverage",
                        region_stat['coverage'],
                        self.training_step_idx,
                    )
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/{region_name}/Priority",
                        region_stat['priority'],
                        self.training_step_idx,
                    )
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/{region_name}/Avg_Visits",
                        region_stat['avg_visits'],
                        self.training_step_idx,
                    )
                    # Status as binary (completed=1, in_progress=0)
                    status_val = 1.0 if region_stat['status'] == 'completed' else 0.0
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/{region_name}/Completed",
                        status_val,
                        self.training_step_idx,
                    )
                
                # Calculate search efficiency: covered cells per total visits
                if stats['total_visits'] > 0:
                    search_efficiency = stats['covered_cells'] / stats['total_visits']
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/Search_Efficiency",
                        search_efficiency,
                        self.training_step_idx,
                    )
                
                # Calculate priority satisfaction: weighted completion by priority
                total_priority = sum(r['priority'] for r in stats['regions'])
                if total_priority > 0:
                    priority_satisfaction = sum(
                        r['coverage'] * r['priority'] for r in stats['regions']
                    ) / total_priority
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/Priority_Satisfaction",
                        priority_satisfaction,
                        self.training_step_idx,
                    )
                
                # Calculate redundant search rate: (total_visits - covered_cells) / total_visits
                if stats['total_visits'] > 0:
                    redundant_rate = (stats['total_visits'] - stats['covered_cells']) / stats['total_visits']
                    self.writer.add_scalar(
                        f"{self.mode}RegionSearch/Redundant_Search_Rate",
                        redundant_rate,
                        self.training_step_idx,
                    )
                
            except Exception as e:
                logger.warning(f"Failed to log region search statistics: {e}")

    def save_best_model(self, actor_network):
        running_mean_return = sum(self.episode_returns) / len(self.episode_returns)
        patience = self.params["experiment"]["missions"]["patience"]
        best_model_file_path = os.path.join(constants.LOG_DIR, "best_model.pth")

        if (
            len(self.episode_returns) >= patience
            and running_mean_return > self.max_mean_episode_return
        ):
            self.max_mean_episode_return = running_mean_return
            torch.save(actor_network, best_model_file_path)
        if self.training_step_idx == 300:
            torch.save(
                actor_network, os.path.join(constants.LOG_DIR, "best_model_300.pth")
            )
        if self.training_step_idx == 400:
            torch.save(
                actor_network, os.path.join(constants.LOG_DIR, "best_model_400.pth")
            )
        if self.training_step_idx == 500:
            torch.save(
                actor_network, os.path.join(constants.LOG_DIR, "best_model_500.pth")
            )
        if self.training_step_idx == 600:
            torch.save(
                actor_network, os.path.join(constants.LOG_DIR, "best_model_600.pth")
            )

    def save_training_results(self):
        """Save training results and metrics to res/ folder"""
        import json
        import csv
        from datetime import datetime
        
        # Create res directory if it doesn't exist
        res_dir = constants.EXPERIMENTS_FOLDER
        os.makedirs(res_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare training summary data
        training_summary = {
            "timestamp": timestamp,
            "total_training_steps": self.total_training_steps,
            "final_training_step": self.training_step_idx,
            "max_mean_episode_return": float(self.max_mean_episode_return),
            "final_episode_returns_mean": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
            "final_episode_returns_std": float(np.std(self.episode_returns)) if self.episode_returns else 0.0,
            "total_episodes": len(self.episode_returns),
            "collision_returns_mean": float(np.mean(self.collision_returns)) if self.collision_returns else 0.0,
            "utility_returns_mean": float(np.mean(self.utility_returns)) if self.utility_returns else 0.0,
            "params": {
                "n_agents": self.n_agents,
                "budget": self.budget,
                "batch_size": self.batch_size,
                "batch_number": self.batch_number,
                "n_actions": self.n_actions
            }
        }
        
        # Save training summary as JSON
        summary_file = os.path.join(res_dir, f"training_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Training summary saved: {summary_file}")
        
        # Save episode returns as CSV using standard library
        if self.episode_returns:
            returns_file = os.path.join(res_dir, f"episode_returns_{timestamp}.csv")
            with open(returns_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'episode_return', 'collision_return', 'utility_return'])
                
                for i, ep_return in enumerate(self.episode_returns):
                    collision_return = self.collision_returns[i] if i < len(self.collision_returns) else 0
                    utility_return = self.utility_returns[i] if i < len(self.utility_returns) else 0
                    writer.writerow([i+1, ep_return, collision_return, utility_return])
                    
            logger.info(f"Episode returns saved: {returns_file}")
        
        # Save configuration snapshot
        config_file = os.path.join(res_dir, f"config_snapshot_{timestamp}.json") 
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Configuration snapshot saved: {config_file}")
        
        logger.info(f"All training results saved to: {res_dir}")
