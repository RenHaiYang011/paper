import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Disable autograd anomaly detection for performance
torch.autograd.set_detect_anomaly(False)


class CriticNetwork(nn.Module):
    def __init__(self, params: Dict):
        super(CriticNetwork, self).__init__()
        self.params = params
        self.n_agents = params["experiment"]["missions"]["n_agents"]
        self.n_actions = params["experiment"]["constraints"]["num_actions"]
        
        # Calculate input channels for Critic network
        # Critic receives Actor output + additional global features
        
        # 1. Actor network input channels (base layers)
        actor_input_channels = 9  # Base: budget, agent_id, position, w_entropy, local_w_entropy, prob, footprint, discovery_history, exploration_intensity
        
        if "search_regions" in self.params:
            actor_input_channels += 3  # Add region search features
            logger.info("Critic network: Actor has 3 region search feature channels")
        
        # Check if frontier-based intrinsic rewards are enabled
        intrinsic_rewards_config = self.params.get("experiment", {}).get("intrinsic_rewards", {})
        if intrinsic_rewards_config.get("enable", False) and intrinsic_rewards_config.get("frontier_reward_weight", 0) > 0:
            state_repr_config = self.params.get("state_representation", {})
            if state_repr_config.get("use_frontier_map", False):
                actor_input_channels += 1  # Add frontier map
                logger.info("Critic network: Actor has 1 frontier map channel")
        
        # 2. Additional critic-specific features (added in critic/transformations.py)
        critic_additional_channels = 5  # position_map(1) + w_entropy_map(1) + prob_map(1) + footprint_map(1) + other_actions_map(1)
        
        # Total input channels for critic
        self.input_channels = actor_input_channels + critic_additional_channels
        
        logger.info(f"Critic network initialized with {self.input_channels} input channels")
        logger.info(f"  - Actor input channels: {actor_input_channels}")
        logger.info(f"  - Critic additional channels: {critic_additional_channels}")
        
        self.conv1 = nn.Conv2d(self.input_channels, 256, (5, 5))
        self.conv2 = nn.Conv2d(256, 256, (4, 4))
        self.conv3 = nn.Conv2d(256, 256, (4, 4))
        # self.conv4 = nn.Conv2d(128, 256, (3, 3))
        self.flatten = nn.Flatten()
        self.activation = torch.nn.ReLU()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_state):
        if input_state.dim() == 3:
            input_state = torch.permute(input_state, (2, 0, 1))
        elif input_state.dim() == 4:
            input_state = torch.permute(input_state, (0, 3, 1, 2))

        output = self.activation(self.conv1(input_state))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        # output = self.activation(self.conv4(output))
        h = torch.squeeze(self.flatten(output))
        output = self.activation(self.fc1(h))
        # output = self.activation(self.fc2(output))
        output = self.fc3(output)

        with torch.no_grad():
            log_probs = self.log_softmax(output)

        return output, log_probs
