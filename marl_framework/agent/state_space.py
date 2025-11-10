import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AgentStateSpace:
    def __init__(self, params: Dict):
        self.params = params
        self.seed = params["environment"]["seed"]
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        self.space_x_dim = int(params["environment"]["x_dim"] // self.spacing + 1)
        self.space_y_dim = int(params["environment"]["y_dim"] // self.spacing + 1)
        self.space_z_dim = int((self.max_altitude - self.min_altitude) // self.spacing + 1)
        self.space_dim = np.array(
            [self.space_x_dim, self.space_y_dim, self.space_z_dim]
        )

        self.class_weighting = params["experiment"]["missions"]["class_weighting"]
        self.planning_uncertainty = params["experiment"]["missions"][
            "planning_uncertainty"
        ]

    def get_random_agent_state(self, agent_id, episode):
        # Fixed corner start positions for consistent and interpretable results
        # This allows better visualization and analysis of search strategies
        if agent_id == 0:
            state_x = 3    # Near bottom-left corner
            state_y = 3
            state_z = 14   # Mid-altitude
        elif agent_id == 1:
            state_x = 45   # Near bottom-right corner
            state_y = 3
            state_z = 14
        elif agent_id == 2:
            state_x = 45   # Near top-right corner
            state_y = 45
            state_z = 14
        elif agent_id == 3:
            state_x = 3    # Near top-left corner
            state_y = 45
            state_z = 14
        else:
            # Fallback to random if more than 4 agents
            r = np.random.RandomState(seed=self.seed * episode * agent_id)
            state_x = self.spacing * r.randint(0, self.space_x_dim)
            state_y = self.spacing * r.randint(0, self.space_y_dim)
            state_z = self.min_altitude + self.spacing * (self.space_z_dim // 2)

        return np.array([state_x, state_y, state_z])

    def position_to_index(self, position):
        state_x = int(position[0] // self.spacing)
        state_y = int(position[1] // self.spacing)
        state_z = int((position[2] // self.spacing) - 1)
        return np.array([state_x, state_y, state_z])

    def index_to_position(self, state):
        position = np.array(
            [
                state[0] * self.spacing,
                state[1] * self.spacing,
                self.spacing + state[2] * self.spacing,
            ]
        )
        return position
