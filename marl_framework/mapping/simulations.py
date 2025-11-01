import logging
from typing import Dict

import numpy as np
import torch

from mapping import ground_truths
from mapping import ground_truths_region_based
from mapping.grid_maps import GridMap

from marl_framework.sensors import Sensor
from marl_framework.sensors.models.sensor_models import AltitudeSensorModel

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(
        self,
        params: Dict,
        sensor: Sensor,
        episode: int,
        sensor_model: AltitudeSensorModel,
    ):
        self.params = params
        self.sensor = sensor
        self.cluster_radius = self.params["sensor"]["simulation"]["cluster_radius"]
        self.seed = params["environment"]["seed"]
        self.grid_map = GridMap(self.params)
        self.x_dim_pixel = self.grid_map.x_dim
        self.y_dim_pixel = self.grid_map.y_dim
        self.simulated_map = self.simulate_map(episode)
        self.sensor_model = sensor_model

    def simulate_map(self, episode: int):
        """
        生成 simulated_map (地面真值)
        
        可以通过配置选择生成方式：
        - "region_based": 根据 search_regions 配置生成目标分布
        - "random_field": 使用原始的随机场生成（默认）
        """
        map_type = self.params.get("sensor", {}).get("simulation", {}).get("map_type", "random_field")
        
        if map_type == "region_based":
            logger.info(f"Generating region-based target map for episode {episode}")
            return ground_truths_region_based.generate_region_based_map(
                self.params,
                self.y_dim_pixel,
                self.x_dim_pixel,
                episode,
            )
        else:
            # 默认使用原始随机场生成
            return ground_truths.gaussian_random_field(
                lambda k: k ** (-self.cluster_radius),
                self.y_dim_pixel,
                self.x_dim_pixel,
                episode,
            )

    def get_measurement(self, altitude, footprint, mode):
        map_section = self.simulated_map[
            footprint[2] : footprint[3], footprint[0] : footprint[1]
        ].copy()
        sensor_noise = self.sensor_model.get_noise_variance(altitude)
        new_grid_value = np.round(
            self.get_noisy_map_section(sensor_noise, map_section, mode), 3
        )
        new_grid_value = np.float32(new_grid_value)
        return new_grid_value

    @staticmethod
    def get_noisy_map_section(sensor_noise, map_section, mode):
        accuracy = 1 - sensor_noise
        correctness = torch.bernoulli(
                torch.full(np.shape(map_section), 1 - sensor_noise, dtype=torch.float32)
            )
        correctness = (
                torch.reshape(correctness, np.shape(map_section)).cpu().numpy()
            )
        noisy_map_section = np.abs(map_section - (1 - correctness))
        return noisy_map_section
