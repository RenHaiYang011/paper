import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AgentActionSpace:
    def __init__(self, params: Dict, obstacle_manager=None):
        self.params = params
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        self.space_x_dim = 3
        self.space_y_dim = 3
        self.space_z_dim = int((self.max_altitude - self.min_altitude) // self.spacing + 1)
        self.num_actions = params["experiment"]["constraints"]["num_actions"]
        self.environment_x_dim = params["environment"]["x_dim"]
        self.environment_y_dim = params["environment"]["y_dim"]
        self.space_dim = np.array(
            [self.space_x_dim, self.space_y_dim, self.space_z_dim]
        )
        
        # 障碍物管理器（用于避障）
        self.obstacle_manager = obstacle_manager

    def get_action_mask(self, position):
        # General and robust approach: build a 1-D mask of length `num_actions`
        # by checking for each action whether the resulting next position is
        # within bounds. This avoids depending on space_z_dim shape assumptions.
        mask_flatten = np.ones(self.num_actions, dtype=int)

        # For each action, compute the next position and mark invalid actions
        # (moving out of bounds) as 0.
        for a in range(self.num_actions):
            next_pos = self.action_to_position(np.copy(position), a)
            x_ok = 0 <= next_pos[0] <= self.environment_x_dim
            y_ok = 0 <= next_pos[1] <= self.environment_y_dim
            z_ok = self.min_altitude <= next_pos[2] <= self.max_altitude
            if not (x_ok and y_ok and z_ok):
                mask_flatten[a] = 0

        # Enforce "no-op" action masking if the mapping expects it (maintain
        # previous behavior: index 4 for 9-actions, index 13 for 27-actions)
        if self.num_actions == 9:
            if len(mask_flatten) >= 5:
                mask_flatten[4] = 0
            # Provide a 2D view for compatibility
            mask = mask_flatten.reshape((3, 3))
        elif self.num_actions == 27:
            if len(mask_flatten) >= 14:
                mask_flatten[13] = 0
            # Provide a 3D view for compatibility (x,y,z grouping)
            mask = mask_flatten.reshape((3, 3, 3))
        else:
            mask = mask_flatten

        return mask_flatten, mask

    def action_to_position(self, position: np.array, action_index: int):
        offset = [0, 0, 0]

        if self.num_actions == 4:
            if action_index == 0:
                offset = [-self.spacing, 0, 0]
            if action_index == 1:
                offset = [0, -self.spacing, 0]
            if action_index == 2:
                offset = [0, self.spacing, 0]
            if action_index == 3:
                offset = [self.spacing, 0, 0]

        elif self.num_actions == 6:
            if action_index == 0:
                offset = [0, 0, self.spacing]
            if action_index == 1:
                offset = [-self.spacing, 0, 0]
            if action_index == 2:
                offset = [0, -self.spacing, 0]
            if action_index == 3:
                offset = [0, self.spacing, 0]
            if action_index == 4:
                offset = [self.spacing, 0, 0]
            if action_index == 5:
                offset = [0, 0, -self.spacing]

        elif self.num_actions == 9:
            if action_index == 0:
                offset = [-self.spacing, -self.spacing, 0]
            if action_index == 1:
                offset = [-self.spacing, 0, 0]
            if action_index == 2:
                offset = [-self.spacing, self.spacing, 0]
            if action_index == 3:
                offset = [0, -self.spacing, 0]
            if action_index == 4:
                offset = [
                    0,
                    0,  # mask out action of standing still to enforce agent motion
                    0,
                ]
            if action_index == 5:
                offset = [0, self.spacing, 0]
            if action_index == 6:
                offset = [self.spacing, -self.spacing, 0]
            if action_index == 7:
                offset = [self.spacing, 0, 0]
            if action_index == 8:
                offset = [self.spacing, self.spacing, 0]

        elif self.num_actions == 27:
            if action_index == 0:
                offset = [-self.spacing, -self.spacing, self.spacing]
            if action_index == 1:
                offset = [-self.spacing, 0, self.spacing]
            if action_index == 2:
                offset = [-self.spacing, self.spacing, self.spacing]
            if action_index == 3:
                offset = [0, -self.spacing, self.spacing]
            if action_index == 4:
                offset = [0, 0, self.spacing]
            if action_index == 5:
                offset = [0, self.spacing, self.spacing]
            if action_index == 6:
                offset = [self.spacing, -self.spacing, self.spacing]
            if action_index == 7:
                offset = [self.spacing, 0, self.spacing]
            if action_index == 8:
                offset = [self.spacing, self.spacing, self.spacing]
            if action_index == 9:
                offset = [-self.spacing, -self.spacing, 0]
            if action_index == 10:
                offset = [-self.spacing, 0, 0]
            if action_index == 11:
                offset = [-self.spacing, self.spacing, 0]
            if action_index == 12:
                offset = [0, -self.spacing, 0]
            if action_index == 13:
                offset = [0, 0, 0]
            if action_index == 14:
                offset = [0, self.spacing, 0]
            if action_index == 15:
                offset = [self.spacing, -self.spacing, 0]
            if action_index == 16:
                offset = [self.spacing, 0, 0]
            if action_index == 17:
                offset = [self.spacing, self.spacing, 0]
            if action_index == 18:
                offset = [-self.spacing, -self.spacing, -self.spacing]
            if action_index == 19:
                offset = [-self.spacing, 0, -self.spacing]
            if action_index == 20:
                offset = [-self.spacing, self.spacing, -self.spacing]
            if action_index == 21:
                offset = [0, -self.spacing, -self.spacing]
            if action_index == 22:
                offset = [0, 0, -self.spacing]
            if action_index == 23:
                offset = [0, self.spacing, -self.spacing]
            if action_index == 24:
                offset = [self.spacing, -self.spacing, -self.spacing]
            if action_index == 25:
                offset = [self.spacing, 0, -self.spacing]
            if action_index == 26:
                offset = [self.spacing, self.spacing, -self.spacing]

        position = position + offset

        return position

    def apply_collision_mask(
        self, position, mask, next_other_positions, agent_state_space
    ):

        for other_position in next_other_positions:
            relative_idx = agent_state_space.position_to_index(
                other_position
            ) - agent_state_space.position_to_index(position)

            if self.num_actions == 4:
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    mask[0] = 0
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    mask[1] = 0
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    mask[2] = 0
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    mask[3] = 0

            if self.num_actions == 6:
                if relative_idx[0] == 0 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[0] = 0
                        mask[5] = 0
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[1] = 0
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    if np.sum(mask) > 1:
                        mask[2] = 0
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    if np.sum(mask) > 1:
                        mask[3] = 0
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[4] = 0

            elif self.num_actions == 9:
                if relative_idx[0] == -1 and relative_idx[1] == -1:
                    mask[0] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[0] = 1
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    mask[1] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[1] = 1
                if relative_idx[0] == -1 and relative_idx[1] == 1:
                    mask[2] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[2] = 1
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    mask[3] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[3] = 1
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    mask[5] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[5] = 1
                if relative_idx[0] == 1 and relative_idx[1] == -1:
                    mask[6] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[6] = 1
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    mask[7] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[7] = 1
                if relative_idx[0] == 1 and relative_idx[1] == 1:
                    mask[8] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[8] = 1

            elif self.num_actions == 27:
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[4] = 0
                    mask[22] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 0
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 0
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0

                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[4] = 0
                    mask[22] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0

        return mask
    
    def apply_obstacle_mask(self, position, mask, agent_state_space):
        """
        应用障碍物掩码，屏蔽会导致与障碍物碰撞的动作
        
        Args:
            position: 当前位置 [x, y, z]
            mask: 当前动作掩码（一维数组）
            agent_state_space: 智能体状态空间
            
        Returns:
            更新后的动作掩码（一维数组）
        """
        if self.obstacle_manager is None or not self.obstacle_manager.enabled:
            return mask
        
        # 确保mask是一维数组
        mask_1d = mask.flatten() if mask.ndim > 1 else mask
        
        # 获取所有可能的下一步位置
        possible_next_positions = []
        for action in range(self.num_actions):
            next_pos = self.action_to_position(np.copy(position), action)
            possible_next_positions.append(next_pos)
        
        # 获取安全动作掩码
        obstacle_mask = self.obstacle_manager.get_safe_actions_mask(
            position, possible_next_positions
        )
        
        # 确保obstacle_mask也是一维数组且长度匹配
        if obstacle_mask.ndim > 1:
            obstacle_mask = obstacle_mask.flatten()
        
        # 确保两个掩码长度相同
        if len(mask_1d) != len(obstacle_mask):
            logger.error(f"Mask shape mismatch: mask_1d={mask_1d.shape}, obstacle_mask={obstacle_mask.shape}")
            logger.error(f"num_actions={self.num_actions}, position={position}")
            # 返回原始掩码，不应用障碍物掩码
            return mask_1d
        
        # 合并掩码（两个掩码都为1时才为1）
        combined_mask = mask_1d * obstacle_mask
        
        # 如果所有动作都被屏蔽，选择"最不坏"的动作（离障碍物最远的）
        if np.sum(combined_mask) == 0:
            logger.warning(f"All actions blocked by obstacles at position {position}, "
                          f"choosing least bad action (farthest from obstacles)")
            
            # 计算每个有效动作到最近障碍物的距离
            valid_actions = np.where(mask_1d == 1)[0]
            if len(valid_actions) == 0:
                # 如果原始掩码就是空的，返回原掩码
                return mask_1d
            
            max_min_distance = -1
            best_action = None
            
            for action in valid_actions:
                next_pos = possible_next_positions[action]
                # 获取到最近障碍物的距离
                min_distance = self.obstacle_manager.get_nearest_obstacle_distance(next_pos)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_action = action
            
            # 创建新掩码，只允许"最不坏"的动作
            if best_action is not None:
                escape_mask = np.zeros_like(mask_1d)
                escape_mask[best_action] = 1
                logger.info(f"Escape action {best_action} selected with distance {max_min_distance:.2f}m from nearest obstacle")
                return escape_mask
            else:
                # 理论上不应该到这里，但保险起见
                logger.warning(f"Could not find escape action, keeping original mask")
                return mask_1d
        
        return combined_mask
