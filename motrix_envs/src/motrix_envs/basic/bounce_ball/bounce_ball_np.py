# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import BounceBallEnvCfg


@registry.env("bounce_ball", "np")
class BounceBallEnv(NpEnv):
    _cfg: BounceBallEnvCfg

    def __init__(self, cfg: BounceBallEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # Action space: 6D normalized paddle velocity (dx, dy, dz, dr_x, dr_y, dr_z)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        # Observation space: simplified version using only DOF information
        # DOF pos (13) + DOF vel (12) = 25 (this includes ball state implicitly)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        # Initial arm joint positions (degrees converted to radians)
        self._init_arm_qpos = np.array(self._cfg.arm_init_qpos, dtype=np.float32) * np.pi / 180.0
        self._init_dof_vel = np.zeros(self._model.num_dof_vel, dtype=np.float32)

        # Initialize full DOF positions (6 arm joints + 7 for ball free joint)
        self._init_dof_pos = np.zeros(self._model.num_dof_pos, dtype=np.float32)
        self._init_dof_pos[:6] = self._init_arm_qpos

        # Get body and geom IDs
        self._paddle_geom_id = self._model.geom_names.index("blocker")
        self._ball_body_id = self._model.body_names.index("ball_link")

        # Action scaling parameters
        self._action_scale = np.array(self._cfg.action_scale, dtype=np.float32)
        self._action_bias = np.array(self._cfg.action_bias, dtype=np.float32)

        # Track ball initial position for reset
        self._ball_init_pos = np.array(self._cfg.ball_init_pos, dtype=np.float32)
        self._ball_init_vel = np.array(self._cfg.ball_init_vel, dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action to get actual paddle velocity changes"""
        return self._action_scale * action + self._action_bias

    def _compute_observation(self, data: mtx.SceneData) -> np.ndarray:
        """Compute 25-dimensional observation vector from DOF states"""
        # Use DOF positions and velocities directly
        dof_pos = data.dof_pos
        dof_vel = data.dof_vel

        # Concatenate DOF positions (13) and velocities (12)
        obs = np.concatenate([dof_pos, dof_vel], axis=-1)
        return obs.astype(np.float32)

    def _compute_reward(
        self, obs: np.ndarray, data: mtx.SceneData = None, consecutive_bounces: np.ndarray = None
    ) -> np.ndarray:
        """Compute reward based on ball height, position, and controlled upward velocity"""
        # Extract ball position and velocity from DOF
        ball_x = obs[:, 6]  # Ball x position
        ball_z = obs[:, 8]  # Ball z position

        ball_vz = obs[:, 13 + 8]  # Ball z velocity (13 pos + 8 vel)

        # Target positions
        target_ball_x = 0.58856  # Target x position
        target_height = self._cfg.target_ball_height
        tolerance = self._cfg.height_tolerance

        # 1. Position control reward - MOST IMPORTANT for keeping ball centered
        # Strong reward for ball being at the right x position (paddle center)
        x_position_error = np.abs(ball_x - target_ball_x)
        x_position_reward = np.exp(-(x_position_error**2) / (2 * 0.05**2))  # Tight tolerance for x position

        # 2. Height-based reward - less important than position control
        height_error = np.abs(ball_z - target_height)
        height_reward = np.exp(-(height_error**2) / (2 * tolerance**2))

        # 3. Controlled upward velocity reward - only when ball is in good position
        # Only reward upward velocity when ball is well-positioned horizontally
        well_positioned = x_position_error < 0.02  # Ball must be very close to target x
        controlled_upward_reward = np.where(
            well_positioned & (ball_vz > 0.1) & (ball_vz < 1.5),  # Reasonable upward velocity
            np.clip(ball_vz * 1.5, 0.0, 1.5),  # Reduced scale
            0.0,
        )

        # 4. Strong penalty for being out of position horizontally
        out_of_position_penalty = np.where(
            x_position_error > 0.1,
            -2.0,  # Heavy penalty for being far from center
            0.0,
        )

        # 5. Velocity penalties - discourage excessive speeds
        excessive_upward_penalty = np.where(ball_vz > 2.0, -1.0, 0.0)

        downward_velocity_penalty = np.where(ball_vz < -0.5, -np.clip(-ball_vz * 0.3, 0.0, 0.5), 0.0)

        # 6. Position-based penalties (reduced)
        overshoot_penalty = np.where(ball_z > target_height + tolerance, -0.3, 0.0)
        undershoot_penalty = np.where(ball_z < 0.1, -0.5, 0.0)

        # 7. Consecutive bounces reward - only when position is good
        good_position_for_bounce = x_position_error < 0.05  # Reasonable position for bouncing
        consecutive_bounces_reward = np.where(
            good_position_for_bounce & (consecutive_bounces > 0),
            np.log(consecutive_bounces + 1) * 0.3,  # Reduced scale
            0.0,
        )

        # Bonus for high bounce counts (only when well-positioned)
        high_bounce_bonus = np.where(
            good_position_for_bounce & (consecutive_bounces >= 3),
            consecutive_bounces * 0.1,  # Reduced bonus
            0.0,
        )

        # Combine all rewards with corrected priorities
        total_reward = (
            x_position_reward * 2.0  # X position control (200%) - MOST IMPORTANT
            + controlled_upward_reward * 1.0  # Controlled upward velocity (100%)
            + height_reward * 0.3  # Height accuracy (30%) - less important
            + consecutive_bounces_reward * 1.0  # Consecutive bounces (100%) - reduced
            + high_bounce_bonus * 0.3  # High bounce bonus (30%) - reduced
            + out_of_position_penalty * 1.0  # Out of position penalty (100%)
            + excessive_upward_penalty * 1.0  # Excessive upward penalty (100%)
            + downward_velocity_penalty * 1.0  # Downward penalty (100%)
            + overshoot_penalty  # Height overshoot penalty
            + undershoot_penalty  # Height undershoot penalty
        )

        return total_reward

    def _compute_terminated(self, obs: np.ndarray) -> np.ndarray:
        """Check if episode should terminate based on DOF states"""
        # Extract ball position from DOF (indices 6-8 for x,y,z)
        ball_x = obs[:, 6]  # Ball x position
        ball_z = obs[:, 8]  # Ball z position

        # Target height from config
        target_height = self._cfg.target_ball_height

        # Terminate if ball falls below ground or goes significantly higher than target
        terminated = (ball_z < 0.05) | (ball_z > target_height + 1.0)

        # Also terminate if ball goes too far horizontally
        terminated |= np.abs(ball_x) > 1.5

        return terminated

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply action to control paddle position"""
        # Get current joint positions
        current_joint_pos = state.data.dof_pos[:, :6]  # First 6 DOFs are arm joints

        # Denormalize actions to get actual position changes
        delta_positions = self._denormalize_action(actions)

        # Calculate target positions = current positions + position changes
        target_positions = current_joint_pos + delta_positions

        # Apply target positions as actuator controls (position control)
        state.data.actuator_ctrls = target_positions
        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Update state with new observations, rewards, and termination flags"""
        data = state.data

        # Compute observation
        obs = self._compute_observation(data)

        # Get bounce tracking from info
        consecutive_bounces = state.info.get("consecutive_bounces", np.zeros(data.shape[0], dtype=np.int32))
        ball_was_upward = state.info.get("ball_was_upward", np.zeros(data.shape[0], dtype=bool))

        # Detect bounces and update consecutive bounce count
        current_ball_z = obs[:, 8]  # Ball z position
        current_ball_vz = obs[:, 21]  # Ball z velocity

        # Detect bounces: ball moving upward after being near paddle height
        near_paddle = (current_ball_z < 0.4) & (current_ball_z > 0.15)
        moving_upward = current_ball_vz > 0.01

        # A bounce is detected when ball was going down and now goes up near paddle height
        bounce_detected = ~ball_was_upward & moving_upward & near_paddle

        # Update consecutive bounce count
        consecutive_bounces = np.where(bounce_detected, consecutive_bounces + 1, consecutive_bounces)

        # Reset count if ball is falling too much (not bouncing properly)
        falling = (current_ball_vz < -0.5) & (current_ball_z < 0.4)
        consecutive_bounces = np.where(falling, 0, consecutive_bounces)

        # Update tracking variables in info
        state.info["consecutive_bounces"] = consecutive_bounces
        state.info["ball_was_upward"] = moving_upward

        # Track maximum bounces achieved
        max_current = np.max(consecutive_bounces)
        if "max_consecutive_bounces" not in state.info:
            state.info["max_consecutive_bounces"] = 0
        if max_current > state.info["max_consecutive_bounces"]:
            state.info["max_consecutive_bounces"] = max_current

        # For simplicity, use raw observation without normalization for now
        # Could add proper normalization later
        normalized_obs = obs

        # Compute reward and termination
        reward = self._compute_reward(obs, data, consecutive_bounces)
        terminated = self._compute_terminated(obs)

        state.obs = normalized_obs
        state.reward = reward
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData) -> tuple:
        """Reset environment to initial state"""
        cfg: BounceBallEnvCfg = self._cfg
        num_reset = data.shape[0]

        # Add noise to initial arm joint positions only (not ball)
        arm_noise_pos = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (num_reset, 6),  # Only 6 arm joints
        )
        noise_vel = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (num_reset, self._num_dof_vel),
        )

        # Reset simulation first to get proper DOF structure
        data.reset(self._model)

        # Get current DOF positions and modify only the arm joints
        current_dof_pos = data.dof_pos
        current_dof_vel = data.dof_vel

        # Set arm joint positions (first 6 DOFs)
        current_dof_pos[:, :6] = np.tile(self._init_arm_qpos, (num_reset, 1)) + arm_noise_pos
        current_dof_vel[:, :6] = noise_vel[:, :6]

        # Set the quaternion part properly (DOFs 9-12 are quaternion w,x,y,z for freejoint)
        # The ball has a freejoint which uses quaternion representation
        for i in range(num_reset):
            # Set quaternion for ball (indices 9-12: w, x, y, z)
            current_dof_pos[i, 9:13] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion

        data.set_dof_pos(current_dof_pos, self._model)
        data.set_dof_vel(current_dof_vel)

        # Set ball position in DOF (indices 6-8 for x, y, z positions)
        for i in range(num_reset):
            ball_noise_pos = np.random.uniform(-0.01, 0.01, 3)
            ball_pos = self._ball_init_pos + ball_noise_pos
            # Set ball position in DOF coordinates (indices 6-8)
            current_dof_pos[i, 6:9] = ball_pos

        # Final update to set both ball position and quaternion
        data.set_dof_pos(current_dof_pos, self._model)

        # Initialize info dict with bounce tracking variables
        info = {
            "consecutive_bounces": np.zeros(num_reset, dtype=np.int32),
            "ball_was_upward": np.zeros(num_reset, dtype=bool),
            "max_consecutive_bounces": 0,
        }

        # Compute initial observation
        obs = self._compute_observation(data)
        normalized_obs = obs  # No normalization for now

        return normalized_obs, info
