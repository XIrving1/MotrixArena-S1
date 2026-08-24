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

from .cfg import FrankaLiftCubeEnvCfg

# Decay parameters (constants, can be defined during class initialization)
START_EPSILON = 1.0  # Initial value
MIN_EPSILON = 0.05  # Minimum value (typically 0.01 or 0.05)
# Assume we want to complete decay in half of total steps (12000 steps)
END_STEP = 12000


@registry.env("franka-lift-cube", "np")
class FrankaLiftCubeEnv(NpEnv):
    _cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.default_joint_pos = self._cfg.init_state.default_joint_pos

        self._action_dim = 8
        self._obs_dim = 36  # 9 + 9 + 3 + 7 + 8
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9  # self._model.num_dof_pos # 9
        self._num_dof_vel = 9  # self._model.num_dof_vel # 9
        self._init_dof_pos = self.default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)

        self._cube = self._model.get_geom("cube")
        self._body = self._model.get_body("link0")

        self.hand = self._model.get_site("gripper")

        self.joint_pos_min_limit = self._cfg.control_config.min_pos
        self.joint_pos_max_limit = self._cfg.control_config.max_pos

        self.epsilon = START_EPSILON

        self._state_for_render = None

        self.count = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # no gripper
        old_joint_pos = self.get_dof_pos(state.data)[:, : self._action_dim - 1]
        new_joint_pos = actions[:, : self._action_dim - 1] + old_joint_pos  # action as offset

        # with gripper
        # 1. Map to probability p (using Sigmoid)
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. Bernoulli sampling - probability always has chance to sample different results
        # np.random.uniform(0, 1, size) generates a random number r ~ U(0, 1) for each environment
        # If r < p, result is 1 (success/grasp), otherwise 0 (failure/release)
        sampled_gripper_action = np.where(probabilities > np.random.rand(*probabilities.shape), 0, 0.04)[
            :, None
        ]  # Close 0, Open 0.04
        state.info["current_gripper_action"] = sampled_gripper_action.squeeze(axis=-1)

        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # step action
        cliped_new_pos = np.clip(
            new_pos, self.joint_pos_min_limit, self.joint_pos_max_limit, dtype=np.float32
        )  # clip new pos to limit

        state.data.actuator_ctrls = cliped_new_pos

        return state

    def update_state(self, state: NpEnvState):
        self._state_for_render = state
        # compute observation
        obs = self._compute_observation(state.data, state.info)

        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.obs = obs
        state.reward = reward
        state.terminated = truncated  # np.logical_or(truncated, done)

        self.count += 1

        return state

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]

        # Robot arm initial joint angle noise
        noise_pos = np.random.uniform(
            -self._cfg.init_state.joint_pos_reset_noise_scale,
            self._cfg.init_state.joint_pos_reset_noise_scale,
            self._num_dof_pos,
        )
        robot_dof_pos = self._init_dof_pos + noise_pos

        # Domain randomization for cube position
        # x -0.1, 0.1
        # y -0.25, 0.25
        x_low, x_high = -0.1, 0.1
        y_low, y_high = -0.25, 0.25
        pos_x = np.random.uniform(x_low, x_high)
        pos_y = np.random.uniform(y_low, y_high)

        scene_dof_pos = np.concatenate(
            [robot_dof_pos, np.array([pos_x, pos_y, 0.05, 1, 0, 0, 0], dtype=np.float32)]
        )  # Added cube
        scene_dof_pos = np.tile(scene_dof_pos, (num_reset, 1))

        scene_dof_vel = np.concatenate([self._init_dof_vel, np.zeros(6, dtype=np.float32)])
        scene_dof_vel = np.tile(scene_dof_vel, (num_reset, 1))

        # Reset
        data.reset(self._model)
        data.set_dof_vel(scene_dof_vel)
        data.set_dof_pos(scene_dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "commands": self._generated_commands(num_reset),  #
            "current_gripper_action": np.zeros(num_reset, dtype=np.float32),  # 1D
        }

        # Check for nan
        assert not np.isnan(info["commands"]).any(), "commands contain nan"

        obs = self._compute_observation(data, info)
        return obs, info

    def _compute_observation(self, data: mtx.SceneData, info: dict):
        dof_pos = self.get_dof_pos(data)  # shape: # not necessarily (self.num_envs, 9)
        dof_vel = self.get_dof_vel(data)  # shape: # not necessarily (num_envs, 9)
        dof_pos_rel = self._get_joint_pos_rel(dof_pos)
        dof_vel_rel = self._get_joint_vel_rel(dof_vel)

        object_pick_pose = self._cube.get_pose(data)

        object_lift_pos = info["commands"]

        last_actions = info["current_actions"]

        obs = np.concatenate([dof_pos_rel, dof_vel_rel, object_pick_pose, object_lift_pos, last_actions], axis=-1)

        assert obs.shape == (data.shape[0], self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        return obs.astype(np.float32)

    def _check_termination(self, state: NpEnvState):
        cube_height = self._cube.get_pose(state.data)[:, 2]
        truncated = cube_height < -0.05  # New truncated condition

        # Check joint velocity is not too large (set to 5 radians per second here)
        joint_vel = self.get_dof_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 10)

        # Check cube velocity
        cube_vel = self._cube.get_linear_velocity(state.data)  # shape = (*data.shape, 3).
        truncated = np.logical_or(truncated, np.abs(cube_vel).max(axis=-1) > 10)
        return truncated

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray):
        hand_pose = self.hand.get_pose(state.data)
        hand_pos = hand_pose[:, :3]
        cube_pos = self._cube.get_pose(state.data)[:, :3]

        # reach reward
        hand_cube_distance = np.linalg.norm(cube_pos - hand_pos, axis=-1)

        std = 0.1
        reach_reward = 1 - np.tanh(hand_cube_distance / std)

        # lift reward
        lift_height = cube_pos[:, 2]  # Cube center of mass height - initial center of mass height 0.02 = lift height
        minimal_height = 0.04  # 4cm height limit
        lifted = lift_height > minimal_height

        # object_command_tracking reward
        object_command_dist = np.linalg.norm(cube_pos - state.info["commands"], axis=-1)

        def shifted_sigmoid_reward(d, k=8, center=0.3):
            # Sigmoid(-k * (d - center))
            # The larger d is, the more positive (d-center) is, the more negative -k*(...) is, Sigmoid closer to 0
            # The smaller d is, the more negative (d-center) is, the more positive -k*(...) is, Sigmoid closer to 1
            x = -k * (d - center)
            return 1 / (1 + np.exp(-x))

        object_command_tracking_reward = (
            shifted_sigmoid_reward(object_command_dist) * (lift_height > 0.04) * (hand_cube_distance < 0.02)
        )

        object_command_tracking_fine_graind_reward = (
            (1 - np.tanh(object_command_dist / 0.4)) * (lift_height > 0.04) * (hand_cube_distance < 0.02)
        )

        object_command_tracking_close_reward = (
            (1 - np.tanh(object_command_dist / 0.05)) * (object_command_dist < 0.2) * (hand_cube_distance < 0.02)
        )

        # action_diff_sq: Sum of squares of action changes
        action_diff_sq = np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=-1)
        # joint_vel_sq: Sum of squares of joint velocities
        joint_vel_sq = np.sum(np.square(self.get_dof_vel(state.data)[:, : self._num_dof_vel]), axis=1)

        ## action penalty rate
        reach_weight = 1.5  # Cannot be too small
        cmd_tracking_weight = 10
        cmd_tracking_fine_graind_weight = 20  # Should be larger, need strong pull to target area
        object_command_tracking_close_reward_weight = 10

        if self.count < 10000:
            action_penalty_rate = 1e-4
            joint_vel_penalty_rate = 1e-4
        else:
            action_penalty_rate = 1e-1
            joint_vel_penalty_rate = 1e-1

        reward = (
            reach_weight * reach_reward
            + 30 * lifted * (hand_cube_distance < 0.05)
            + (cmd_tracking_weight * object_command_tracking_reward) ** 2
            + (cmd_tracking_fine_graind_weight * object_command_tracking_fine_graind_reward) ** 2
            + (object_command_tracking_close_reward_weight * object_command_tracking_close_reward) ** 2
            + 200 * object_command_tracking_close_reward
            + -action_penalty_rate * action_diff_sq
            + -joint_vel_penalty_rate * joint_vel_sq
        )

        return reward

    def get_dof_pos(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_vel(data)

    def _get_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.default_joint_pos

    def _get_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel

    def _generated_commands(self, num_envs: int):
        # Command is the final object_pose that cube should reach
        x_low, x_high = self._cfg.command_config.target_pos_x
        y_low, y_high = self._cfg.command_config.target_pos_y
        z_low, z_high = self._cfg.command_config.target_pos_z

        pos_x = np.random.uniform(x_low, x_high, num_envs)
        pos_y = np.random.uniform(y_low, y_high, num_envs)
        pos_z = np.random.uniform(z_low, z_high, num_envs)
        command_cube_target_pos = np.stack([pos_x, pos_y, pos_z], axis=-1)

        assert not np.isnan(command_cube_target_pos).any(), "command_cube_target_pos contain nan"
        return command_cube_target_pos
