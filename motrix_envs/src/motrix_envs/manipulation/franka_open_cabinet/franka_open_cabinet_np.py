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
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import FrankaOpenCabinetEnvCfg


@registry.env("franka-open-cabinet", "np")
class FrankaOpenCabinetEnv(NpEnv):
    _cfg: FrankaOpenCabinetEnvCfg

    def __init__(self, cfg: FrankaOpenCabinetEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.robot_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.robot_default_joint_pos = np.array(
            [
                0.0 * np.pi,
                -30 / 180 * np.pi,
                0 * np.pi,
                -156 / 180 * np.pi,
                0.0 * np.pi,
                186 / 180 * np.pi,
                -45 / 180 * np.pi,
                0.04,
                0.04,
            ],
            np.float32,
        )

        self._action_dim = 8
        self._obs_dim = 25  # 8 + 8 + 7 + 1 + 1
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9  # self._model.num_dof_pos # 9
        self._num_dof_vel = 9  # self._model.num_dof_vel # 9
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)
        # Initialize properties
        self.robot = self._model.get_body("link0")
        self.gripper_tcp = self._model.get_site("gripper")
        self.left_finger_pad = self._model.get_geom("left_finger_pad")
        self.right_finger_pad = self._model.get_geom("right_finger_pad")
        self.robot_joint_pos_min_limit = self._model.actuator_ctrl_limits[0]
        self.robot_joint_pos_max_limit = self._model.actuator_ctrl_limits[1]

        self.drawer_top_joint = self._model.get_joint("drawer_top_joint")
        self.drawer_top_handle = self._model.get_site("drawer_top_handle")

        self.count = 0
        # Set print options to 2 decimal places
        np.set_printoptions(precision=2)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        assert not np.isnan(actions).any(), "actions contain nan"

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # no gripper
        old_joint_pos = self.get_robot_joint_pos(state.data)[:, : self._action_dim - 1]
        new_joint_pos = actions[:, : self._action_dim - 1] + old_joint_pos  # action as offset

        # with gripper
        # 1. Map to probability p using Sigmoid
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. Bernoulli sampling - probability can sample different results
        # np.random.uniform(0, 1, size) generates random number r ~ U(0, 1) for each environment
        # If r < p, result is 1 (success/grasp), otherwise 0 (failure/release)
        sampled_gripper_action = np.where(probabilities > np.random.rand(*probabilities.shape), 0, 0.04)[
            :, None
        ]  # 0 for closed, 0.04 for open
        state.info["current_gripper_action"] = sampled_gripper_action.squeeze()

        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # step action
        cliped_new_pos = np.clip(
            new_pos, self.robot_joint_pos_min_limit, self.robot_joint_pos_max_limit, dtype=np.float32
        )  # clip new pos to limit

        # actuator1~8 by order
        state.data.actuator_ctrls = cliped_new_pos

        return state

    def update_state(self, state: NpEnvState):
        # compute obs
        obs = self._compute_observation(state.data, state.info)
        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.obs = obs
        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]

        noise_pos = np.random.uniform(
            -0.125,  # -cfg.reset_noise_scale,
            0.125,  # cfg.reset_noise_scale,
            (num_reset, self._num_dof_pos),
        )

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1)) + noise_pos  # Add noise in range [-0.125, 0.125]
        data.reset(self._model)
        data.set_dof_vel(np.zeros((num_reset, 13), dtype=np.float32))  # Includes robot and cabinet
        data.set_dof_pos(np.concatenate([dof_pos, np.zeros((num_reset, 4), dtype=np.float32)], axis=-1), self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._action_dim), dtype=np.float32),
            "phase2_mask": np.zeros(num_reset, dtype=bool),  # 1D array
            "current_gripper_action": np.zeros(num_reset, dtype=np.float32),  # 1D array
        }
        obs = self._compute_observation(data, info)
        return obs, info

    def _compute_observation(self, data: mtx.SceneData, info: dict):
        num_envs = data.shape[0]

        # dof_pos: (num_envs, 8) range: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(data)  # shape: (num_envs, 8)
        dof_pos_rel = self._get_robot_joint_pos_rel(dof_pos)[:, : self._action_dim]

        dof_lower_limits = np.tile(self.robot_joint_pos_min_limit, (num_envs, 1))
        dof_upper_limits = np.tile(self.robot_joint_pos_max_limit, (num_envs, 1))

        dof_pos_scaled = 2.0 * dof_pos_rel / (dof_upper_limits - dof_lower_limits) - 1.0
        # relative vel: (num_envs, 8) range approximately (-pi ~ pi) / 2 (divided by 2 for smaller values)
        dof_vel = self.get_robot_joint_vel(data)
        dof_vel_rel = self._get_robot_joint_vel_rel(dof_vel)[:, : self._action_dim] / 2

        # relative orientation: (num_envs, 1)
        robot_grasp_pose = self.gripper_tcp.get_pose(data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(data)
        to_target = drawer_grasp_pose - robot_grasp_pose

        # Cabinet joint
        drawer_top_joint_pos = self.drawer_top_joint.get_dof_pos(data)  # shape: (num_envs, 1)
        drawer_top_joint_vel = self.drawer_top_joint.get_dof_vel(data)  # shape: (num_envs, 1)

        obs = np.concatenate(
            [dof_pos_scaled, dof_vel_rel, to_target, drawer_top_joint_pos, drawer_top_joint_vel], axis=-1
        )

        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        return np.clip(obs, -5, 5)

    def _compute_reward(self, state: NpEnvState, truncated: np.ndarray):
        robot_grasp_pose = self.gripper_tcp.get_pose(state.data)
        drawer_grasp_pose = self.drawer_top_handle.get_pose(state.data)

        gripper_drawer_dist = np.linalg.norm(drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1)

        ## distance reward
        std = 0.1
        dist_reward = 1 - np.tanh(gripper_drawer_dist / std)
        dist_reward *= 10

        ## matching orientation reward
        quat_reward = Quaternion.similarity(robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:])

        ## close gripper reward
        # When gripper distance < 0.025, closing gripper gets reward
        # When gripper distance > 0.025, closing gripper gets penalty
        # When gripper distance > 0.025 or < 0.025, opening gripper gets no reward
        open_gripper = np.where(gripper_drawer_dist < 0.025, 100.0, -20) * (
            0.04 - state.info["current_gripper_action"]
        )  # dist_reward * 0 or 0.04

        ## open drawer reward
        open_dist = self.drawer_top_joint.get_dof_pos(state.data).squeeze()
        open_dist = np.clip(open_dist, 0, 1)
        open_reward = (np.exp(open_dist) - 1) * 20

        wrong_open = np.logical_and(
            open_dist > 0, gripper_drawer_dist > 0.03
        )  # Drawer opened but gripper not on handle
        open_reward = (
            np.bitwise_not(wrong_open) * open_reward
        )  # No reward for forced opening (can't force open after increasing MJCF resistance)

        ##################### Penalty Terms #####################"
        ## Action penalty
        ## Joint velocity penalty - sometimes some joints rotate more while others rotate less
        action_penalty = np.sum(np.square(state.info["current_actions"] - state.info["last_actions"]), axis=-1)
        joint_vel_penalty = np.sum(np.square(state.data.dof_vel[:, : self._action_dim]), axis=-1)

        ## finger position penalty
        lfinger_dist = self.left_finger_pad.get_pose(state.data)[:, 2] - drawer_grasp_pose[:, 2]
        rfinger_dist = drawer_grasp_pose[:, 2] - self.right_finger_pad.get_pose(state.data)[:, 2]
        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0, lfinger_dist, np.zeros_like(lfinger_dist))
        finger_dist_penalty += np.where(rfinger_dist < 0, rfinger_dist, np.zeros_like(rfinger_dist))

        ##################### Coefficient Schedule #####################"

        ## action penalty rate
        if self.count < 8000:
            action_penalty_rate = 1e-3
            joint_vel_penalty_rate = 0 * 10  # Keep very small at the beginning
        else:
            action_penalty_rate = 2e-3
            joint_vel_penalty_rate = 2e-7

        ##################### Reward Calculation #####################"

        step2_reward = dist_reward + quat_reward + open_gripper + open_reward + finger_dist_penalty

        # Final reward
        reward = step2_reward - action_penalty_rate * action_penalty - joint_vel_penalty_rate * joint_vel_penalty

        # Apply truncation penalty
        reward = np.where(truncated, reward - np.array(10.0), reward)

        return reward

    def _check_termination(self, state: NpEnvState):
        # Check if robot arm extends too far forward causing collision
        robot_grasp_pos_x = self.gripper_tcp.get_pose(state.data)[:, 0]
        drawer_grasp_pos_x = self.drawer_top_handle.get_pose(state.data)[:, 0]
        truncated = robot_grasp_pos_x - drawer_grasp_pos_x < -0.03

        # Check that joint velocity doesn't exceed threshold of 5 rad/s
        joint_vel = self.get_robot_joint_vel(state.data)
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 5)
        return truncated

    def get_robot_joint_pos(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_pos(data)[:, : self._num_dof_pos]

    def get_robot_joint_vel(self, data: mtx.SceneModel):
        return self.robot.get_joint_dof_vel(data)[:, : self._num_dof_pos]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.robot_default_joint_pos

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel
