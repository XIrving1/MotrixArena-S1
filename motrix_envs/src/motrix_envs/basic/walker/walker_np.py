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
from motrix_envs.basic.walker.cfg import WalkerEnvCfg
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("dm-walker", "np")
@registry.env("dm-runner", "np")
@registry.env("dm-stander", "np")
class Walker2DEnv(NpEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: WalkerEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_obs_space()
        self._init_action_space()
        self._torso = self._model.get_link("torso")
        self._move_speed = cfg.move_speed
        self._joint_limits = self._model.joint_limits
        self._stand_height = cfg.stand_height

    def _init_obs_space(self):
        model = self._model
        num = 0
        num += (model.num_links - 1) * 2  # planar rotation (x,z) for each link except the root
        num += 1  # torso height
        num += model.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num,), dtype=np.float32)

    def _init_action_space(self):
        model = self._model

        self._action_space = gym.spaces.Box(
            model.actuator_ctrl_limits[0],
            model.actuator_ctrl_limits[1],
            (model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions, state):
        state.data.actuator_ctrls = actions
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        num_env = data.shape[0]
        link_rotations = self._model.get_link_rotation_mats(data)
        dof_vel = data.dof_vel
        up_right = link_rotations[:, 0, 2, 2].reshape(num_env, 1)  # 1
        orientations = link_rotations[:, 1:, [0, 0], [2, 0]].reshape(num_env, -1)  # (num_links - 1) * 2
        obs = np.concatenate([orientations, up_right, dof_vel], axis=-1)
        return obs

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data

        # === compute obs ====
        obs = self._get_obs(data)
        offset = (self._model.num_links - 1) * 2 + 1
        dof_vel = obs[:, offset:]

        torso_upright = self._torso.get_rotation_mat(data)[:, 2, 2]
        torso_height = self._torso.get_position(data)[:, 2]
        torso_vel = self._model.get_sensor_value("torso_subtreelinvel", data)
        horizontal_velocity = torso_vel[:, 0]

        # ==== compute terminated
        terminated = np.isnan(dof_vel).any(axis=-1)

        # ==== compute reward
        rwd_height = reward.tolerance(
            torso_height,
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height * 4 / 5,
        )
        rwd_upright = (1 + torso_upright) / 2
        rwd_stand = (3 * rwd_height + 1 * rwd_upright) / 4

        rwd = rwd_stand

        state.info["Reward"] = {
            "height": rwd_height,
            "upright": rwd_upright,
            "stand": rwd_stand,
        }

        if self._move_speed > 0.0:
            rwd_move = reward.tolerance(
                horizontal_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            state.info["Reward"]["move"] = rwd_move
            rwd = rwd_stand * (5 * rwd_move + 1) / 6

        rwd[terminated] = 0.0

        return state.replace(
            obs=obs,
            reward=rwd,
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num_reset = data.shape[0]

        dof_pos = np.zeros((num_reset, self._model.num_dof_pos))
        dof_pos[:, 2] = np.random.uniform(low=-np.pi, high=np.pi, size=(num_reset,))  # randomize root yaw
        dof_pos[:, 3:] = np.random.uniform(
            low=self._joint_limits[0, 3:],
            high=self._joint_limits[1, 3:],
            size=(num_reset, self._model.num_dof_pos - 3),
        )  # randomize other joint angles
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)
        obs = self._get_obs(data)
        rewards = {
            "height": np.zeros((num_reset,)),
            "upright": np.zeros((num_reset,)),
            "stand": np.zeros((num_reset,)),
        }
        if self._move_speed > 0.0:
            rewards["move"] = np.zeros((num_reset,))

        return obs, {"Reward": rewards}
