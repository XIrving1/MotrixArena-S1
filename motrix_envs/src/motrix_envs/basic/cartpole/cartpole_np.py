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

from .cfg import CartPoleEnvCfg


@registry.env("cartpole", "np")
class CartPoleEnv(NpEnv):
    _cfg: CartPoleEnvCfg

    def __init__(self, cfg: CartPoleEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        state.data.actuator_ctrls = actions
        return state

    def update_state(self, state: NpEnvState):
        # compute observation
        data = state.data
        dof_pos = data.dof_pos
        dof_vel = data.dof_vel
        obs = np.concatenate([dof_pos, dof_vel], axis=-1)
        assert obs.shape == (self._num_envs, 4)

        # compute reward
        reward = np.ones((self._num_envs,), dtype=np.float32)

        # compute terminated
        cart_pos = dof_pos[:, 0]
        angle = dof_pos[:, 1]
        terminated = np.logical_or(np.isnan(angle), np.abs(angle) > 0.2)
        terminated = np.logical_or(cart_pos < -0.8, terminated)
        terminated = np.logical_or(cart_pos > 0.8, terminated)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

    def reset(self, data: mtx.SceneData):
        cfg: CartPoleEnvCfg = self._cfg
        noise_pos = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (*data.shape, self._num_dof_pos),
        )
        noise_vel = np.random.uniform(
            -cfg.reset_noise_scale,
            cfg.reset_noise_scale,
            (*data.shape, self._num_dof_vel),
        )

        dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1)) + noise_pos
        dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1)) + noise_vel

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        obs = np.concatenate([dof_pos, dof_vel], axis=-1)
        return obs, {}
