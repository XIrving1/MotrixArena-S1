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
from motrix_envs.basic.cheetah.cfg import CheetahEnvCfg
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("dm-cheetah", "np")
class CheetahEnv(NpEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: CheetahEnvCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_obs_space()
        self._init_action_space()
        self._torso = self._model.get_link("torso")
        self._run_speed = cfg.run_speed
        self._joint_limits = self._model.joint_limits

    def _init_obs_space(self):
        obs_dim = (self._model.num_dof_pos - 1) + self._model.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float64)

    def _init_action_space(self):
        model = self._model
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions, state):
        state.data.actuator_ctrls = actions
        return state

    def _get_obs(self, data: mtx.SceneData) -> np.ndarray:
        qpos = data.dof_pos
        pos = qpos[:, 1:].copy()  # exclude x position
        vel = data.dof_vel
        obs = np.concatenate([pos, vel], axis=-1)
        return obs

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data

        # === compute obs ===
        obs = self._get_obs(data)

        # === Terminated ===
        terminated = np.zeros(data.shape[0], dtype=bool)

        # ==== compute reward ====
        vel = self._model.get_sensor_value("torso_subtreelinvel", data)
        rwd_speed = reward.tolerance(
            vel[:, 0],
            bounds=(self._run_speed, float("inf")),
            margin=self._run_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        torso_height = self._torso.get_position(data)[:, 2]
        rwd_posture = -1.5 * (torso_height - 0.75) ** 2
        rwd_posture = np.clip(rwd_posture, -1.0, 1.0)

        rwd = rwd_speed + rwd_posture

        return state.replace(
            obs=obs,
            reward=rwd,
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        data.reset(self._model)
        num = data.shape[0]

        limited_idx = np.where(self._model.joint_limits == 1)[0]
        low = self._joint_limits[0, limited_idx]
        high = self._joint_limits[1, limited_idx]

        qpos = data.dof_pos
        qpos[:, limited_idx] = np.random.uniform(low, high, size=(num, len(limited_idx)))
        data.set_dof_pos(qpos, self._model)

        for _ in range(200):
            self._model.step(data)

        obs = self._get_obs(data)

        return obs, {}
