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
from motrix_envs.basic.hopper.cfg import HopperStandCfg
from motrix_envs.np import reward
from motrix_envs.np.env import NpEnv, NpEnvState


@registry.env("dm-hopper-stand", "np")
@registry.env("dm-hopper-hop", "np")
class HopperEnv(NpEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: HopperStandCfg, num_envs=1):
        super().__init__(cfg, num_envs)
        self._init_obs_space()
        self._init_action_space()

        self._torso = self._model.get_link("torso")
        self._foot = self._model.get_link("foot")

        self._stand_height = cfg.stand_height
        self._hop_speed = cfg.hop_speed
        self._joint_limits = self._model.joint_limits

    def _init_obs_space(self):
        model = self._model
        num = 0
        num += model.num_dof_pos - 1
        num += model.num_dof_vel
        num += 2
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
        qpos = data.dof_pos[:, 1:]
        qvel = data.dof_vel
        num_env = int(data.shape[0])

        toe = np.asarray(self._model.get_sensor_value("touch_toe", data)).reshape(num_env, -1)[:, 0]
        heel = np.asarray(self._model.get_sensor_value("touch_heel", data)).reshape(num_env, -1)[:, 0]

        toe = np.log1p(toe)
        heel = np.log1p(heel)
        touch = np.stack([toe, heel], axis=-1)  # shape -> (num_env, 2)
        return np.concatenate([qpos, qvel, touch], axis=-1)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data

        # === obs ===
        obs = self._get_obs(data)

        num_env = int(data.shape[0])
        toe = np.asarray(self._model.get_sensor_value("touch_toe", data)).reshape(num_env, -1)[:, 0]
        heel = np.asarray(self._model.get_sensor_value("touch_heel", data)).reshape(num_env, -1)[:, 0]

        toe = np.log1p(toe)
        heel = np.log1p(heel)

        # === physical values ===
        torso_pos = self._torso.get_position(data)
        foot_pos = self._foot.get_position(data)
        torso_height = torso_pos[:, 2] - foot_pos[:, 2]

        torso_vel = self._model.get_sensor_value("torso_subtreelinvel", data)
        speed = torso_vel[:, 0]

        # === terminated ===
        over_speed = np.sum(np.square(data.dof_vel[:, 4:7]), axis=-1) > 1e8
        terminated = np.isnan(obs).any(axis=-1)
        terminated |= over_speed

        standing = reward.tolerance(
            torso_height,
            bounds=(self._stand_height, 2.0),
            margin=self._stand_height * 0.5,
        )

        if self._hop_speed > 0.0:
            hopping = reward.tolerance(
                speed,
                bounds=(self._hop_speed * 0.3, float("inf")),
                margin=self._hop_speed * 0.3,
                value_at_margin=0.0,
                sigmoid="linear",
            )

            leg_vel = np.linalg.norm(data.dof_vel[:, 4:7], axis=-1)
            leg_bonus = np.tanh(leg_vel * 0.3) * 0.2 * standing

            knee_vel = data.dof_vel[:, 5]
            extend_reward = np.maximum(knee_vel, 0) * 0.2 * standing

            stand_condition = (torso_height > self._stand_height * 0.8).astype(np.float32)
            effective_hop_reward = hopping * stand_condition

            contact_strength = toe + heel
            contact_reward = np.clip(contact_strength, 0.0, 1.0) * 0.1 * standing

            rwd = standing * 0.8 + effective_hop_reward * 0.8 + leg_bonus * 0.5 + extend_reward + contact_reward
            if np.average(rwd) > 1000:
                print(
                    "standing",
                    np.sum(standing),
                    "effective_hop_reward",
                    np.sum(effective_hop_reward),
                    "leg_bonus",
                    np.sum(leg_bonus),
                    "extend_reward",
                    np.sum(extend_reward),
                    "contact_reward",
                    np.sum(contact_reward),
                )

        else:
            control_magnitude = np.linalg.norm(data.actuator_ctrls, axis=-1)
            small_control = reward.tolerance(
                control_magnitude,
                bounds=(0, 1),
                margin=1,
                value_at_margin=0,
                sigmoid="quadratic",
            )
            small_control = (small_control + 4) / 5

            rwd = standing * small_control
            state.info["Reward"] = {"stand": standing, "control": small_control, "total": rwd}

        rwd[terminated] = 0.0

        return state.replace(
            obs=obs,
            reward=rwd,
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData):
        data.reset(self._model)
        num_env = data.shape[0]

        dof_pos = np.zeros((num_env, self._model.num_dof_pos))

        dof_pos[:, 2] = 0

        if self._model.num_dof_pos > 3:
            dof_pos[:, 3:] = np.random.uniform(
                low=self._joint_limits[0, 3:],
                high=self._joint_limits[1, 3:],
                size=(num_env, self._model.num_dof_pos - 3),
            )

        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        obs = self._get_obs(data)

        rewards = {"stand": np.zeros((num_env,))}
        if self._hop_speed > 0.0:
            rewards["hop"] = np.zeros((num_env,))

        return obs, {"Reward": rewards}
