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
import numpy as np
from absl import app, flags

from motrix_envs import registry
from motrix_envs.np.env import NpEnv
from motrix_envs.np.renderer import NpRenderer

_ENV = flags.DEFINE_string("env", "cartpole", "The env to view")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "The simulation backend to use.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 1, "Number of parallel environments.")


class NpEnvRunner:
    _renderer: NpRenderer

    def __init__(self, env: NpEnv):
        self._env = env
        self._renderer = NpRenderer(env)

    def _sample_random_action(self):
        action_space = self._env.action_space
        if isinstance(action_space, gym.spaces.Box):
            size = (self._env.num_envs, *action_space.shape)
            low = action_space.low
            high = action_space.high
            low = np.where(np.isneginf(low), -1e6, low)
            high = np.where(np.isposinf(high), 1e6, high)
            return np.random.uniform(
                low=low,
                high=high,
                size=size,
            ).astype(action_space.dtype)
        else:
            raise NotImplementedError("Only Box action space is supported")

    def step(self):
        actions = self._sample_random_action()
        self._env.step(actions)

    def start(self):
        import time

        env_dt = self._env.cfg.ctrl_dt
        while True:
            t0 = time.monotonic()
            actions = self._sample_random_action()
            self._env.step(actions)
            self._renderer.render()
            real_dt = time.monotonic() - t0
            sleep_dt = env_dt - real_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)


def main(argv):
    env_name = _ENV.value
    sim_backend = _SIM_BACKEND.value
    num_envs = _NUM_ENVS.value
    env = registry.make(env_name, sim_backend=sim_backend, num_envs=num_envs)

    runner = NpEnvRunner(env)
    runner.start()


if __name__ == "__main__":
    app.run(main)
