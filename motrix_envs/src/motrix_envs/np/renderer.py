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

import numpy as np
from motrixsim.render import RenderApp, RenderSettings

from motrix_envs.np.env import NpEnv


class NpRenderer:
    """
    The renderer for Np sim environments.
    """

    _env: NpEnv

    def __init__(self, env: NpEnv):
        num_envs = env.num_envs
        num_envs = 1 if num_envs is None else num_envs
        spacing = env.render_spacing
        cols = int(np.ceil(np.sqrt(num_envs)))
        offsets = []
        for i in range(num_envs):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])

        self._env = env
        self._render = RenderApp()
        settings = RenderSettings.performance()
        settings.enable_shadow = True  # disable shadow for better performance
        self._render.launch(
            env.model,
            batch=num_envs,
            render_offset=offsets,
            render_settings=settings,
        )
        self._sync_render_data = True
        self._render.system_camera.active = self._sync_render_data

    def render(self) -> None:
        """
        render the env
        """

        self._render.sync(data=self._env.state.data if self._sync_render_data else None)
        if self._render.input.is_key_just_pressed("space"):
            self._sync_render_data = not self._sync_render_data
            self._render.system_camera.active = self._sync_render_data
