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

import os
from dataclasses import dataclass

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/bounce_ball_ctrl.xml"


@registry.envcfg("bounce_ball")
@dataclass
class BounceBallEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 20.0

    # Ball and paddle physics parameters
    ball_restitution: float = 0.9  # Slightly less than perfect for realistic bouncing
    ball_linear_damping: float = 0.55
    ball_lateral_friction: float = 1.1
    paddle_restitution: float = 0.8
    paddle_linear_damping: float = 0.55
    paddle_lateral_friction: float = 1.1

    # Initial conditions
    ball_init_pos: list = None
    ball_init_vel: list = None
    arm_init_qpos: list = None

    # Target height for bouncing (configurable parameter)
    target_ball_height: float = 0.8  # Default target height in meters
    height_tolerance: float = 0.1  # Tolerance for reward calculation

    # Action scaling parameters
    action_scale: list = None
    action_bias: list = None

    def __post_init__(self):
        if self.ball_init_pos is None:
            self.ball_init_pos = [0.58856, 0, 1.27796]  # Slightly above paddle (paddle z=0.2803)
        if self.ball_init_vel is None:
            self.ball_init_vel = [0.0, 0.0, 0.0]
        if self.arm_init_qpos is None:
            self.arm_init_qpos = [0, 40, 110, 0, -60, 0]

        if self.action_scale is None:
            self.action_scale = [0.0008] * 6
        if self.action_bias is None:
            self.action_bias = [0.0] * 6
