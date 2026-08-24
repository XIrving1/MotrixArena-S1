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
from dataclasses import dataclass, field

import numpy as np

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/mjx_scene.xml"


@dataclass
class InitState:
    # robot joint names and default positions [rad]
    joint_names = [
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
    default_joint_pos = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], np.float32)
    joint_pos_reset_noise_scale = 0.125


@dataclass
class ControlConfig:
    # Position control
    # The actuator defined in xml file is <position ..../>
    # From ctrlrange in actuator in xml
    # Using position control and action as offset effectively solves the problem of large joint angle changes
    actuators = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7", "actuator8"]
    min_pos = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -np.pi / 2, 0]
    max_pos = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, np.pi / 2, 0.04]


@dataclass
class Commands:
    target_pos_x = [0.4, 0.6]
    target_pos_y = [-0.25, 0.25]
    target_pos_z = [0.25, 0.5]


@dataclass
class Asset:
    ground_name = "table"
    terminate_after_contacts_on = ["left_finger_pad", "left_finger_pad"]


@registry.envcfg("franka-lift-cube")
@dataclass
class FrankaLiftCubeEnvCfg(EnvCfg):
    render_spacing: float = 2.0
    model_file: str = model_file
    max_episode_seconds: float = 2.5
    sim_dt: float = 0.01
    move_speed: float = 1.0
    ctrl_dt: float = 0.01
    reset_noise_scale = 0.05

    init_state: InitState = field(default_factory=InitState)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    command_config: Commands = field(default_factory=Commands)
    asset: Asset = field(default_factory=Asset)
