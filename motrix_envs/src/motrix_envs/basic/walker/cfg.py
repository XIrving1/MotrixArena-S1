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

model_file = os.path.dirname(__file__) + "/walker.xml"


@registry.envcfg("dm-walker")
@dataclass
class WalkerEnvCfg(EnvCfg):
    model_file: str = model_file
    max_episode_seconds: float = 25.0
    sim_dt: float = 0.0125
    move_speed: float = 1.0
    ctrl_dt: float = 0.025
    stand_height: float = 1.2


@registry.envcfg("dm-stander")
@dataclass
class StanderEnvCfg(WalkerEnvCfg):
    move_speed: float = 0.0


@registry.envcfg("dm-runner")
@dataclass
class RunnerEnvCfg(WalkerEnvCfg):
    move_speed: float = 5.0
