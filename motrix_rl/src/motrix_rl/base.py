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

import dataclasses
from typing import Optional


@dataclasses.dataclass
class BaseRLCfg:
    """
    Config for the reinforcement learning algorithm
    """

    # Basic training parameters
    seed: Optional[int] = None
    num_envs: int = 2048
    play_num_envs: int = 16
    max_env_steps: int = 20480000
    check_point_interval: int = 1000
    write_interval: Optional[int] = None
    checkpoint_interval: Optional[int] = None

    def replace(self, **updates) -> "BaseRLCfg":
        return dataclasses.replace(self, **updates)

    @property
    def max_batch_env_steps(self) -> int:
        """
        The max batched environment steps for the RL algorithm.
        """
        n = int(self.max_env_steps / self.num_envs)
        return (int)(n / self.check_point_interval) * self.check_point_interval

    @property
    def effective_write_interval(self) -> int:
        """TensorBoard write interval, falling back to the legacy field."""
        if self.write_interval is not None:
            return self.write_interval
        return self.check_point_interval

    @property
    def effective_checkpoint_interval(self) -> int:
        """Checkpoint save interval, falling back to the legacy field."""
        if self.checkpoint_interval is not None:
            return self.checkpoint_interval
        return self.check_point_interval
