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


"""PPO Agent Configuration for SKRL

This module provides configuration classes for PPO agents that match the
YAML configuration structure used in SKRL.
"""

from dataclasses import dataclass

from motrix_rl.base import BaseRLCfg


@dataclass
class PPOCfg(BaseRLCfg):
    """PPO configuration .

    This class provides all the parameters needed to configure a PPO agent
    in SKRL
    """

    # Model architecture settings
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    # Whether to share feature extraction layers between policy and value networks. only works if:
    # 1. both networks have the same architecture
    # 2. the backend is torch
    share_policy_value_features: bool = True

    # Agent settings
    rollouts: int = 32
    learning_epochs: int = 2
    mini_batches: int = 32
    discount_factor: float = 0.99
    lambda_param: float = 0.95

    # Learning rate settings
    learning_rate: float = 1e-3
    learning_rate_scheduler_kl_threshold: float = 0.008

    # Training settings
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 1.0

    time_limit_bootstrap: bool = True

    # PPO clipping settings
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    clip_predicted_values: bool = True

    # Loss settings
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 2.0
    kl_threshold: float = 0

    # Reward shaping
    rewards_shaper_scale: float = 1.0
