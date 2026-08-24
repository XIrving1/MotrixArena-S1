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

from typing import Any

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO as BasePPO
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.envs.torch import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from motrix_envs import registry as env_registry
from motrix_rl import registry
from motrix_rl.skrl import get_log_dir
from motrix_rl.skrl.cfg import PPOCfg
from motrix_rl.skrl.torch import wrap_env


def _get_cfg(
    rlcfg: PPOCfg,
    env: Wrapper,
    log_dir: str = None,
) -> dict:
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rlcfg.rollouts  # memory_size
    cfg["learning_epochs"] = rlcfg.learning_epochs
    cfg["mini_batches"] = rlcfg.mini_batches  # mini_batch_size = rollouts * num_envs / mini_batches
    cfg["discount_factor"] = rlcfg.discount_factor
    cfg["lambda"] = rlcfg.lambda_param
    cfg["learning_rate"] = rlcfg.learning_rate
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": rlcfg.learning_rate_scheduler_kl_threshold}
    cfg["random_timesteps"] = rlcfg.random_timesteps
    cfg["learning_starts"] = rlcfg.learning_starts
    cfg["grad_norm_clip"] = rlcfg.grad_norm_clip
    cfg["ratio_clip"] = rlcfg.ratio_clip
    cfg["value_clip"] = rlcfg.value_clip
    cfg["clip_predicted_values"] = rlcfg.clip_predicted_values
    cfg["entropy_loss_scale"] = rlcfg.entropy_loss_scale
    cfg["value_loss_scale"] = rlcfg.value_loss_scale
    cfg["kl_threshold"] = rlcfg.kl_threshold
    if rlcfg.rewards_shaper_scale != 1.0:
        cfg["rewards_shaper"] = lambda reward, timestep, timesteps: reward * rlcfg.rewards_shaper_scale
    else:
        cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = rlcfg.time_limit_bootstrap
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    if log_dir:
        cfg["experiment"]["write_interval"] = rlcfg.effective_write_interval
        cfg["experiment"]["checkpoint_interval"] = rlcfg.effective_checkpoint_interval
        cfg["experiment"]["directory"] = log_dir
    else:
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0

    return cfg


class PPO(BasePPO):
    _total_custom_rewards: dict[str, torch.Tensor] = {}

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ) -> None:
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )
        if "Reward" in infos:
            for key, value in infos["Reward"].items():
                value = torch.tensor(value, device=self.device)
                self.tracking_data[f"Reward Instant / {key} (max)"].append(torch.max(value).item())
                self.tracking_data[f"Reward Instant / {key} (min)"].append(torch.min(value).item())
                self.tracking_data[f"Reward Instant / {key} (mean)"].append(torch.mean(value).item())
                if key not in self._total_custom_rewards:
                    self._total_custom_rewards[key] = torch.zeros_like(value)
                self._total_custom_rewards[key] += value
            done = terminated | truncated
            done = done.reshape(-1)
            if done.any():
                for key in self._total_custom_rewards:
                    self.tracking_data[f"Reward Total/ {key} (mean)"].append(
                        torch.mean(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (min)"].append(
                        torch.min(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (max)"].append(
                        torch.max(self._total_custom_rewards[key][done]).item()
                    )

                    self._total_custom_rewards[key] = self._total_custom_rewards[key] * (~done)

        if "metrics" in infos:
            for key, value in infos["metrics"].items():
                tracked_value = torch.tensor(value, device=self.device)
                self.tracking_data[f"metrics / {key} (max)"].append(torch.max(tracked_value).item())
                self.tracking_data[f"metrics / {key} (min)"].append(torch.min(tracked_value).item())
                self.tracking_data[f"metrics / {key} (mean)"].append(torch.mean(tracked_value).item())


class Trainer:
    _trainer: SequentialTrainer
    _env_name: str
    _sim_backend: str
    _rlcfg: PPOCfg
    _enable_render: bool

    def __init__(
        self,
        env_name: str,
        sim_backend: str = None,
        enable_render: bool = False,
        cfg_override: dict = None,
    ) -> None:
        rlcfg = registry.default_rl_cfg(env_name, "skrl", backend="torch")
        if cfg_override is not None:
            rlcfg = rlcfg.replace(**cfg_override)
        self._rlcfg = rlcfg
        self._env_name = env_name
        self._sim_backend = sim_backend
        self._enable_render = enable_render

    def train(self) -> None:
        """
        Start training the agent.
        """
        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.num_envs)
        set_seed(rlcfg.seed)
        skrl_env = wrap_env(env, self._enable_render)
        models = self._make_model(skrl_env, rlcfg)
        ppo_cfg = _get_cfg(rlcfg, skrl_env, log_dir=get_log_dir(self._env_name))
        agent = self._make_agent(models, skrl_env, ppo_cfg)
        cfg_trainer = {
            "timesteps": rlcfg.max_batch_env_steps,
            "headless": not self._enable_render,
        }
        trainer = SequentialTrainer(cfg=cfg_trainer, env=skrl_env, agents=agent)
        trainer.train()

    def play(self, policy: str) -> None:
        import time

        rlcfg = self._rlcfg
        env = env_registry.make(self._env_name, sim_backend=self._sim_backend, num_envs=rlcfg.play_num_envs)
        set_seed(rlcfg.seed)
        env = wrap_env(env, self._enable_render)
        models = self._make_model(env, rlcfg)
        ppo_cfg = _get_cfg(rlcfg, env)
        agent = self._make_agent(models, env, ppo_cfg)
        agent.load(policy)
        with torch.no_grad():
            obs, _ = env.reset()
            fps = 60
            while True:
                t = time.time()
                outputs = agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
                env.render()
                delta_time = time.time() - t
                if delta_time < 1.0 / fps:
                    time.sleep(1.0 / fps - delta_time)

    def _make_model(self, env: Wrapper, rlcfg: PPOCfg) -> dict[str, Model]:
        def build_mlp(
            input_size: int,
            hidden_sizes: tuple[int, ...],
            output_size: int,
            activation=nn.ELU,
        ):
            """Helper function to build MLP layers."""
            layers = []
            current_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(activation())
                current_size = hidden_size

            layers.append(nn.Linear(current_size, output_size))
            return nn.Sequential(*layers)

        # define shared model (stochastic and deterministic models) using mixins
        class Shared(GaussianMixin, DeterministicMixin, Model):
            def __init__(
                self,
                observation_space,
                action_space,
                device,
                policy_hidden_sizes,
                value_hidden_sizes,
                share_features=True,
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2,
                reduction="sum",
            ):
                Model.__init__(self, observation_space, action_space, device)
                GaussianMixin.__init__(
                    self,
                    clip_actions,
                    clip_log_std,
                    min_log_std,
                    max_log_std,
                    reduction,
                )
                DeterministicMixin.__init__(self, clip_actions)

                # Use configured share_features setting
                self.share_features = share_features and policy_hidden_sizes == value_hidden_sizes

                if self.share_features:
                    # Build shared feature extraction layers
                    shared_layers = []
                    current_size = self.num_observations

                    for hidden_size in policy_hidden_sizes:
                        shared_layers.append(nn.Linear(current_size, hidden_size))
                        shared_layers.append(nn.ELU())
                        current_size = hidden_size

                    self.net = nn.Sequential(*shared_layers)
                    self.mean_layer = nn.Linear(current_size, self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))
                    self.value_layer = nn.Linear(current_size, 1)
                else:
                    # Build separate networks for policy and value
                    self.policy_net = build_mlp(
                        self.num_observations,
                        policy_hidden_sizes[:-1],
                        policy_hidden_sizes[-1] if len(policy_hidden_sizes) > 0 else self.num_actions,
                    )
                    self.value_net = build_mlp(
                        self.num_observations,
                        value_hidden_sizes[:-1],
                        value_hidden_sizes[-1] if len(value_hidden_sizes) > 0 else 1,
                    )

                    # Output layers
                    if len(policy_hidden_sizes) > 0:
                        self.mean_layer = nn.Linear(policy_hidden_sizes[-1], self.num_actions)
                    else:
                        self.mean_layer = nn.Linear(self.num_observations, self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

                    if len(value_hidden_sizes) > 0:
                        self.value_layer = nn.Linear(value_hidden_sizes[-1], 1)
                    else:
                        self.value_layer = nn.Linear(self.num_observations, 1)

            def act(self, inputs, role):
                if role == "policy":
                    return GaussianMixin.act(self, inputs, role)
                elif role == "value":
                    return DeterministicMixin.act(self, inputs, role)

            def compute(self, inputs, role):
                if role == "policy":
                    if self.share_features:
                        self._shared_output = self.net(inputs["states"])
                        return (
                            self.mean_layer(self._shared_output),
                            self.log_std_parameter,
                            {},
                        )
                    else:
                        policy_features = self.policy_net(inputs["states"])
                        return (
                            self.mean_layer(policy_features),
                            self.log_std_parameter,
                            {},
                        )
                elif role == "value":
                    if self.share_features:
                        shared_output = (
                            self.net(inputs["states"]) if self._shared_output is None else self._shared_output
                        )
                        self._shared_output = None
                        return self.value_layer(shared_output), {}
                    else:
                        value_features = self.value_net(inputs["states"])
                        return self.value_layer(value_features), {}

        models = {}
        models["policy"] = Shared(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            policy_hidden_sizes=rlcfg.policy_hidden_layer_sizes,
            value_hidden_sizes=rlcfg.value_hidden_layer_sizes,
            share_features=rlcfg.share_policy_value_features,
        )

        models["value"] = models["policy"]

        return models

    def _make_agent(self, models: dict[str, Model], env: Wrapper, ppo_cfg: dict[str, Any]) -> PPO:
        memory = RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=env.num_envs, device=env.device)

        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        return agent
