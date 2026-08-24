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

import logging
from pathlib import Path

from absl import app, flags
from skrl import config

from motrix_rl import utils
from motrix_rl.skrl import get_log_dir

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "The env to play")
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)
_POLICY = flags.DEFINE_string("policy", None, "The policy to load")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to play")
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")


def get_inference_backend(policy_path: str):
    if policy_path.endswith(".pt"):
        return "torch"
    if policy_path.endswith(".pickle"):
        return "jax"
    else:
        raise Exception(f"Unknown policy format: {policy_path}")


def find_best_policy(env_name: str) -> str:
    """
    Find the most recent best policy for the given environment.

    Args:
        env_name: The name of the environment

    Returns:
        Path to the best policy file

    Raises:
        FileNotFoundError: If no policy files are found
    """
    # Base runs directory

    env_dir = Path(get_log_dir(env_name))

    if not env_dir.exists():
        raise FileNotFoundError(f"No training results found for environment '{env_name}' in {env_dir}")

    # Find all training run directories (pattern: YY-MM-DD_HH-MM-SS-_XXXXX_PPO)
    training_runs = [d for d in env_dir.iterdir() if d.is_dir()]

    if not training_runs:
        raise FileNotFoundError(f"No training runs found for environment '{env_name}'")

    # Sort by modification time to get the most recent
    latest_run = max(training_runs, key=lambda x: x.stat().st_mtime)
    checkpoints_dir = latest_run / "checkpoints"

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_run}")

    # First, try to find best_agent files (highest performance models)
    best_files = list(checkpoints_dir.glob("best_agent.*"))

    if best_files:
        # Return the first best_agent file found (there should only be one)
        return str(best_files[0])

    # If no best_agent files, find the checkpoint with the highest timestep
    checkpoint_files = list(checkpoints_dir.glob("agent_*.pt")) + list(checkpoints_dir.glob("agent_*.pickle"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No policy files found in {checkpoints_dir}")

    # Extract timestep from filename and find the highest
    def extract_timestep(filename):
        # Pattern: agent_{timestep}.ext
        stem = Path(filename).stem  # agent_{timestep}
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    latest_checkpoint = max(checkpoint_files, key=extract_timestep)
    return str(latest_checkpoint)


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    env_name = _ENV.value
    enable_render = True

    rl_override = {}

    if _NUM_ENVS.present:
        rl_override["play_num_envs"] = _NUM_ENVS.value

    if _RAND_SEED.value:
        rl_override["seed"] = None
    elif _SEED.present:
        rl_override["seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value

    # Determine policy path: use explicit policy if provided, otherwise auto-discover
    if _POLICY.present:
        policy_path = _POLICY.value
        logger.info(f"Using specified policy: {policy_path}")
    else:
        try:
            policy_path = find_best_policy(env_name)
            logger.info(f"Auto-discovered best policy: {policy_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("Please specify a policy using --policy flag or train a model first")
            return

    backend = get_inference_backend(policy_path)

    if backend == "jax":
        assert device_supports.jax, "jax is not avaliable on your device "
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # or "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)

    elif backend == "torch":
        assert device_supports.torch, "torch is not avaliable on your device"
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
        trainer.play(policy_path)


if __name__ == "__main__":
    app.run(main)
