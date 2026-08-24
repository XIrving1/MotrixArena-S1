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

from absl import app, flags
from skrl import config

from motrix_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "The env to train")
_SIM_BACKEND = flags.DEFINE_string(
    "sim-backend",
    None,
    "The simulation backend to use.(If not specified, it will be choosen automatically)",
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", 2048, "Number of envs to train")
_MAX_ENV_STEPS = flags.DEFINE_integer(
    "max-env-steps", None, "Override total environment steps across parallel envs"
)
_WRITE_INTERVAL = flags.DEFINE_integer(
    "write-interval", None, "Override the TensorBoard write interval for this run"
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    "checkpoint-interval", None, "Override the checkpoint save interval for this run"
)
_RENDER = flags.DEFINE_bool("render", False, "Render the env")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", "jax", "The learning backend. (jax/torch)")
_SEED = flags.DEFINE_integer("seed", None, "Random seed for reproducibility")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Generate random seed")


def get_train_backend(supports: utils.DeviceSupports):
    if supports.jax and supports.jax_gpu:
        return "jax"
    elif supports.torch and supports.torch_gpu:
        return "torch"
    elif supports.jax:
        return "jax"
    elif supports.torch:
        return "torch"
    else:
        raise Exception("neither jax nor torch not avaliable on the device.")


def main(argv):
    device_supports = utils.get_device_supports()
    logger.info(device_supports)
    env_name = _ENV.value
    enable_render = _RENDER.value

    rl_override = {}

    if _NUM_ENVS.present:
        rl_override["num_envs"] = _NUM_ENVS.value

    if _MAX_ENV_STEPS.present:
        rl_override["max_env_steps"] = _MAX_ENV_STEPS.value
    if _WRITE_INTERVAL.present:
        rl_override["write_interval"] = _WRITE_INTERVAL.value
    if _CHECKPOINT_INTERVAL.present:
        rl_override["checkpoint_interval"] = _CHECKPOINT_INTERVAL.value

    if _RAND_SEED.value:
        rl_override["seed"] = None
    elif _SEED.present:
        rl_override["seed"] = _SEED.value

    sim_backend = _SIM_BACKEND.value
    train_backend = "jax"
    if not _TRAIN_BACKEND.present:
        train_backend = get_train_backend(device_supports)
    else:
        train_backend = _TRAIN_BACKEND.value

    trainer = None
    if train_backend == "jax":
        from motrix_rl.skrl.jax.train import ppo

        config.jax.backend = "jax"  # or "numpy"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)

    elif train_backend == "torch":
        from motrix_rl.skrl.torch.train import ppo

        config.torch.backend = "torch"
        trainer = ppo.Trainer(env_name, sim_backend, cfg_override=rl_override, enable_render=enable_render)
    else:
        raise Exception(f"Unknown train backend: {train_backend}")

    trainer.train()


if __name__ == "__main__":
    app.run(main)
