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

"""环境注册表的一致性检查。

`import motrix_rl` 会连带导入 cfgs.py，而 cfgs.py 里每一个 @rlcfg 装饰器
都会去 env_registry 里查这个名字（motrix_rl/registry.py:52-53），查不到就抛
ValueError。所以"能 import 成功"本身就是一个有意义的断言。

但它只保证 env **配置**注册过，不保证 env **类**注册过 —— 后者才是能不能
真正创建环境的条件。下面的测试补上这个缺口。
"""

import pytest

# navigation/vbot/__init__.py:16 引用的几个 env 模块已经从磁盘上删除了
# （只剩 __pycache__ 里的孤儿 .pyc），但它们的 @registry.envcfg 仍然生效，
# 于是注册表里留下了"有配置、没实现"的空壳。
# 这些是继承来的历史包袱，本项目只记录、不修复 ——
# 修它们要动 cfgs.py 里被其它 env 共享的部分，属于范围蔓延。
KNOWN_BACKENDLESS_ENVS = {
    "vbot_navigation_flat",
    "vbot_navigation_stairs",
    "vbot_navigation_stairs_obstacles",
    "vbot_navigation_long_course",
    "VBotStairsMultiTarget-v0",
}


def test_package_import_registers_all_referenced_environments():
    """导入 motrix_rl 时，cfgs.py 里引用的每个 env 名字都必须已注册。"""
    import motrix_rl  # noqa: F401


def test_every_registered_env_has_at_least_one_backend():
    """注册表里不应存在"有配置、没实现"的空壳环境。

    当前有 5 个这样的空壳（见 KNOWN_BACKENDLESS_ENVS）：它们的 env 模块
    已被删除，但配置还留着。对它们调 registry.make() 会得到
    "does not support any simulation backend"。
    """
    import motrix_rl  # noqa: F401
    from motrix_envs import registry

    def has_backend(name: str) -> bool:
        # find_available_sim_backend 对没有后端的 env 是**抛异常**而不是返回 None
        try:
            return bool(registry.find_available_sim_backend(name))
        except ValueError:
            return False

    backendless = {name for name in registry.list_registered_envs() if not has_backend(name)}
    unexpected = backendless - KNOWN_BACKENDLESS_ENVS
    repaired = KNOWN_BACKENDLESS_ENVS - backendless

    assert not unexpected, f"出现了新的空壳环境：{sorted(unexpected)}"
    assert not repaired, f"这些空壳环境已被修复，请从 KNOWN_BACKENDLESS_ENVS 里移除：{sorted(repaired)}"


def test_section01_env_is_registered_with_np_backend():
    """本项目的主环境必须可用 —— 最基本的守门断言。"""
    import motrix_rl  # noqa: F401
    from motrix_envs import registry

    assert registry.contains("vbot_navigation_section01")
    assert "np" in registry.find_available_sim_backend("vbot_navigation_section01")


def test_section01_has_a_default_rl_config():
    """train.py / play.py 依赖 default_rl_cfg 能查到 PPO 超参。"""
    import motrix_rl  # noqa: F401
    from motrix_rl import registry as rl_registry

    cfg = rl_registry.default_rl_cfg("vbot_navigation_section01", "skrl", "torch")
    assert cfg is not None
    assert cfg.num_envs == 4096
    assert cfg.seed == 42
    # max_batch_env_steps 决定 SequentialTrainer 的 timesteps
    assert cfg.max_batch_env_steps == 45000, "A5 应为 15000 的三倍 trainer timesteps"
    assert cfg.effective_write_interval == 100
    assert cfg.effective_checkpoint_interval == 1000


def test_legacy_rl_configs_still_fall_back_to_checkpoint_interval():
    """旧配置只设置 check_point_interval 时，两个输出间隔保持兼容。"""
    from motrix_rl.base import BaseRLCfg

    cfg = BaseRLCfg(num_envs=256, max_env_steps=51200, check_point_interval=100)
    assert cfg.max_batch_env_steps == 200
    assert cfg.effective_write_interval == 100
    assert cfg.effective_checkpoint_interval == 100


@pytest.mark.parametrize("env_name", ["vbot_navigation_section001", "vbot_navigation_section011"])
def test_known_envs_without_rl_config_are_documented(env_name):
    """记录这两个 env 没有 @rlcfg 的事实 —— 对它们跑 train.py 会直接失败。"""
    import motrix_rl  # noqa: F401
    from motrix_rl import registry as rl_registry

    with pytest.raises(ValueError):
        rl_registry.default_rl_cfg(env_name, "skrl", "torch")
