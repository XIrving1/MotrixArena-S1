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

"""Section01 的端到端冒烟测试 —— "我有没有把 env 改坏"的守门测试。

每次改动 vbot_section01_np.py 之后都应该先跑这个：
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -k integration
"""

import numpy as np
import pytest

from motrix_envs import registry

pytestmark = pytest.mark.slow

NUM_ENVS = 32
NUM_STEPS = 100
OBS_DIM = 68
NUM_ACTIONS = 12


def test_section01_32_envs_run_100_steps_with_finite_68d_observations():
    """32 个环境跑 100 步随机动作，观测/奖励/终止标志全程必须合法。

    随机动作会让机器人以各种姿态摔倒、翻滚、卡在坡上，是很好的数值鲁棒性压力源。
    历史上这个 env 的观测里出现过 NaN（接触力传感器在某些姿态下返回异常），
    _get_obs 末尾的 np.nan_to_num 就是为此加的 —— 这个测试守住它。
    """
    env = registry.make("vbot_navigation_section01", sim_backend="np", num_envs=NUM_ENVS)
    state = env.init_state()

    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (NUM_ACTIONS,)
    assert state.obs.shape == (NUM_ENVS, OBS_DIM)

    rng = np.random.default_rng(42)
    for step in range(NUM_STEPS):
        actions = rng.uniform(-1.0, 1.0, size=(NUM_ENVS, NUM_ACTIONS)).astype(np.float32)
        state = env.step(actions)

        assert state.obs.shape == (NUM_ENVS, OBS_DIM), f"step {step}: 观测形状 {state.obs.shape}"
        assert np.all(np.isfinite(state.obs)), f"step {step}: 观测里有 NaN/Inf"
        assert state.reward.shape == (NUM_ENVS,), f"step {step}: 奖励形状 {state.reward.shape}"
        assert np.all(np.isfinite(state.reward)), f"step {step}: 奖励里有 NaN/Inf"
        # _compute_reward 末尾的 np.clip(-100, 1000)
        assert np.all(state.reward >= -100.0) and np.all(state.reward <= 1000.0)
        assert state.terminated.shape == (NUM_ENVS,)
        assert state.truncated.shape == (NUM_ENVS,)
        assert state.terminated.dtype == np.bool_
        assert state.truncated.dtype == np.bool_


def test_observation_layout_ends_with_eight_terrain_channels():
    """G3 恢复 48 基础 + 12 足力 + 8 地形的 68 维历史观测语义。"""
    env = registry.make("vbot_navigation_section01", sim_backend="np", num_envs=2)
    env.init_state()
    data, info = env.state.data, env.state.info

    obs = env._get_obs(data, info)
    assert obs.shape == (2, 68)

    terrain_scan = env._get_terrain_scan(env._body.get_pose(data)[:, :2])
    assert terrain_scan.shape == (2, 8), "地形采样必须是 8 维"
    assert np.all(terrain_scan >= 0.0) and np.all(terrain_scan <= 1.0), "地形采样应当归一化到 [0,1]"

    np.testing.assert_allclose(obs[:, -8:], terrain_scan, rtol=1e-6)


def test_kinematic_gait_state_resets_without_cross_episode_leak():
    """episode reset 必须清零离地状态和逐腿 stale，不能继承上一局。"""
    env = registry.make("vbot_navigation_section01", sim_backend="np", num_envs=2)
    env.init_state()
    env.state.info["_kinematic_airborne"][:] = True
    env.state.info["_leg_swing_active"][:] = True
    env.state.info["_leg_stale_time"][:] = 9.0
    env.state.info["_leg_last_air_time"][:] = 9.0
    env.state.terminated[:] = True

    env._reset_done_envs()

    np.testing.assert_array_equal(env.state.info["_kinematic_airborne"], False)
    np.testing.assert_array_equal(env.state.info["_leg_swing_active"], False)
    np.testing.assert_allclose(env.state.info["_leg_stale_time"], 0.0, atol=1e-7)
    np.testing.assert_allclose(env.state.info["_leg_last_air_time"], 0.0, atol=1e-7)
    np.testing.assert_array_equal(env.state.info["steps"], 0)
    assert env.state.obs.shape == (2, 68)


def test_reset_places_robot_on_the_start_platform_facing_the_course():
    """reset 必须把机器人放在起步平台、朝向 +Y（yaw ≈ π/2）。"""
    env = registry.make("vbot_navigation_section01", sim_backend="np", num_envs=NUM_ENVS)
    env.init_state()

    pose = env._body.get_pose(env.state.data)
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]

    # cfg: pos=(0, -2.4, 0.5)，随机化 ±0.5
    assert np.all(np.abs(x) <= 0.55), "出生点 X 超出随机化范围"
    assert np.all(y >= -3.0) and np.all(y <= -1.85), "出生点 Y 不在起步平台上"
    assert np.all(z > 0.2), "出生高度异常"

    yaw = env._quat_to_yaw(pose[:, 3:7])
    assert np.all(np.abs(yaw - 0.5 * np.pi) <= 0.2), "初始朝向不是 +Y"
