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

"""把 68 维 G3 观测里的 8 维地形采样与真实碰撞几何对照。"""

import numpy as np
import pytest

from motrix_envs.navigation.vbot.vbot_section01_np import (
    TERRAIN_PLATFORM_HEIGHT,
    TERRAIN_ROUGH_Y_END,
    TERRAIN_SLOPE_TAN,
    TERRAIN_SLOPE_Y_START,
    section01_terrain_height,
)

pytestmark = pytest.mark.slow


def _scan_at(env, base_y: float) -> np.ndarray:
    """机器人位于 (0, base_y) 时的 8 维地形采样。"""
    return env._get_terrain_scan(np.array([[0.0, base_y]], dtype=np.float32))[0]


def _truth_at(env, base_y: float) -> np.ndarray:
    """同样 8 个采样点处的真实归一化高度。"""
    sample_y = base_y + env.TERRAIN_SCAN_OFFSETS
    return section01_terrain_height(sample_y) / TERRAIN_PLATFORM_HEIGHT


def test_terrain_scan_slope_section_matches_truth(section01_env):
    """坡道段（Y 2.0~6.8）的地形采样与解析真值一致。"""
    env = section01_env
    for base_y in (2.2, 3.0, 4.0, 5.0, 6.0):
        scan = _scan_at(env, base_y)
        truth = _truth_at(env, base_y)
        assert np.max(np.abs(scan - truth)) < 0.02, f"base_y={base_y} 处坡道采样偏差过大"

    # 直接检查梯度
    ys = np.linspace(2.5, 6.0, 50)
    front = np.array([_scan_at(env, y)[0] for y in ys])
    gradient_code = np.polyfit(ys + float(env.TERRAIN_SCAN_OFFSETS[0]), front, 1)[0]
    gradient_true = TERRAIN_SLOPE_TAN / TERRAIN_PLATFORM_HEIGHT
    assert gradient_code == pytest.approx(gradient_true, rel=0.02)


def test_terrain_scan_is_flat_on_start_and_finish_platforms(section01_env):
    """起步平台读 0、终点平台读 1 —— 两端也是对的。"""
    env = section01_env
    assert _scan_at(env, -3.0)[0] == pytest.approx(0.0, abs=1e-6)
    assert _scan_at(env, 7.5)[0] == pytest.approx(1.0, abs=1e-6)


def test_terrain_scan_reveals_the_drop_lip(section01_env):
    """Y=1.5 的落差坎必须在地形采样里可见。

    真实剖面：Y<1.5 是最高 0.277m 的 hfield 起伏，Y>=1.5 骤降到 z=0 的平地。
    这是全赛道最容易摔的一处，采样必须在缺口段返回接近 0 的高度。
    """
    env = section01_env
    # 采样点落在缺口段 [1.5, 2.0) 时，真值是 0。
    for base_y in (1.35, 1.45, 1.6, 1.75):
        scan = _scan_at(env, base_y)
        truth = _truth_at(env, base_y)
        in_gap = np.logical_and(
            base_y + env.TERRAIN_SCAN_OFFSETS >= TERRAIN_ROUGH_Y_END,
            base_y + env.TERRAIN_SCAN_OFFSETS < TERRAIN_SLOPE_Y_START,
        )
        if not np.any(in_gap):
            continue
        err = np.max(np.abs(scan[in_gap] - truth[in_gap]))
        assert err < 0.05, (
            f"base_y={base_y}: 缺口段采样报 {scan[in_gap][0]:.3f}，"
            f"真值 {truth[in_gap][0]:.3f}，误差 {err:.3f}（{err * TERRAIN_PLATFORM_HEIGHT:.3f} m）"
        )


def test_terrain_scan_is_informative_in_rough_zone(section01_env):
    """机器人深入崎岖区时，8 个前向采样点之间必须有区分度。

    否则策略无法区分"刚进崎岖区"和"快到落差坎了"，只能盲走。
    """
    env = section01_env
    for base_y in (-1.2, -0.6, 0.0):
        scan = _scan_at(env, base_y)
        assert np.std(scan) > 1e-6, f"base_y={base_y} 处 8 个采样点完全相同（全为 {scan[0]:.3f}），零信息"


def test_terrain_truth_is_self_consistent():
    """section01_terrain_height 自身的连续性 —— 坡底接平地、坡顶接平台。"""
    eps = 1e-4
    assert section01_terrain_height(np.array([TERRAIN_SLOPE_Y_START - eps]))[0] == pytest.approx(0.0, abs=1e-5)
    assert section01_terrain_height(np.array([TERRAIN_SLOPE_Y_START + eps]))[0] == pytest.approx(0.0, abs=1e-4)
    top_from_slope = section01_terrain_height(np.array([6.8296282 - eps]))[0]
    top_from_platform = section01_terrain_height(np.array([6.8296282 + eps]))[0]
    assert top_from_slope == pytest.approx(TERRAIN_PLATFORM_HEIGHT, abs=1e-4)
    assert top_from_platform == pytest.approx(TERRAIN_PLATFORM_HEIGHT, abs=1e-6)
