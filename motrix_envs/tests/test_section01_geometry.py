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

"""Section01 赛道几何与接触覆盖的测试。

这一组测试不需要跑仿真，只读碰撞模型，却能抓出 BUG-001（终止检测只覆盖崎岖区）。

这组断言同时覆盖已知缺陷和修复后的回归行为。测试因此既是缺陷文档，
也是终止检测覆盖完成的信号。
"""

import numpy as np
import pytest

from motrix_envs.navigation.vbot.vbot_section01_np import (
    TERRAIN_PLATFORM_HEIGHT,
    TERRAIN_SLOPE_TAN,
    TERRAIN_SLOPE_Y_END,
    TERRAIN_SLOPE_Y_START,
)

pytestmark = pytest.mark.slow

# 从 xmls/0126_C_section01.xml 抄下来的可行走面（全部是无名的 <default class> box）。
# (类名, Y 下界, Y 上界, 顶面 z, 人类可读名)
WALKABLE_SURFACES = [
    ("C_Adiban_001", -3.5, -1.5, 0.0, "起步平台"),
    ("C_Adiban_005", -1.5, 1.5, 0.0, "崎岖区基座"),
    ("hfield", -1.5, 1.5, 0.277056, "崎岖起伏(hfield)"),
    ("C_Adiban_002", 1.5, 2.0, 0.0, "平地缺口(落差着陆点)"),
    ("C_Adiban_003", 2.0, 6.8296282, None, "15°坡道"),
    ("C_Adiban_004", 6.8296282, 8.8296282, 1.2940952, "终点平台"),
]


def test_named_ground_geoms_are_only_hfields(section01_model):
    """记录 BUG-001 的成因：碰撞模型里只有 hfield 是具名的。

    这是一个回归探针 —— 哪天资产方给那些 box 起了名字，
    `_init_termination_contact` 的子串扫描就会突然开始生效，本测试会第一时间告诉你。
    """
    model = section01_model
    named_ground = [n for n in model.geom_names if n and any(p in n for p in ("C1_", "C2_", "C3_"))]

    assert named_ground == [
        "C1_V_Adixing_Plane",
        "C2_V_Bdixing_Plane",
        "C3_V_Cdixing_Plane",
    ], "碰撞体的具名 geom 集合发生了变化，BUG-001 的前提需要重新评估"

    # 三个都是 hfield（赛段 01/02/03 各一张），坡道与平台的 box 全部无名。
    assert all("dixing_Plane" in n for n in named_ground)

    unnamed = sum(1 for n in model.geom_names if not n)
    assert model.num_geoms == 271
    assert unnamed == 242, f"无名 geom 数量变了（{unnamed}），碰撞资产可能被替换过"


def test_base_contact_sensors_exist_and_are_readable(section01_env_stepped):
    """证明"正确的那条路"是可用的：base_contact_{1,2,3} subtree 传感器能读。

    它们在 xmls/scene_section01.xml:53,59,65 用 subtree2="C{1,2,3}_ground_root" 声明，
    覆盖整棵碰撞子树（全部 271 个 geom），而不像 geom 名字扫描那样只能命中 3 个。

    注意 SceneModel 没有 sensor_names 属性，只能按名字逐个探测。
    """
    env = section01_env_stepped
    data = env.state.data

    for sensor_name in ("base_contact_1", "base_contact_2", "base_contact_3"):
        value = env.model.get_sensor_value(sensor_name, data)
        assert value is not None, f"{sensor_name} 读不到"
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float32)))


def test_slope_surface_matches_collision_geometry():
    """坡面的闭式解必须与 C_Adiban_003 的 quat/pos/size 吻合，并与终点平台接得上。

    这道题锁住的是 section01_terrain_height() 的正确性 —— 后续的地形观测、
    height_above_terrain 指标、失败区间直方图全都建立在它之上。
    """
    # C_Adiban_003: size(5, 2.5, 0.25), quat = R_x(15°), pos(0, 4.4795189, 0.4055660)
    half_thickness, half_length = 0.25, 2.5
    theta = np.deg2rad(15.0)
    center = np.array([0.0, 4.4795189, 0.4055660])

    # 顶面中心 = pos + R_x(θ)·(0, 0, half_thickness)
    top_center = center + np.array([0.0, -half_thickness * np.sin(theta), half_thickness * np.cos(theta)])
    # 沿坡方向 = R_x(θ)·(0, 1, 0)
    along = np.array([0.0, np.cos(theta), np.sin(theta)])
    low_end = top_center - half_length * along
    high_end = top_center + half_length * along

    assert low_end[1] == pytest.approx(TERRAIN_SLOPE_Y_START, abs=1e-6)
    assert low_end[2] == pytest.approx(0.0, abs=1e-6)
    assert high_end[1] == pytest.approx(TERRAIN_SLOPE_Y_END, abs=1e-6)
    assert high_end[2] == pytest.approx(TERRAIN_PLATFORM_HEIGHT, abs=1e-6)

    # 与终点平台 C_Adiban_004 顶面（pos.z 1.0440952 + size.z 0.25）严丝合缝
    assert high_end[2] == pytest.approx(1.0440952 + 0.25, abs=1e-5)
    # 坡度就是 tan15°
    assert (high_end[2] - low_end[2]) / (high_end[1] - low_end[1]) == pytest.approx(TERRAIN_SLOPE_TAN, rel=1e-6)


def test_termination_pairs_cover_all_walkable_surfaces(section01_env):
    """基座触地终止必须覆盖赛道上**每一个**可行走面，而不只是崎岖 hfield。

    漏检的后果：狗在坡道或平台上肚皮着地时既不终止 episode，也收不到
    base_contact 惩罚，于是它可以趴在那里白拿 30 秒的存活奖励。
    实测基线里 65% 的 episode 是超时截断而非摔倒终止，就是这个 bug 的指纹。
    """
    env = section01_env
    model = env.model

    if env.num_termination_check == 0:
        assert env._base_contact_sensors == ["base_contact_1", "base_contact_2", "base_contact_3"]
        return

    covered_ground_idx = set(int(i) for i in env.termination_contact[:, 1])
    covered_names = {model.geom_names[i] for i in covered_ground_idx}

    # 覆盖到的应当是全部可行走面，而不是只有 3 张 hfield
    assert len(covered_ground_idx) > 3, (
        f"终止检测只覆盖了 {len(covered_ground_idx)} 个 geom：{sorted(covered_names)}。"
        f"赛道共有 {len(WALKABLE_SURFACES)} 类可行走面，坡道与平台全部漏检。"
    )


def test_base_contact_uses_full_coverage_path(section01_env):
    """基座触地检测应当走覆盖完整的 base_contact_{1,2,3} 传感器路径。

    vbot_section01_np.py 的逻辑是"有完整 geom 对就用 geom 对，否则用传感器"。
    Section01 的名字扫描只会命中 hfield，因此修复后应主动走传感器路径。
    """
    env = section01_env
    assert env.num_termination_check == 0, (
        f"num_termination_check == {env.num_termination_check}，"
        "geom 路径抢占了传感器 fallback；但这 6 个 geom 对只覆盖 hfield。"
    )
