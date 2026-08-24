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

"""Section01 奖励项的形状、有效性与独立性测试。

本文件里最重要的一条是 test_no_reward_term_is_identically_zero_over_a_rollout：
它把"一个奖励项可以被完整定义、加权、记录到 TensorBoard，却永远不产生梯度"
这件事变成了一个六行的可执行断言。

基线训练（15000 iter）的实测结果是 drop_leg_catchup(权重 4.0)、
slope_leg_drive(3.0)、slope_front_drive(3.0) 三项的 episode 累计值**恒为 0.000**，
合计 10.0 的奖励权重完全空转。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# 一个典型 trot 时刻：对角腿 FR/RL 处于摆动相（已腾空 0.30s），
# 另一对角 FL/RR 支撑着地（腾空时间刚被清零）。
# 顺序与 cfg.sensor.feet 一致：FR, FL, RR, RL
TROT_AIR_TIME = np.array([0.30, 0.0, 0.0, 0.30], dtype=np.float32)
TROT_CONTACTS = np.array([False, True, True, False], dtype=np.bool_)


def _reward_terms_with_forced_gait(
    env,
    base_y: float,
    air_time: np.ndarray,
    contacts: np.ndarray,
    stale_time: np.ndarray | None = None,
    last_air_time: np.ndarray | None = None,
    first_contact: np.ndarray | None = None,
    command: np.ndarray | None = None,
    base_pitch: float | None = None,
    base_roll: float | None = None,
):
    """把机器人放到指定 Y、强制指定步态状态，然后直接算一次奖励。

    绕开物理是**故意的**：我们要测的是奖励公式本身，而不是"随机动作下
    碰巧有没有触发"。直接注入步态状态，结果完全确定、不会 flaky。
    """
    from conftest import teleport_envs_along_course

    if env.state is None:
        env.init_state()

    num_envs = env.num_envs
    teleport_envs_along_course(env, np.full((num_envs,), base_y, dtype=np.float32))

    info = env.state.info
    info["feet_air_time"] = np.tile(air_time, (num_envs, 1))
    info["contacts"] = np.tile(contacts, (num_envs, 1))
    info["_kinematic_airborne"] = np.tile(~contacts, (num_envs, 1))
    if stale_time is None:
        stale_time = np.zeros(4, dtype=np.float32)
    info["_leg_stale_time"] = np.tile(stale_time, (num_envs, 1)).astype(np.float32)
    if last_air_time is None:
        last_air_time = np.zeros(4, dtype=np.float32)
    info["_leg_last_air_time"] = np.tile(last_air_time, (num_envs, 1)).astype(np.float32)
    if first_contact is None:
        first_contact = np.zeros(4, dtype=np.bool_)
    info["first_contact"] = np.tile(first_contact, (num_envs, 1)).astype(np.bool_)
    info["air_time_before_contact"] = np.tile(air_time, (num_envs, 1))
    # 给一个明确的前进指令，让带 active_move 掩码的项处于激活状态
    if command is None:
        command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    info["commands"] = np.tile(command, (num_envs, 1))

    if base_pitch is not None or base_roll is not None:
        pitch = 0.0 if base_pitch is None else base_pitch
        roll = 0.0 if base_roll is None else base_roll
        dof_pos = np.array(env.state.data.dof_pos, dtype=np.float32, copy=True)
        dof_pos[:, 6:10] = np.array(
            [
                np.sin(roll / 2.0) * np.cos(pitch / 2.0),
                np.cos(roll / 2.0) * np.sin(pitch / 2.0),
                0.0,
                np.cos(roll / 2.0) * np.cos(pitch / 2.0),
            ],
            dtype=np.float32,
        )
        env.state.data.set_dof_pos(dof_pos, env.model)
        env.model.forward_kinematic(env.state.data)

    env._compute_reward(env.state.data, info, np.zeros((num_envs,), dtype=np.bool_))
    return {k: np.asarray(v, dtype=np.float64) for k, v in info["Reward"].items()}


def test_leg_order_matches_sensor_and_actuator_convention(section01_env):
    """足端运动学、传感器与每组三关节必须共用 FR/FL/RR/RL 顺序。"""
    expected = ["FR", "FL", "RR", "RL"]
    assert list(section01_env._cfg.sensor.feet) == expected
    for leg_name, sensor_group in zip(expected, section01_env._foot_contact_sensor_groups, strict=True):
        assert sensor_group
        assert all(name.startswith(f"{leg_name}_") for name in sensor_group)

    actuator_legs = [name.split("_", 1)[0] for name in section01_env.model.actuator_names]
    assert actuator_legs == [leg for leg in expected for _ in range(3)]


def test_kinematic_liftoff_requires_height_and_upward_velocity(section01_env):
    """越过离地阈值且向上运动才是抬腿，单纯高度变化不能伪造事件。"""
    previous_height = np.array([[0.030, 0.070, 0.030, 0.030]], dtype=np.float32)
    current_height = np.array([[0.060, 0.060, 0.050, 0.030]], dtype=np.float32)
    previous_airborne = np.zeros((1, 4), dtype=np.bool_)

    airborne, liftoff, touchdown, velocity = section01_env._kinematic_gait_events(
        current_height,
        previous_height,
        previous_airborne,
    )

    np.testing.assert_array_equal(airborne, [[True, True, False, False]])
    np.testing.assert_array_equal(liftoff, [[True, False, False, False]])
    np.testing.assert_array_equal(touchdown, False)
    assert velocity[0, 0] > 0.0
    assert velocity[0, 1] < 0.0


def test_kinematic_airborne_state_uses_hysteresis(section01_env):
    """已离地足下降到低阈值前保持 airborne，防止阈值附近抖动。"""
    previous_airborne = np.ones((1, 4), dtype=np.bool_)
    previous_height = np.full((1, 4), 0.060, dtype=np.float32)
    current_height = np.array([[0.050, 0.036, 0.035, 0.020]], dtype=np.float32)

    airborne, liftoff, touchdown, _ = section01_env._kinematic_gait_events(
        current_height,
        previous_height,
        previous_airborne,
    )

    np.testing.assert_array_equal(airborne, [[True, True, False, False]])
    np.testing.assert_array_equal(liftoff, False)
    np.testing.assert_array_equal(touchdown, [[False, False, True, True]])


def test_every_configured_scale_has_a_matching_term(section01_env_stepped):
    """配置里的每一个奖励权重都必须对应一个真实计算出来的项。

    _compute_reward 对 reward_terms 里缺失的 key 是静默 continue 的
    （vbot_section01_np.py:783-791），所以拼错一个 key 不会报错，
    只会让那一项悄悄消失。这个测试就是那道防线。
    """
    env = section01_env_stepped
    configured = set(env._reward_scales.keys())
    produced = set(env.state.info["Reward"].keys())

    missing = configured - produced
    assert not missing, f"这些权重配了却没有对应的奖励项（会被静默忽略）：{sorted(missing)}"


def test_all_reward_terms_are_shape_n_and_finite(section01_env_stepped):
    """每个奖励项都必须是 (num_envs,) 的有限 float —— 形状错了会广播出灾难。"""
    env = section01_env_stepped
    num_envs = env.num_envs

    for key, value in env.state.info["Reward"].items():
        array = np.asarray(value)
        assert array.shape == (num_envs,), f"{key} 形状是 {array.shape}，应为 ({num_envs},)"
        assert np.all(np.isfinite(array)), f"{key} 含有 NaN/Inf"


def test_total_reward_stays_within_clip_bounds(section01_env_stepped):
    """总奖励必须落在 _compute_reward 末尾 np.clip(-100, 1000) 的界内。"""
    env = section01_env_stepped
    reward = env.state.reward
    assert reward.shape == (env.num_envs,)
    assert np.all(reward >= -100.0) and np.all(reward <= 1000.0)


def test_min_feet_air_time_is_zero_for_any_normal_gait(section01_env):
    """先把机制本身钉死：只要有任何一只脚着地，min(feet_air_time) 就是 0。

    这条是**绿的**，它陈述的是事实而非期望 —— 也正是 BUG-002 的根因。
    四足步态（trot / walk / crawl）在定义上始终至少有一只脚支撑，
    因此"四腿瞬时腾空时间取 min"这个量恒等于 0，除非出现四脚同时离地的飞行相。
    """
    assert float(np.min(TROT_AIR_TIME)) == 0.0
    # 只有四脚全部腾空时才非零
    all_airborne = np.array([0.30, 0.12, 0.08, 0.30], dtype=np.float32)
    assert float(np.min(all_airborne)) > 0.0


def test_leg_drive_terms_reward_a_healthy_trot(section01_env):
    """在落差段与坡道段做一个健康的 trot，腿部驱动奖励应当为正。

    这三项的设计意图（见 cfg.py 注释）是"逼每条腿都迈步，治后腿不动"。
    一个对角腿正常摆动的 trot 正是它们应当奖励的行为 —— 但因为用的是
    瞬时 min 而不是"每条腿最近是否迈过步"，它们给出 0。
    """
    env = section01_env

    # 落差段 Y ∈ (1.3, 1.8] -> drop_leg_catchup
    healthy_stale = np.full(4, 0.2, dtype=np.float32)
    drop_terms = _reward_terms_with_forced_gait(
        env, 1.55, TROT_AIR_TIME, TROT_CONTACTS, healthy_stale
    )
    assert np.max(drop_terms["drop_leg_catchup"]) > 0.0, "健康 trot 在落差段拿不到 drop_leg_catchup"

    # 坡道段 Y ∈ [1.8, 6.9) -> slope_leg_drive / slope_front_drive
    slope_terms = _reward_terms_with_forced_gait(
        env, 4.0, TROT_AIR_TIME, TROT_CONTACTS, healthy_stale
    )
    assert np.max(slope_terms["slope_leg_drive"]) > 0.0, "健康 trot 在坡道段拿不到 slope_leg_drive"
    assert np.max(slope_terms["slope_front_drive"]) > 0.0, "健康 trot 在坡道段拿不到 slope_front_drive"


def test_leg_drive_terms_only_fire_during_a_flight_phase(section01_env):
    """后腿完全离地时，坡上支撑推进奖励必须为零。"""
    env = section01_env
    flight = np.array([0.30, 0.12, 0.08, 0.30], dtype=np.float32)
    airborne = np.zeros(4, dtype=np.bool_)

    active_stale = np.full(4, 0.2, dtype=np.float32)
    drop_terms = _reward_terms_with_forced_gait(env, 1.55, flight, airborne, active_stale)
    slope_terms = _reward_terms_with_forced_gait(env, 4.0, flight, airborne, active_stale)

    assert np.max(drop_terms["drop_leg_catchup"]) > 0.0
    np.testing.assert_allclose(slope_terms["slope_leg_drive"], 0.0, atol=1e-7)
    assert np.max(slope_terms["slope_front_drive"]) > 0.0


def test_leg_drive_terms_require_support_not_stale_metadata(section01_env):
    """站立时坡上支撑奖励为零；stale 元数据不再伪造后腿驱动。"""
    env = section01_env
    standing = np.zeros(4, dtype=np.bool_)
    air_time = np.zeros(4, dtype=np.float32)

    standing_terms = _reward_terms_with_forced_gait(
        env,
        4.0,
        air_time,
        np.ones(4, dtype=np.bool_),
        np.full(4, 2.0, dtype=np.float32),
    )
    assert np.max(standing_terms["per_leg_swing"]) == 0.0
    zero_progress = env._rear_support_progress(
        np.zeros((env.num_envs,), dtype=np.float32),
        np.ones((env.num_envs, 4), dtype=np.bool_),
        np.ones((env.num_envs,), dtype=np.bool_),
    )
    np.testing.assert_allclose(zero_progress, 0.0, atol=1e-7)
    assert np.max(standing_terms["slope_front_drive"]) == 0.0

    stale_back_leg = np.array([0.2, 0.2, 2.0, 0.2], dtype=np.float32)
    stale_terms = _reward_terms_with_forced_gait(
        env, 4.0, TROT_AIR_TIME, TROT_CONTACTS, stale_back_leg
    )
    assert np.max(stale_terms["slope_leg_drive"]) > 0.0
    assert np.max(stale_terms["slope_front_drive"]) > 0.0


def test_per_leg_swing_uses_the_weakest_leg_air_time(section01_env):
    """四腿都迈过才得分；任意一腿没有有效腾空记录时整项归零。"""
    healthy_air_time = np.full(4, 0.30, dtype=np.float32)
    healthy_stale = np.full(4, 0.2, dtype=np.float32)
    healthy = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        healthy_stale,
        last_air_time=healthy_air_time,
    )
    one_dead_leg = healthy_air_time.copy()
    one_dead_leg[3] = 0.0
    dead_leg = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        healthy_stale,
        last_air_time=one_dead_leg,
    )

    assert np.min(healthy["per_leg_swing"]) > 0.0
    np.testing.assert_allclose(dead_leg["per_leg_swing"], 0.0, atol=1e-7)


def test_per_leg_swing_rewards_short_real_liftoffs(section01_env):
    """G6：0.09 s 的真实短迈步也要有连续梯度，不能被 0.12 s 硬阈值截断。"""
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.full(4, 0.2, dtype=np.float32),
        last_air_time=np.full(4, 0.09, dtype=np.float32),
    )
    expected = section01_env._reward_scales["per_leg_swing"] * 0.09 / section01_env._feet_air_time_target
    np.testing.assert_allclose(terms["per_leg_swing"], expected, rtol=1e-5, atol=1e-7)


def test_per_leg_swing_expires_when_one_leg_stops(section01_env):
    """旧腾空记录不能永久吃分；某腿长期不迈后最小腿评分必须降为零。"""
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.array([0.2, 0.2, 0.2, 2.0], dtype=np.float32),
        last_air_time=np.full(4, 0.30, dtype=np.float32),
    )
    np.testing.assert_allclose(terms["per_leg_swing"], 0.0, atol=1e-7)


def test_energy_penalty_is_removed(section01_env_stepped):
    """机械功率代理项不再参与训练奖励，避免鼓励省力拖腿。"""
    env = section01_env_stepped
    assert "energy" not in env._reward_scales
    assert "energy" not in env.state.info["Reward"]


def test_slope_hip_penalty_only_targets_outward_opening(section01_env):
    """坡外、内收和 0.2 rad 阈值内不罚；坡上过度外张才罚。"""
    env = section01_env
    assert len(env.hip_indices) == 4
    np.testing.assert_array_equal(env.hip_outward_signs, [-1.0, 1.0, -1.0, 1.0])

    dof_pos = np.tile(env.default_angles, (1, 1))
    hip_idx = np.asarray(env.hip_indices, dtype=np.int64)
    outward_sign = np.asarray(env.hip_outward_signs, dtype=np.float32)
    dof_pos[:, hip_idx] += 0.4 * outward_sign
    np.testing.assert_allclose(env._slope_hip_opening_penalty(dof_pos, np.array([0.0])), 0.0)

    inward_pos = np.tile(env.default_angles, (1, 1))
    inward_pos[:, hip_idx] -= 0.4 * outward_sign
    np.testing.assert_allclose(env._slope_hip_opening_penalty(inward_pos, np.array([4.0])), 0.0)

    threshold_pos = np.tile(env.default_angles, (1, 1))
    threshold_pos[:, hip_idx] += 0.2 * outward_sign
    np.testing.assert_allclose(env._slope_hip_opening_penalty(threshold_pos, np.array([4.0])), 0.0)
    assert env._slope_hip_opening_penalty(dof_pos, np.array([4.0]))[0] > 0.0


def test_grounded_robot_gets_no_swing_height_reward(section01_env):
    """四脚着地时没有摆动腿，不能利用 exp(0) 获得满额抬腿奖励。"""
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.full(4, 2.0, dtype=np.float32),
    )
    assert np.max(terms["swing_foot_height"]) == 0.0


@pytest.mark.parametrize("landing_air_time", [0.20, 0.30])
def test_normal_landing_air_time_is_rewarded_not_penalised(section01_env, landing_air_time):
    """0.20--0.30 秒的正常落地应获得非负且为正的有界奖励。"""
    air_time = np.zeros(4, dtype=np.float32)
    air_time[2] = landing_air_time
    first_contact = np.zeros(4, dtype=np.bool_)
    first_contact[2] = True
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        air_time,
        np.ones(4, dtype=np.bool_),
        first_contact=first_contact,
    )
    assert np.min(terms["feet_air_time"]) > 0.0
    assert np.max(terms["feet_air_time"]) <= 1.0


def test_air_time_reward_stays_bounded_when_multiple_feet_land(section01_env):
    """多只脚同帧落地时按落地腿平均，整项仍保持在 [0, 1]。"""
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        np.full(4, 0.30, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        first_contact=np.ones(4, dtype=np.bool_),
    )
    assert np.min(terms["feet_air_time"]) == pytest.approx(1.0)
    assert np.max(terms["feet_air_time"]) == pytest.approx(1.0)


def test_g3_reenables_stale_penalties_with_kinematic_events(section01_env):
    """G3 的可靠离地事件重新让逐腿 stale 惩罚参与优化。"""
    terms = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.array([0.2, 0.2, 1.0, 0.2], dtype=np.float32),
    )
    assert np.max(terms["leg_stale_penalty"]) < 0.0
    assert np.max(terms["rear_leg_stale_penalty"]) < 0.0


def test_g4_rear_balance_penalizes_asymmetric_airborne_state(section01_env):
    """RR/RL 只有一条离地时受罚，两条同步离地时该项为零。"""
    asymmetric = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        TROT_AIR_TIME,
        np.array([True, True, True, False], dtype=np.bool_),
        np.full(4, 0.2, dtype=np.float32),
    )
    paired = _reward_terms_with_forced_gait(
        section01_env,
        0.0,
        TROT_AIR_TIME,
        np.array([True, True, False, False], dtype=np.bool_),
        np.full(4, 0.2, dtype=np.float32),
    )
    assert np.max(asymmetric["rear_airborne_balance"]) < 0.0
    np.testing.assert_allclose(paired["rear_airborne_balance"], 0.0, atol=1e-7)


def test_front_stale_does_not_disable_rear_specific_drive(section01_env):
    """前腿停滞只影响前腿项，不能把落差/坡道的后腿驱动一起关掉。"""
    front_stale = np.array([1.0, 0.2, 0.2, 0.2], dtype=np.float32)
    drop_terms = _reward_terms_with_forced_gait(
        section01_env, 1.55, TROT_AIR_TIME, TROT_CONTACTS, front_stale
    )
    slope_terms = _reward_terms_with_forced_gait(
        section01_env, 4.0, TROT_AIR_TIME, TROT_CONTACTS, front_stale
    )
    assert np.min(drop_terms["drop_leg_catchup"]) > 0.0
    assert np.max(slope_terms["slope_leg_drive"]) > 0.0
    assert np.max(slope_terms["slope_front_drive"]) == 0.0


def test_slope_leg_drive_requires_rear_support_and_forward_progress(section01_env):
    """坡上后腿奖励必须同时依赖后腿接触和实际前向速度。"""
    env = section01_env
    forward_speed = np.array([0.4, 0.4, 0.0], dtype=np.float32)
    active_move = np.array([True, True, True], dtype=np.bool_)
    rear_supported = np.array(
        [[True, True, True, True], [True, True, False, False], [True, True, True, True]],
        dtype=np.bool_,
    )
    reward = env._rear_support_progress(forward_speed, rear_supported, active_move)
    assert reward[0] > 0.0
    assert reward[1] == 0.0
    assert reward[2] == 0.0


def test_gait_diagnostics_are_exposed_per_leg(section01_env_stepped):
    """训练日志必须同时暴露运动学步态和原始接触传感器。"""
    metrics = section01_env_stepped.state.info["metrics"]
    for leg_name in ("fr", "fl", "rr", "rl"):
        for prefix in (
            "leg_stale_time",
            "leg_contact",
            "leg_sensor_contact",
            "leg_airborne",
            "leg_liftoff",
            "leg_clearance",
            "leg_vertical_velocity",
            "leg_last_air_time",
            "leg_torque_abs",
            "leg_power_abs",
            "leg_contact_force_norm",
        ):
            key = f"{prefix}_{leg_name}"
            assert key in metrics
            assert np.asarray(metrics[key]).shape == (section01_env_stepped.num_envs,)
    for key in (
        "rear_airborne_balance",
        "rear_clearance_balance",
        "min_leg_swing_score",
        "rear_support_progress",
        "rear_torque_abs",
        "rear_power_abs",
        "rear_contact_force_norm",
    ):
        assert key in metrics
        assert np.asarray(metrics[key]).shape == (section01_env_stepped.num_envs,)


def test_a7_alive_rewards_do_not_pay_for_standing_still(section01_env):
    """A7 的三项存活奖励不能把站立不动当成有效步态。"""
    env = section01_env
    standing = _reward_terms_with_forced_gait(
        env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.full(4, 2.0, dtype=np.float32),
    )
    assert np.max(standing["gait_symmetry"]) == 0.0
    assert np.max(standing["tracking_yaw"]) > 0.0

    standing_idle = _reward_terms_with_forced_gait(
        env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.full(4, 2.0, dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
    )
    assert np.max(standing_idle["gait_symmetry"]) == 0.0
    assert np.max(standing_idle["tracking_yaw"]) == 0.0


def test_a7_trot_gets_gait_symmetry_but_standing_does_not(section01_env):
    """反相对角 trot 获得步态奖励，四脚同状态不获得。"""
    env = section01_env
    trot = _reward_terms_with_forced_gait(
        env, 0.0, TROT_AIR_TIME, TROT_CONTACTS, np.full(4, 0.2, dtype=np.float32)
    )
    standing = _reward_terms_with_forced_gait(
        env,
        0.0,
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.bool_),
        np.full(4, 2.0, dtype=np.float32),
    )
    assert np.max(trot["gait_symmetry"]) > 0.0
    assert np.max(standing["gait_symmetry"]) == 0.0


def test_a7_clearance_reward_continues_onto_slope(section01_env):
    """抬腿奖励不应在崎岖区出口突然归零，坡道仍需越障抬腿。"""
    env = section01_env
    rough = _reward_terms_with_forced_gait(
        env, 1.4, TROT_AIR_TIME, TROT_CONTACTS, np.full(4, 0.2, dtype=np.float32)
    )
    slope = _reward_terms_with_forced_gait(
        env, 4.0, TROT_AIR_TIME, TROT_CONTACTS, np.full(4, 0.2, dtype=np.float32)
    )
    assert np.max(rough["swing_foot_height"]) > 0.0
    assert np.max(slope["swing_foot_height"]) > 0.0


def test_a8_orientation_is_relative_to_slope(section01_env):
    """Terrain-matched pitch is cheaper than forcing a horizontal body on slope."""
    env = section01_env
    horizontal = _reward_terms_with_forced_gait(
        env,
        4.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        base_pitch=0.0,
    )
    terrain_matched = _reward_terms_with_forced_gait(
        env,
        4.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        base_pitch=np.deg2rad(15.0),
    )
    assert np.max(terrain_matched["orientation"]) > np.max(horizontal["orientation"])


def test_a8_orientation_still_penalises_roll(section01_env):
    """Terrain-relative pitch correction must not remove the roll penalty."""
    env = section01_env
    roll = _reward_terms_with_forced_gait(
        env,
        4.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        base_roll=np.deg2rad(10.0),
    )
    assert np.max(roll["orientation"]) < 0.0


def test_a8_slope_pitch_weight_is_disabled(section01_env):
    """The redundant fixed-pitch term contributes nothing in A8."""
    env = section01_env
    terms = _reward_terms_with_forced_gait(
        env,
        4.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        base_pitch=0.0,
    )
    assert np.max(np.abs(terms["slope_pitch"])) == 0.0


def test_a8_anti_stall_has_a_deadband(section01_env):
    """Tiny speed deficits are tolerated while larger deficits remain penalized."""
    env = section01_env
    small_deficit = _reward_terms_with_forced_gait(
        env,
        0.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        command=np.array([0.10, 0.0, 0.0], dtype=np.float32),
    )
    large_deficit = _reward_terms_with_forced_gait(
        env,
        0.0,
        TROT_AIR_TIME,
        TROT_CONTACTS,
        np.full(4, 0.2, dtype=np.float32),
        command=np.array([0.40, 0.0, 0.0], dtype=np.float32),
    )
    assert np.max(small_deficit["anti_stall"]) == 0.0
    assert np.max(large_deficit["anti_stall"]) < 0.0


def test_finishing_the_course_is_not_penalised(section01_env):
    """完赛的那一步不应该同时吃到 termination 惩罚。

    _compute_terminated 把 goal_done 也 OR 进了 terminated，导致
    "到达终点"和"摔倒"走同一条终止路径、共享同一个 -10 的 termination 惩罚。
    净额虽然仍是 +290，但这让 TensorBoard 上的 termination 曲线
    无法区分"摔了多少次"和"成功了多少次"。
    """
    from conftest import teleport_envs_along_course

    env = section01_env
    if env.state is None:
        env.init_state()
    num_envs = env.num_envs

    teleport_envs_along_course(env, np.full((num_envs,), 7.8, dtype=np.float32))
    info = env.state.info
    info["goal_done"] = np.ones((num_envs,), dtype=np.bool_)

    terminated = env._compute_terminated(env.state.data, info)
    assert np.all(terminated), "前提校验：goal_done 应当导致 terminated"

    env._compute_reward(env.state.data, info, terminated)
    termination_contrib = np.asarray(env.state.info["Reward"]["termination"])

    assert np.all(termination_contrib == 0.0), (
        f"完赛时仍被扣了 termination 惩罚 {termination_contrib[0]:.1f}，"
        "成功与摔倒共用了同一条终止路径"
    )
