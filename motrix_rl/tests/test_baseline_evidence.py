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

"""从**已提交的** B0 基线 tfevents 里读取实证，把审计结论锁成回归测试。

合成 rollout 只能证明某个奖励项"能不能"触发；要证明它在真实训练里
"有没有"触发，唯一可信的证据是训练日志本身。
docs/section01/tb/B0_baseline_15k/ 里那份 262KB 的 tfevents 就是这个证据，
它随仓库提交，因此这组测试对任何人都可复现，且完全不需要 GPU 或仿真。

数据来源：runs/vbot_navigation_section01/26-08-04_12-21-58-225910_PPO/
         4096 envs × 15000 iterations，约 64 分钟，torch 后端。
"""

import pathlib
import re

import pytest

from motrix_envs.navigation.vbot.vbot_section01_np import (
    TERRAIN_ROUGH_Y_END,
    TERRAIN_START_PLATFORM_Y_END,
)

BASELINE_TB_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "section01" / "tb" / "B0_baseline_15k"
)

# 权重非零、但整个训练过程中 episode 累计恒为 0.000 的奖励项。
DEAD_IN_TRAINING = ["drop_leg_catchup", "slope_leg_drive", "slope_front_drive"]


@pytest.fixture(scope="module")
def baseline_scalars():
    """把 B0 的全部标量读成 {tag: [values]}。"""
    pytest.importorskip("tensorboard", reason="需要 tensorboard 才能解析 tfevents")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(BASELINE_TB_DIR.glob("events.out.tfevents.*"))
    if not events:
        pytest.skip(f"基线 tfevents 不存在：{BASELINE_TB_DIR}")

    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {tag: [s.value for s in accumulator.Scalars(tag)] for tag in accumulator.Tags()["scalars"]}


def test_baseline_never_completed_the_course(baseline_scalars):
    """B0 的成功率自始至终是 0.000 —— 这是整个项目的出发点。"""
    success = baseline_scalars["metrics / goal_success_rate (mean)"]
    assert len(success) == 15, "标量点数变了，基线 tfevents 可能被换过"
    assert max(success) == 0.0, f"成功率峰值 {max(success)}，与审计结论不符"


def test_baseline_curves_have_only_fifteen_points(baseline_scalars):
    """BUG-010: 整整 15000 次迭代的训练，TensorBoard 上只有 15 个数据点。

    B0 记录的是拆分前的历史行为：`check_point_interval` 当时同时控制两件事
    （motrix_rl/skrl/torch/train/ppo.py:76-77）：
        cfg["experiment"]["write_interval"]      = rlcfg.check_point_interval
        cfg["experiment"]["checkpoint_interval"] = rlcfg.check_point_interval
    它被设成 1000 是为了"别存太多权重"，副作用是训练曲线的时间分辨率
    也被压到了 1/1000。当前实现已用独立的 `write_interval` 修复该耦合，
    这组断言只验证提交的 B0 历史证据没有被替换。

    Section01 当前通过独立字段实现更高日志分辨率，同时保持 checkpoint 数量。
    """
    for tag, values in baseline_scalars.items():
        if tag.startswith("metrics / ") or tag.startswith("Reward Total/"):
            assert len(values) == 15, f"{tag} 有 {len(values)} 个点"


def test_baseline_reached_at_most_waypoint_five_of_seven(baseline_scalars):
    """最好的 env 只到 waypoint 5（Y≈6.0，坡道中上部），共 7 个路径点。"""
    goal_idx_max = baseline_scalars["metrics / goal_idx (max)"]
    assert max(goal_idx_max) == 5.0
    assert baseline_scalars["metrics / goal_idx (mean)"][-1] == pytest.approx(2.82, abs=0.05)


def test_most_baseline_episodes_hit_the_step_limit_instead_of_terminating(baseline_scalars):
    """65% 的 episode 是被 4000 步截断的，不是摔倒终止 —— BUG-001 的指纹。

    平均 episode 长度 2906/4000 说明大量 env 在"活着但没进展"的状态里耗满时长。
    如果基座触地能正常终止，这个数字应该显著更低。
    """
    mean_len = baseline_scalars["Episode / Total timesteps (mean)"][-1]
    assert mean_len == pytest.approx(2906, rel=0.02)
    assert mean_len / 4000.0 > 0.7, "平均 episode 长度占比不再异常，审计结论需要复核"


def test_leg_drive_reward_terms_were_identically_zero_during_training(baseline_scalars):
    """BUG-002 的实证：三个腿部驱动项在 15000 次迭代里累计值恒为 0。

    合计 10.0 的奖励权重完全空转 —— 它们在 TensorBoard 上是三条完美的直线。
    """
    for term in DEAD_IN_TRAINING:
        tag = f"Reward Total/ {term} (mean)"
        assert tag in baseline_scalars, f"基线里找不到 {tag}"
        values = baseline_scalars[tag]
        assert max(abs(v) for v in values) == 0.0, f"{term} 在训练中并非恒为 0，BUG-002 的结论需要复核"


def test_termination_and_base_contact_curves_are_identical(baseline_scalars):
    """BUG-009 的实证：两条曲线逐点重合，说明同一事件被扣了两次分。

    也顺带证明 speed_overflow / invalid / goal_done 三条终止路径从未触发过。
    """
    termination = baseline_scalars["Reward Total/ termination (mean)"]
    base_contact = baseline_scalars["Reward Total/ base_contact (mean)"]
    assert termination == base_contact, "两条曲线不再重合，终止路径的构成发生了变化"


def test_gait_symmetry_was_pinned_at_its_theoretical_maximum(baseline_scalars):
    """BUG-003 的实证：gait_symmetry 的 max 恰好等于 0.2 × 4000 = 800。

    实测 800.03，即存在 env 在整整 4000 步里几乎每一步都拿满这一项 ——
    而该项在"四脚都着地"时取最大值，也就是说：原地站着满分。
    """
    peak = max(baseline_scalars["Reward Total/ gait_symmetry (max)"])
    theoretical_max = 0.2 * 4000  # 权重 × episode 步数上限
    assert peak == pytest.approx(theoretical_max, rel=1e-3), f"实测峰值 {peak}，不再顶在 0.2×4000 的上限"


def test_dof_acc_is_the_largest_term_by_magnitude(baseline_scalars):
    """BUG-007 的实证：一个权重 -2.5e-7 的"轻微正则项"是绝对值最大的奖励项。

    100Hz 下 dof_acc = Δv/0.01 量级达 1e3~1e4，平方后 1e6~1e8，再乘 12 个关节。
    结果是最强的学习信号在惩罚"运动本身"。
    """
    finals = {
        tag: values[-1]
        for tag, values in baseline_scalars.items()
        if tag.startswith("Reward Total/") and tag.endswith("(mean)")
    }
    largest = max(finals, key=lambda t: abs(finals[t]))
    assert "dof_acc" in largest, f"绝对值最大的项变成了 {largest}（{finals[largest]:.1f}）"
    assert finals[largest] == pytest.approx(-2157, rel=0.02)


def test_episode_length_dips_when_the_policy_is_on_the_covered_terrain(baseline_scalars):
    """BUG-001 的训练侧证据：episode 长度在训练中段有一个明显的谷。

    序列（每 1000 iter 一个点）：
        357, 1080, 961, 806, [410], 580, 687, 740, 885, 1058, 1384, 1945, 2564, 2808, 2906
                              ^^^ iter 5000

    第一个点（iter 1000）要排除：此时 Policy/Standard deviation = 2.637，
    策略基本是噪声，摔在哪里都一样快，不反映地形。

    从第二个点起，episode 长度先跌到 iter 5000 的 410 步、再单调涨到 2906。
    这个谷与逐 checkpoint 评估（docs/section01/eval/B0_sweep.json）里
    截断率跌到 0% 的窗口**完全重合**（iter 4000 = 379 步 / 0.0% 截断）——
    那正是策略恰好待在唯一被终止检测覆盖的崎岖 hfield 上的时候。

    离开那块地面（爬上坡道）之后，摔倒不再终止，episode 长度一路涨到 2906/4000。

    修好 BUG-001 之后这个谷应当消失：episode 长度会单调反映真实存活能力。
    """
    lengths = baseline_scalars["Episode / Total timesteps (mean)"][1:]
    trough_index = min(range(len(lengths)), key=lambda i: lengths[i])

    assert 0 < trough_index < len(lengths) - 1, f"没有中段谷底，最小值在第 {trough_index} 个点"
    assert lengths[trough_index] == pytest.approx(410, rel=0.05), f"谷底值 {lengths[trough_index]}"
    assert lengths[-1] > lengths[trough_index] * 5, "尾段没有远高于谷底"


def test_checkpoint_sweep_truncation_matches_termination_coverage():
    """BUG-001 最强的一份证据：截断率与终止检测的覆盖边界严格对应。

    逐 checkpoint 的确定性评估（docs/section01/eval/B0_sweep.json，随仓库提交）：

        iter  1000  Y=-1.90  截断 92.2%   起步平台（未覆盖）
        iter  4000  Y=-1.22  截断  0.0%   崎岖 hfield（已覆盖）
        iter 15000  Y= 6.23  截断 81.2%   坡道（未覆盖）

    策略恰好待在全赛道唯一被覆盖的那块地面上时，截断率跌到 0；一离开就回升。
    这个形状不可能是巧合 —— 它就是覆盖边界在数据上的印记。
    """
    sweep_file = BASELINE_TB_DIR.parents[1] / "eval" / "B0_sweep.json"
    if not sweep_file.exists():
        pytest.skip(f"扫描结果不存在：{sweep_file}")

    import json

    results = json.loads(sweep_file.read_text())["results"]
    by_iter = {}
    for result in results:
        iteration = int(re.search(r"agent_(\d+)", result["checkpoint"]).group(1))
        by_iter[iteration] = (
            result["final_y"]["p50"],
            result["episode_len_steps"]["frac_truncated"],
        )

    assert len(by_iter) == 15, f"扫描结果有 {len(by_iter)} 个 checkpoint"
    assert all(r["success_rate"] == 0.0 for r in results), "有 checkpoint 完赛了？"

    # 覆盖区内（崎岖 hfield，-1.5 <= Y < 1.5）的截断率应当远低于两端
    on_covered = [t for y, t in by_iter.values() if TERRAIN_START_PLATFORM_Y_END <= y < TERRAIN_ROUGH_Y_END]
    off_covered = [t for y, t in by_iter.values() if not (TERRAIN_START_PLATFORM_Y_END <= y < TERRAIN_ROUGH_Y_END)]

    assert on_covered and off_covered, "分组为空，数据形状变了"
    assert min(on_covered) == pytest.approx(0.0, abs=1e-9), "覆盖区内截断率没有跌到 0"
    assert max(by_iter[1000][1], by_iter[15000][1]) > 0.8, "两端截断率没有高于 80%"
    assert min(on_covered) < min(off_covered), "覆盖区内的截断率没有低于覆盖区外"


def test_policy_std_was_still_decreasing_when_training_stopped(baseline_scalars):
    """BUG-006/假设7 的实证：训练结束时策略标准差仍在下降 —— 是被掐断的，不是收敛的。"""
    std = baseline_scalars["Policy / Standard deviation"]
    assert std[0] == pytest.approx(2.637, abs=0.01)
    assert std[-1] == pytest.approx(1.113, abs=0.01)
    # 最后 10% 仍在明显下降
    tail = std[int(len(std) * 0.9) :]
    assert tail[0] - tail[-1] > 0.01, "尾段已经平了，'训练被提前掐断'的结论需要复核"
