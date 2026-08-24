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

"""Section01 环境的只读诊断工具。

不训练、不改状态，只把环境内部的接线摊开来看。用法：

    uv run scripts/diag_section01.py --section all
    uv run scripts/diag_section01.py --section contacts
    uv run scripts/diag_section01.py --section terrain --probe   # 含投放探针实测(慢)

四个诊断面：
    model     模型规模、geom/actuator 命名、DOF 布局
    contacts  基座触地检测到底覆盖了哪些地面（BUG-001 的决定性输出）
    terrain   地形采样 vs 真实几何 vs 投放探针实测，三方对照
    rewards   奖励项沿赛道各处的取值表 —— 死项会是一整列 0
"""

import contextlib
import os
import sys

import numpy as np
from absl import app, flags

from motrix_envs import registry
from motrix_envs.navigation.vbot.vbot_section01_np import (
    TERRAIN_PLATFORM_HEIGHT,
    section01_terrain_height,
)

_ENV = flags.DEFINE_string("env", "vbot_navigation_section01", "要诊断的环境名")
_SECTION = flags.DEFINE_string("section", "all", "model / contacts / terrain / rewards / all")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 32, "并行环境数（rewards 面用它来铺满赛道）")
_PROBE = flags.DEFINE_bool("probe", False, "terrain 面是否跑投放探针实测（慢，约 1 分钟）")

# 从 xmls/0126_C_section01.xml 抄下来的可行走面。这些 box 全部由
# <default class="..."> 声明，没有 name 属性 —— 这正是 BUG-001 的成因。
WALKABLE_SURFACES = [
    ("C_Adiban_001", -3.5, -1.5, "起步平台        z=0.000"),
    ("C_Adiban_005", -1.5, 1.5, "崎岖区基座      z=0.000"),
    ("hfield", -1.5, 1.5, "崎岖起伏        z<=0.277"),
    ("C_Adiban_002", 1.5, 2.0, "平地缺口(落差)  z=0.000"),
    ("C_Adiban_003", 2.0, 6.8296, "15°坡道         z 0->1.294"),
    ("C_Adiban_004", 6.8296, 8.8296, "终点平台        z=1.294"),
]

RULE = "=" * 78


def _header(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


# --------------------------------------------------------------------------- model


def diag_model(env) -> None:
    _header("MODEL — 模型规模与命名")
    model = env.model

    named = [n for n in model.geom_names if n]
    print(f"geoms      : {model.num_geoms}  (具名 {len(named)}, 无名 {model.num_geoms - len(named)})")
    print(f"bodies     : {model.num_bodies}")
    print(f"actuators  : {model.num_actuators}")
    print(f"sensors    : {model.num_sensors}")
    print(f"dof_pos    : {model.num_dof_pos}     dof_vel: {model.num_dof_vel}")
    print(f"obs / act  : {env.observation_space.shape[0]} / {env.action_space.shape[0]}")

    print("\n执行器顺序（_init_buffers 靠子串匹配认关节，顺序错了默认角就会错配）：")
    for idx, name in enumerate(model.actuator_names):
        marker = []
        if idx in env.hip_indices:
            marker.append("hip")
        if idx in env.calf_indices:
            marker.append("calf")
        print(
            f"  [{idx:2d}] {str(name):24s} default={env.default_angles[idx]:+.3f}"
            f"  limits=[{env.joint_lower_limits[idx]:+.4f}, {env.joint_upper_limits[idx]:+.4f}]"
            f"  {','.join(marker)}"
        )

    print("\n具名 geom（碰撞资产里只有 hfield 有名字，坡道/平台的 box 全是无名的）：")
    for name in named:
        print(f"  {name}")


# ------------------------------------------------------------------------ contacts


def diag_contacts(env) -> None:
    _header("CONTACTS — 基座触地检测的覆盖范围（BUG-001）")
    model = env.model

    base_geoms = list(env._cfg.asset.terminate_after_contacts_on)
    ground_idx = sorted({int(i) for i in env.termination_contact[:, 1]}) if env.num_termination_check else []
    ground_names = [model.geom_names[i] for i in ground_idx]
    uses_sensor_fallback = env.num_termination_check == 0

    print(f"termination_contact 对数 : {env.num_termination_check}")
    print(f"  基座侧 geom : {base_geoms}")
    print(f"  地面侧 geom : {ground_names}")
    print(f"  地面侧是否全为 hfield : {all('dixing_Plane' in n for n in ground_names) if ground_names else 'N/A'}")

    print("\n按可行走面逐一核对覆盖情况：")
    covered_any_box = False
    for class_name, y_lo, y_hi, desc in WALKABLE_SURFACES:
        # 名字扫描只能命中 hfield（C{1,2,3}_V_*dixing_Plane），box 一律无名
        covered = uses_sensor_fallback or (class_name == "hfield" and bool(ground_names))
        covered_any_box = covered_any_box or (covered and class_name != "hfield")
        flag = "covered    " if covered else "NOT COVERED"
        print(f"  {flag}  {class_name:14s} Y[{y_lo:6.2f},{y_hi:6.2f}]  {desc}")

    print(f"\nundesired_contact 对数   : {env.num_undesired_contact_check}")
    print(f"  配置里列了 {len(env._cfg.asset.undesired_contacts_on)} 个 geom 名，"
          f"实际匹配到 {env.num_undesired_contact_check // max(len(ground_idx), 1)} 个")

    print("\nfallback（覆盖完整的传感器路径）：")
    print(f"  base_contact 传感器 : {env._base_contact_sensors}")
    print("  它们在 scene_section01.xml 用 subtree2=\"C{1,2,3}_ground_root\" 声明，")
    print(f"  覆盖整棵碰撞子树（全部 {model.num_geoms} 个 geom），不受 geom 有没有名字影响。")
    if env.num_termination_check > 0:
        print(f"  ⚠ 但 _get_base_contacts 的判据是 num_termination_check > 0（当前 {env.num_termination_check}），")
        print("    所以永远走 geom 路径，这条完整的 fallback 不可达。")
    else:
        print("  ✓ 当前使用 base_contact 传感器 fallback，覆盖全部碰撞子树。")

    if uses_sensor_fallback:
        print("\n结论：base_contact 传感器现在覆盖起步平台、崎岖区、落差、坡道和终点平台。")
    else:
        print("\n结论：狗在起步平台/坡道/终点平台上肚皮着地时，既不会终止 episode，")
        print("      也收不到 base_contact 惩罚 —— 它可以趴在那里白拿满 40 秒的存活奖励。")


# ------------------------------------------------------------------------- terrain


@contextlib.contextmanager
def _suppress_native_stdout():
    """在 fd 层面吞掉物理引擎的诊断输出。

    3 米自由落体砸到地面时，求解器偶尔会把一个 18x18 的矩阵打到 stdout。
    那是 Rust 侧直接写文件描述符的，contextlib.redirect_stdout 拦不住，
    只能在 fd 层面重定向。纯属噪音，不影响结果。
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)


def _probe_ground_height(env, y: float, settle_steps: int = 400) -> float:
    """把机器人零力矩投放到 (0, y, 3.0)，自由落体到静止，读稳定后的基座高度。

    这是最笨但零假设的地面高度测量法 —— 它连"我对碰撞几何的解读"都不相信，
    直接问物理引擎"这里的地面到底在哪"。
    """
    data = env.state.data
    dof_pos = np.array(data.dof_pos, dtype=np.float32, copy=True)
    dof_pos[:, 4] = y
    dof_pos[:, 5] = 3.0
    data.set_dof_pos(dof_pos, env.model)
    data.set_dof_vel(np.zeros_like(np.asarray(data.dof_vel)))
    env.model.forward_kinematic(data)

    zero = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)
    with _suppress_native_stdout():
        for _ in range(settle_steps):
            data.actuator_ctrls = zero
            env.model.step(data)

    return float(np.median(env._body.get_pose(data)[:, 2]))


def diag_terrain(env, run_probe: bool) -> None:
    _header("TERRAIN — 地形采样 vs 真实几何" + ("（含投放探针实测）" if run_probe else ""))

    print("_get_terrain_scan 使用真实 hfield 双线性插值与闭式坡道剖面，不是运行时射线检测。")
    print(f"高度统一除以真实平台高度 {TERRAIN_PLATFORM_HEIGHT:.4f} m 做归一化。\n")

    y_values = np.arange(-3.0, 9.01, 0.25)
    print(f"{'base_Y':>7} {'scan[0]':>8} {'scan_std':>9} {'真值/PH':>9} {'误差':>8}   备注")
    print("-" * 78)

    worst_err, worst_y = 0.0, None
    for base_y in y_values:
        scan = env._get_terrain_scan(np.array([[0.0, base_y]], dtype=np.float32))[0]
        sample_y = base_y + env.TERRAIN_SCAN_OFFSETS
        truth = section01_terrain_height(sample_y) / TERRAIN_PLATFORM_HEIGHT
        err = float(np.max(np.abs(scan - truth)))
        if err > worst_err:
            worst_err, worst_y = err, base_y

        note = ""
        # 平台上 8 点相同是合理的（地面本来就平）；崎岖区 8 点相同才是问题，
        # 因为那里地面并不平，而观测里又没有绝对位置可用来定位。
        if np.std(scan) < 1e-9 and -1.5 < base_y < 1.5:
            note = "崎岖区采样全相同（异常）"
        if 1.3 <= base_y <= 1.9:
            note = "落差坎附近"
        print(f"{base_y:7.2f} {scan[0]:8.4f} {np.std(scan):9.4f} {truth[0]:9.4f} {scan[0] - truth[0]:+8.4f}   {note}")

    print("-" * 78)
    print(
        f"全程最大误差 {worst_err:.4f}（归一化）= {worst_err * TERRAIN_PLATFORM_HEIGHT:.4f} m，"
        f"出现在 base_Y={worst_y:.2f}"
    )
    print("\n要点：")
    print("  1. 坡道段其实很准（归一化梯度误差仅 1.44%）—— 这部分作者做对了。")
    print("  2. 崎岖区使用真实 hfield 插值，8 个采样点应随地形起伏变化。")
    print("  3. Y=1.5 的落差坎由闭式真值剖面显式呈现。")

    if not run_probe:
        print("\n（加 --probe 可跑投放探针实测，用物理引擎独立验证上面的真值列）")
        return

    _header("TERRAIN PROBE — 投放探针实测（零假设）")
    print(f"{'Y':>7} {'探针 base_z':>12} {'闭式真值':>10} {'差值':>8}")
    print("-" * 46)
    for base_y in [-2.5, -1.0, 0.0, 1.0, 1.7, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]:
        settled_z = _probe_ground_height(env, base_y)
        truth = float(section01_terrain_height(np.array([base_y]))[0])
        print(f"{base_y:7.2f} {settled_z:12.4f} {truth:10.4f} {settled_z - truth:+8.4f}")
    print("\n差值应当近似等于机器人趴下时的基座离地高度（各处一致即说明真值列正确）。")


# ------------------------------------------------------------------------- rewards


ZONES = [
    ("起步/崎岖 Y<1.3", lambda y: y < 1.3),
    ("落差 1.3~1.8", lambda y: (y >= 1.3) & (y <= 1.8)),
    ("坡道 1.8~6.8", lambda y: (y > 1.8) & (y < 6.83)),
    ("平台 Y>6.83", lambda y: y >= 6.83),
]

# 四腿/前后腿活跃度塑形项 —— 用于核对 BUG-002 修复后是否真正产生信号
LEG_DRIVE_FAMILY = {"per_leg_swing", "drop_leg_catchup", "slope_leg_drive", "slope_front_drive"}


def diag_rewards(env, rounds: int = 6, steps_per_round: int = 60) -> None:
    _header(f"REWARDS — {len(env._reward_scales)} 个奖励项沿赛道的取值（BUG-002）")

    num_envs = env.num_envs
    y_grid = np.linspace(-2.4, 8.0, num_envs).astype(np.float32)
    rng = np.random.default_rng(0)
    num_actions = env.action_space.shape[0]

    # (奖励项, 区段) -> 加权贡献的峰值绝对值
    peak = {}

    for _ in range(rounds):
        # 每轮重新把 env 铺满赛道（否则它们会摔倒重置、全都挤回起点）
        data = env.state.data
        dof_pos = np.array(data.dof_pos, dtype=np.float32, copy=True)
        dof_pos[:, 4] = y_grid
        dof_pos[:, 5] = section01_terrain_height(y_grid) + 0.42
        data.set_dof_pos(dof_pos, env.model)
        env.model.forward_kinematic(data)

        for _ in range(steps_per_round):
            env.step(rng.uniform(-1.0, 1.0, size=(num_envs, num_actions)).astype(np.float32))
            actual_y = env._body.get_pose(env.state.data)[:, 1]
            for key, value in env.state.info["Reward"].items():
                contribution = np.abs(np.asarray(value, dtype=np.float64))
                for zone_name, predicate in ZONES:
                    mask = predicate(actual_y)
                    if not np.any(mask):
                        continue
                    magnitude = float(np.max(contribution[mask]))
                    slot = (key, zone_name)
                    if magnitude > peak.get(slot, 0.0):
                        peak[slot] = magnitude

    print(f"场景：{num_envs} 个 env 沿赛道铺开，随机动作跑 {rounds}×{steps_per_round} 步。")
    print("按每一步的**实际** Y 归入区段，取加权贡献绝对值的峰值。")
    print("一整行全 0 = 这个奖励项在任何地方都没起过作用。\n")

    print(f"{'奖励项':<22}{'权重':>10} " + "".join(f"{name:>17}" for name, _ in ZONES))
    print("-" * 98)

    def overall(key):
        return max(peak.get((key, zone), 0.0) for zone, _ in ZONES)

    dead = []
    for key in sorted(env._reward_scales, key=lambda k: -overall(k)):
        scale = float(env._reward_scales[key])
        row = "".join(f"{peak.get((key, zone), 0.0):>17.5f}" for zone, _ in ZONES)
        print(f"{key:<22}{scale:>10.2e} {row}")
        if overall(key) == 0.0 and scale != 0.0:
            dead.append(key)

    print("-" * 98)
    if not dead:
        print("\n所有奖励项在赛道上都至少非零过一次。")
        return

    print("\n⚠ 权重非零、却在整条赛道任何区段都恒为 0 的项（完全空转）：")
    for key in dead:
        print(f"    {key:<22} 权重 {env._reward_scales[key]:+.2f}")
    wasted = sum(abs(env._reward_scales[k]) for k in dead)
    print(f"  合计空转权重 {wasted:.1f}")


def diag_leg_drive_gait_contrast(env) -> None:
    """BUG-002 的决定性对照：同一个奖励项，两种步态假设下的取值。

    上面那张表用的是随机动作 —— 机器人被乱抡到空中，四脚同时离地，
    于是腿部驱动项"看起来是活的"。这恰恰是最容易误导人的地方：
    真正在训练里出现的是四足步态，永远至少有一只脚支撑。
    """
    _header("REWARDS — 腿部驱动项的步态对照（BUG-002 的决定性证据）")

    num_envs = env.num_envs
    # 每个场景除了瞬时步态相位，还要给出"每条腿距上次离地多久"(_leg_stale_time)，
    # 因为修复后的奖励度量的是持续行为而不是某一瞬间的接触状态。
    # (标签, feet_air_time, contacts, leg_stale_time, 最近一次有效腾空时长)
    scenarios = [
        (
            "健康 trot（对角腿交替摆动）",
            np.array([0.30, 0.0, 0.0, 0.30], dtype=np.float32),
            np.array([False, True, True, False]),
            np.array([0.00, 0.18, 0.18, 0.00], dtype=np.float32),
            np.full(4, 0.30, dtype=np.float32),
        ),
        (
            "三脚支撑 walk（四腿轮流摆动）",
            np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([False, True, True, True]),
            np.array([0.00, 0.15, 0.30, 0.45], dtype=np.float32),
            np.full(4, 0.25, dtype=np.float32),
        ),
        (
            "飞行相（四脚同时离地）",
            np.array([0.30, 0.12, 0.08, 0.30], dtype=np.float32),
            np.zeros(4, dtype=np.bool_),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
        (
            "★ 后腿不动（前腿迈步，后腿 2 秒没抬）",
            np.array([0.20, 0.20, 0.0, 0.0], dtype=np.float32),
            np.array([False, False, True, True]),
            np.array([0.00, 0.00, 2.00, 2.00], dtype=np.float32),
            np.array([0.20, 0.20, 0.0, 0.0], dtype=np.float32),
        ),
        (
            "四脚全部站立不动（已 2 秒）",
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.bool_),
            np.full(4, 2.0, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
    ]

    # 落差段取 Y=1.55，坡道段取 Y=4.0，两个分区奖励都能激活
    probes = [("落差段 Y=1.55", 1.55, "drop_leg_catchup"),
              ("坡道段 Y=4.00", 4.00, "slope_leg_drive"),
              ("坡道段 Y=4.00", 4.00, "slope_front_drive"),
              ("落差段 Y=1.55", 1.55, "per_leg_swing")]

    print(f"{'步态假设':<44}" + "".join(f"{term:>20}" for _, _, term in probes))
    print("-" * 124)

    rows = {}
    for label, air_time, contacts, leg_stale, last_air_time in scenarios:
        cells = []
        for _, base_y, term in probes:
            data = env.state.data
            dof_pos = np.array(data.dof_pos, dtype=np.float32, copy=True)
            dof_pos[:, 4] = base_y
            dof_pos[:, 5] = float(section01_terrain_height(np.array([base_y]))[0]) + 0.42
            data.set_dof_pos(dof_pos, env.model)
            env.model.forward_kinematic(data)

            info = env.state.info
            info["feet_air_time"] = np.tile(air_time, (num_envs, 1))
            info["contacts"] = np.tile(contacts, (num_envs, 1))
            info["first_contact"] = np.zeros((num_envs, 4), dtype=np.bool_)
            info["air_time_before_contact"] = np.tile(air_time, (num_envs, 1))
            info["_leg_stale_time"] = np.tile(leg_stale, (num_envs, 1))
            info["_leg_last_air_time"] = np.tile(last_air_time, (num_envs, 1))
            info["commands"] = np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (num_envs, 1))

            env._compute_reward(data, info, np.zeros((num_envs,), dtype=np.bool_))
            cells.append(float(np.max(np.abs(np.asarray(info["Reward"][term])))))
        rows[label] = cells
        print(f"{label:<44}" + "".join(f"{c:>20.5f}" for c in cells))

    print("-" * 124)

    # 依据实测结果给结论，而不是写死一段叙述 —— 你改完奖励函数后
    # 这段话会自动跟着变，不需要回来改脚本。
    def row_sum(label):
        return sum(rows.get(label, []))

    trot = row_sum("健康 trot（对角腿交替摆动）")
    walk = row_sum("三脚支撑 walk（四腿轮流摆动）")
    flight = row_sum("飞行相（四脚同时离地）")
    lazy = row_sum("★ 后腿不动（前腿迈步，后腿 2 秒没抬）")
    stand = row_sum("四脚全部站立不动（已 2 秒）")

    print("\n读法：")
    healthy_gaits_rewarded = trot > 0 and walk > 0
    discriminates = (trot > 0) and (lazy < trot * 0.5)

    if not healthy_gaits_rewarded and flight > 0:
        print("  ✗ 正常步态一分拿不到，只有'飞行相'非零 —— 这四项**度量错了对象**。")
        print("    公式是 min(四条腿的瞬时腾空时间)：任一脚触地就把该腿计时清零，")
        print("    所以四腿取 min 只在'四脚同时离地'时非零。而四足越障步态")
        print("    (trot/walk/crawl) 在定义上始终至少有一只脚支撑 —— 于是恒为 0。")
        print("    实测：B0 基线 15000 次迭代里这三项的 episode 累计值恒为 0.000。")
        if lazy > trot:
            print()
            print(f"    ⚠ 更糟的是：'★后腿不动'({lazy:.2f}) 竟然比'健康 trot'({trot:.2f}) 拿得**更多**。")
            print("      因为 slope_front_drive 只看前两腿，而'后腿不动'时前腿双双腾空、")
            print("      min(前两腿) > 0 —— 这一项正在**奖励**它要治的那个失败模式。")
            print("      cfg.py 的注释写着'逼每条腿都迈步，治后腿不动'，公式做的却是反的。")
    elif discriminates:
        print(f"  ✓ 正常步态有分 (trot {trot:.2f} / walk {walk:.2f})，")
        print(f"    目标失败模式被压制 (后腿不动 {lazy:.2f} / 站立不动 {stand:.2f})。")
        print("    奖励现在能区分'每条腿都在迈'和'后腿不动'了。")
    else:
        print(f"  ⚠ trot={trot:.2f} walk={walk:.2f} 后腿不动={lazy:.2f} 站立={stand:.2f}")
        print("    正常步态与目标失败模式的差距不明显，检查一下判据是否真的在度量'每条腿'。")

    print("\n  练习见 docs/section01/05-exercises.md 的 EX-2。")


# ---------------------------------------------------------------------------- main


def main(argv):
    del argv
    section = _SECTION.value.lower()
    valid = {"model", "contacts", "terrain", "rewards", "all"}
    if section not in valid:
        raise SystemExit(f"--section 必须是 {sorted(valid)} 之一，收到 {section!r}")

    num_envs = _NUM_ENVS.value if section in ("rewards", "all") else 2
    print(f"正在构建 {_ENV.value}（{num_envs} 个环境，加载 265MB 网格需要几秒）…")
    env = registry.make(_ENV.value, sim_backend="np", num_envs=num_envs)
    env.init_state()

    if section in ("model", "all"):
        diag_model(env)
    if section in ("contacts", "all"):
        diag_contacts(env)
    if section in ("terrain", "all"):
        diag_terrain(env, _PROBE.value)
    if section in ("rewards", "all"):
        diag_rewards(env)
        diag_leg_drive_gait_contrast(env)

    print()


if __name__ == "__main__":
    app.run(main)
