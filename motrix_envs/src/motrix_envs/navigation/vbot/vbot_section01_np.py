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

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import VBotSection01EnvCfg

# ============================================================================
# Section01 赛道真实几何 —— 从 xmls/0126_C_section01.xml 的碰撞体反算得到。
#
# 相关碰撞体（均为 <default class> 声明的无名 box，见 BUG-001）：
#   C_Adiban_001  size(5, 1.0,  0.25) pos(0, -2.5,      -0.25)  -> 起步平台, 顶面 z=0
#   C_Adiban_005  size(5, 1.5,  0.25) pos(0,  0.0,      -0.25)  -> 崎岖区基座, 顶面 z=0
#   hfield        size "5 1.5 0.277056"                         -> 崎岖起伏, 0 ~ 0.277056
#   C_Adiban_002  size(5, 0.25, 0.25) pos(0,  1.75,     -0.25)  -> 平地缺口, 顶面 z=0
#   C_Adiban_003  size(5, 2.5,  0.25) pos(0,  4.4795189, 0.4055660)
#                 quat(0.9914449, 0.1305262, 0, 0) = R_x(15°)   -> 15° 坡道
#   C_Adiban_004  size(5, 1.0,  0.25) pos(0,  7.8296285, 1.0440952) -> 终点平台, 顶面 z=1.2940952
#
# 坡道顶面推导：
#   顶面中心 = pos + R_x(15°)·(0, 0, 0.25) = (0, 4.4148141, 0.6470474)
#   沿坡方向 = R_x(15°)·(0, 1, 0) = (0, cos15°, sin15°)，半长 2.5
#   低端 = (0, 2.0000000, 0.0000000)      高端 = (0, 6.8296282, 1.2940948)
# 自洽性校验：坡顶 z 1.2940948 == C_Adiban_004 顶面 1.2940952（差 4e-7）。
#
# 注意：代码中原有的 ROUGH_Y_RANGE=(-1.65, 1.85)、坡道终点 6.9、platform_h=0.66
# 都与此处的真值不符，详见 docs/section01/02-bugs/。
# ============================================================================
TERRAIN_START_PLATFORM_Y_END = -1.5       # 起步平台 / 崎岖区 分界
TERRAIN_ROUGH_Y_END = 1.5                 # 崎岖 hfield 终点（真实 hfield 半径 1.5）
TERRAIN_ROUGH_MAX_HEIGHT = 0.277056       # hfield 最大起伏高度
TERRAIN_SLOPE_Y_START = 2.0               # 坡道低端
TERRAIN_SLOPE_Y_END = 6.8296282           # 坡道高端
TERRAIN_SLOPE_ANGLE_RAD = np.deg2rad(15.0)
TERRAIN_SLOPE_TAN = float(np.tan(TERRAIN_SLOPE_ANGLE_RAD))   # 0.2679492
TERRAIN_PLATFORM_HEIGHT = 1.2940948       # 终点平台顶面高度
TERRAIN_FINISH_Y_END = 8.8296282          # 终点平台末端


def section01_terrain_height(y: np.ndarray) -> np.ndarray:
    """Section01 赛道沿 Y 的**真实**可行走高度（米），由碰撞几何闭式给出。

    这是赛道地形的唯一真值来源，供地形观测、诊断脚本与测试共用。
    崎岖区返回其基座高度 0.0（hfield 的逐点起伏另见 TERRAIN_ROUGH_MAX_HEIGHT），
    因为 hfield 的具体高度需要查纹理，此处只描述宏观剖面。
    """
    y = np.asarray(y, dtype=np.float32)
    h = np.zeros_like(y, dtype=np.float32)
    # 坡道：线性上升
    on_slope = np.logical_and(y >= TERRAIN_SLOPE_Y_START, y < TERRAIN_SLOPE_Y_END)
    h = np.where(on_slope, (y - TERRAIN_SLOPE_Y_START) * TERRAIN_SLOPE_TAN, h)
    # 终点平台：恒定高度
    h = np.where(y >= TERRAIN_SLOPE_Y_END, TERRAIN_PLATFORM_HEIGHT, h)
    return h.astype(np.float32)


def section01_expected_pitch(y: np.ndarray) -> np.ndarray:
    """贴合地面时 projected_gravity[:, 0] 的期望值（坡道上为 sin15°=0.2588）。

    用于把"顺坡贴合"与"往前栽"区分开 —— 见 BUG-005。
    """
    y = np.asarray(y, dtype=np.float32)
    on_slope = np.logical_and(y >= TERRAIN_SLOPE_Y_START, y < TERRAIN_SLOPE_Y_END)
    return np.where(on_slope, np.float32(np.sin(TERRAIN_SLOPE_ANGLE_RAD)), np.float32(0.0)).astype(np.float32)


@registry.env("vbot_navigation_section01", "np")
class VBotSection01Env(NpEnv):
    """Section01 越障导航任务环境。

    我们为 Section01 设计的越障导航环境，核心包括：机身坐标系下的平滑速度指令、
    足端接触观测、腾空时间塑形、前方地形采样观测，以及基座触地终止。机器人沿一组
    路径点逐段前进，最终到达 2026 平台。
    """

    _cfg: VBotSection01EnvCfg

    # G3 恢复历史 checkpoint 使用的观测语义：基础 48 + 足端接触力 12
    # + 地形采样 8 = 68。步态事件只用于奖励与诊断，不再污染策略输入。
    OBS_DIM = 68
    ACTION_SCALE = 0.45
    KP = 80.0
    KD = 6.0
    TRACKING_SIGMA = 0.2
    FEET_AIR_TIME_MIN = 0.12
    FEET_AIR_TIME_TARGET = 0.30
    FOOT_LIFT_ON_HEIGHT = 0.055
    FOOT_LIFT_OFF_HEIGHT = 0.035
    FOOT_LIFT_MIN_VELOCITY = 0.05
    FOOT_SWING_TARGET_HEIGHT = 0.12
    LEG_STALE_GRACE = 0.45
    LEG_STALE_FULL = 1.00
    REACH_THRESHOLD = 0.35
    WAYPOINT_WAIT_STEPS = 20
    COMMAND_SMOOTH_TAU = 0.25
    ROUGH_Y_RANGE = (TERRAIN_START_PLATFORM_Y_END, TERRAIN_ROUGH_Y_END)
    TERRAIN_SCAN_OFFSETS = np.asarray([0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60], dtype=np.float32)
    GOAL_SUCCESS_WINDOW_SIZE = 200

    # ⚠ 改这里**没有任何效果**。_init_buffers 的最后一步是
    #       self._reward_scales = dict(self.REWARD_SCALES)
    #       self._reward_scales.update(cfg.reward_config.scales)
    #   cfg 里配了同名 key 的项会整个覆盖掉下面的值，而 VBotSection01EnvCfg
    #   把这些项全配齐了。所以调权重请改 cfg.py 的 RewardConfig.scales。
    #   下面这份只在 cfg 缺项时兜底。
    REWARD_SCALES = {
        "tracking_lin_vel": 1.2,
        "tracking_ang_vel": 0.6,
        "tracking_goal_vel": 2.0,
        "tracking_yaw": 0.5,
        "forward_progress": 1.0,
        "target_progress": 1.0,
        "reach_goal": 8.0,
        "reach_all_goal": 300.0,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.5,
        "torques": -1e-5,
        "dof_vel": -1e-4,
        "dof_acc": -2.5e-7,
        "action_rate": -0.01,
        "feet_air_time": 0.8,
        "anti_stall": -0.8,
        "dof_pos_limits": -0.5,
        "undesired_contacts": -1.0,
        "base_contact": -10.0,
        "termination": -10.0,
        # ===== 我们自定义的奖励项(权重写在这里,确保不依赖cfg也生效) =====
        "per_leg_swing": 3.0,      # 逼每条腿都迈步,治"后腿不动迈不过去"(核心,权重给大)
        "leg_stale_penalty": -0.5,
        "rear_leg_stale_penalty": -1.0,
        "gait_symmetry": 0.2,      # 鼓励对角腿同步的对角步态(权重取0.2,避免后腿仅搭地不迈步)
        "rear_airborne_balance": -0.8,
        "rear_clearance_balance": -0.4,
        "swing_foot_height": 1.5,  # 崎岖区抬腿高度,跨过0.28m起伏
        "drop_leg_catchup": 4.0,   # 落差点(Y=1.5)额外逼后腿跟上,治"前腿出去后腿没出"
        "drop_pitch": -2.0,        # 落差段压身体前倾
        # ===== 上坡段处理(真实坡道 Y=1.5~7.0) —— 治"上坡前面栽下去" =====
        "slope_leg_drive": 3.0,    # 上坡段逼每条腿(尤其后腿)持续蹬地迈步,提供爬坡动力
        "slope_pitch": -1.5,       # 上坡身体应顺坡微前倾,惩罚相对坡面的过度俯仰,防前栽
        "slope_hip": -1.0,         # 上坡髋关节张开惩罚,治前腿髋张开软倒
        "slope_front_drive": 3.0,  # 前腿上坡驱动,逼前腿积极迈步出力(跟后腿对称)
    }

    def __init__(self, cfg: VBotSection01EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._body = self._model.get_body(cfg.asset.body_name)
        self._num_action = self._model.num_actuators
        self._num_dof_vel = self._model.num_dof_vel
        self._num_dof_pos = self._model.num_dof_pos
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_action,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32)
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._action_scale = float(getattr(cfg.control_config, "action_scale", self.ACTION_SCALE))
        self._kp = float(getattr(cfg.control_config, "stiffness", self.KP))
        self._kd = float(getattr(cfg.control_config, "damping", self.KD))
        self._init_buffers()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _init_buffers(self):
        cfg = self._cfg
        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32,
        )

        self.default_angles = np.zeros((self._num_action,), dtype=np.float32)
        self.hip_indices = []
        self.hip_outward_signs = []
        self.calf_indices = []
        self.joint_lower_limits = np.full((self._num_action,), -1e6, dtype=np.float32)
        self.joint_upper_limits = np.full((self._num_action,), 1e6, dtype=np.float32)
        for idx, actuator_name in enumerate(self._model.actuator_names):
            if actuator_name is None:
                continue
            for joint_name, default_angle in cfg.init_state.default_joint_angles.items():
                if joint_name in actuator_name:
                    self.default_angles[idx] = default_angle
                    break
            if "hip" in actuator_name:
                self.hip_indices.append(idx)
                self.hip_outward_signs.append(
                    -1.0 if actuator_name.startswith(("FR_", "RR_")) else 1.0
                )
            if "calf" in actuator_name:
                self.calf_indices.append(idx)
            if "hip_joint" in actuator_name:
                low, high = -0.73304, 0.73304
            elif "thigh_joint" in actuator_name:
                if actuator_name.startswith("FR_") or actuator_name.startswith("FL_"):
                    low, high = -1.559, 3.1298
                else:
                    low, high = -0.51181, 4.177
            elif "calf_joint" in actuator_name:
                low, high = -2.6387, -0.7854
            else:
                low, high = -1e6, 1e6
            self.joint_lower_limits[idx] = low
            self.joint_upper_limits[idx] = high

        limit_center = 0.5 * (self.joint_lower_limits + self.joint_upper_limits)
        half_range = 0.5 * (self.joint_upper_limits - self.joint_lower_limits)
        self.soft_joint_lower_limits = limit_center - 0.9 * half_range
        self.soft_joint_upper_limits = limit_center + 0.9 * half_range

        self._spawn_center = np.asarray(cfg.init_state.pos, dtype=np.float32)
        random_range = np.asarray(getattr(cfg.init_state, "pos_randomization_range", [-0.5, -0.5, 0.5, 0.5]), dtype=np.float32)
        self._spawn_low = random_range[:2]
        self._spawn_high = random_range[2:4]
        self._joint_noise_scale = 0.03
        self._command_smooth_alpha = float(np.clip(cfg.ctrl_dt / (self.COMMAND_SMOOTH_TAU + cfg.ctrl_dt), 0.0, 1.0))
        self._tracking_sigma = max(float(getattr(cfg.reward_config, "tracking_sigma", self.TRACKING_SIGMA)), 1e-4)
        self._feet_air_time_min = max(
            float(getattr(cfg.reward_config, "feet_air_time_min", self.FEET_AIR_TIME_MIN)),
            0.0,
        )
        self._feet_air_time_target = max(
            float(getattr(cfg.reward_config, "feet_air_time_target", self.FEET_AIR_TIME_TARGET)),
            self._feet_air_time_min + 1e-6,
        )
        self._foot_lift_on_height = max(
            float(getattr(cfg.reward_config, "foot_lift_on_height", self.FOOT_LIFT_ON_HEIGHT)),
            1e-4,
        )
        self._foot_lift_off_height = float(
            np.clip(
                getattr(cfg.reward_config, "foot_lift_off_height", self.FOOT_LIFT_OFF_HEIGHT),
                0.0,
                self._foot_lift_on_height - 1e-4,
            )
        )
        self._foot_lift_min_velocity = max(
            float(getattr(cfg.reward_config, "foot_lift_min_velocity", self.FOOT_LIFT_MIN_VELOCITY)),
            0.0,
        )
        self._foot_swing_target_height = max(
            float(getattr(cfg.reward_config, "foot_swing_target_height", self.FOOT_SWING_TARGET_HEIGHT)),
            self._foot_lift_on_height,
        )
        self._reach_threshold = max(float(getattr(cfg.commands, "waypoint_reach_threshold", self.REACH_THRESHOLD)), 0.05)
        wait_seconds = max(float(getattr(cfg.commands, "waypoint_wait_seconds", 0.5)), 0.0)
        self._waypoint_wait_steps = 0 if wait_seconds <= 0.0 else max(1, int(round(wait_seconds / max(cfg.ctrl_dt, 1e-6))))
        self._waypoint_lin_kp = float(getattr(cfg.commands, "waypoint_lin_kp", 0.8))
        self._waypoint_ang_kp = float(getattr(cfg.commands, "waypoint_ang_kp", 1.0))
        smooth_tau = max(float(getattr(cfg.commands, "waypoint_cmd_smooth_tau", self.COMMAND_SMOOTH_TAU)), 1e-6)
        self._command_smooth_alpha = float(np.clip(cfg.ctrl_dt / (smooth_tau + cfg.ctrl_dt), 0.0, 1.0))
        self._vel_low = np.asarray(getattr(cfg.commands, "vel_limit", ((-0.35, -0.16, -0.45), (0.55, 0.16, 0.45)))[0], dtype=np.float32)
        self._vel_high = np.asarray(getattr(cfg.commands, "vel_limit", ((-0.35, -0.16, -0.45), (0.55, 0.16, 0.45)))[1], dtype=np.float32)
        self._rough_vel_low = np.asarray(getattr(cfg.commands, "rough_vel_limit", ((-0.16, -0.10, -0.35), (0.42, 0.10, 0.35)))[0], dtype=np.float32)
        self._rough_vel_high = np.asarray(getattr(cfg.commands, "rough_vel_limit", ((-0.16, -0.10, -0.35), (0.42, 0.10, 0.35)))[1], dtype=np.float32)
        waypoint_targets = np.asarray(getattr(cfg.commands, "waypoint_targets", ((-2.7, 8.0),)), dtype=np.float32).reshape(-1, 2)
        if waypoint_targets.shape[0] == 0:
            waypoint_targets = np.asarray([(-2.7, 8.0)], dtype=np.float32)
        self._waypoint_targets = waypoint_targets
        self._reward_scales = dict(self.REWARD_SCALES)
        self._reward_scales.update(dict(getattr(cfg.reward_config, "scales", {})))
        hfield = self._model.get_hfield("C1_hfield_terrain")
        self._terrain_hfield_matrix = np.asarray(hfield.height_matrix, dtype=np.float32)
        self._terrain_hfield_bound = np.asarray(hfield.bound, dtype=np.float32)
        self._foot_contact_sensor_groups = [list(group) for group in getattr(cfg.sensor, "foot_contact_sensor_groups", [
            ["FR_foot_contact_1", "FR_foot_contact_2", "FR_foot_contact_3"],
            ["FL_foot_contact_1", "FL_foot_contact_2", "FL_foot_contact_3"],
            ["RR_foot_contact_1", "RR_foot_contact_2", "RR_foot_contact_3"],
            ["RL_foot_contact_1", "RL_foot_contact_2", "RL_foot_contact_3"],
        ])]
        self._foot_force_sensor_groups = [list(group) for group in self._foot_contact_sensor_groups]
        self._base_contact_sensors = list(getattr(cfg.sensor, "base_contact_sensors", ["base_contact_1", "base_contact_2", "base_contact_3"]))
        self._foot_geoms = []
        for foot in getattr(cfg.sensor, "feet", ["FR", "FL", "RR", "RL"]):
            try:
                self._foot_geoms.append(self._model.get_geom(foot))
            except BaseException:
                self._foot_geoms.append(None)
        self._missing_contact_sensors = set()
        self._init_termination_contact()
        self._init_undesired_contact()

        self._goal_success_episodes = 0
        self._goal_done_episodes = 0
        self._goal_success_window = np.zeros((self.GOAL_SUCCESS_WINDOW_SIZE,), dtype=np.float32)
        self._goal_success_window_head = 0
        self._goal_success_window_count = 0
        self._goal_success_window_sum = 0.0

    @staticmethod
    def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
        half = 0.5 * yaw
        quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
        quat[:, 2] = np.sin(half)
        quat[:, 3] = np.cos(half)
        return quat

    @staticmethod
    def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
        x = quat_xyzw[:, 0]
        y = quat_xyzw[:, 1]
        z = quat_xyzw[:, 2]
        w = quat_xyzw[:, 3]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    def _init_termination_contact(self):
        base_geom_names = list(getattr(self._cfg.asset, "terminate_after_contacts_on", []))
        ground_prefixes = ["C1_", "C2_", "C3_"]
        ground_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        ground_geoms = list(dict.fromkeys(ground_geoms))

        ground_names = [self._model.geom_names[index] for index in ground_geoms]
        if ground_names and all("dixing_Plane" in name for name in ground_names):
            ground_geoms = []

        pairs = []
        for base_geom_name in base_geom_names:
            try:
                base_idx = self._model.get_geom_index(base_geom_name)
            except BaseException:
                continue
            for ground_idx in ground_geoms:
                pairs.append([base_idx, ground_idx])
        self.termination_contact = np.asarray(pairs, dtype=np.uint32) if pairs else np.zeros((0, 2), dtype=np.uint32)
        self.num_termination_check = int(self.termination_contact.shape[0])
        if self.num_termination_check == 0:
            print("[Info] 终止接触 geom 不足以覆盖全部地面，基座接触检测使用 contact sensors。")

    def _init_undesired_contact(self):
        geom_names = list(getattr(self._cfg.asset, "undesired_contacts_on", []))
        ground_prefixes = ["C1_", "C2_", "C3_"]
        ground_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        ground_geoms = list(dict.fromkeys(ground_geoms))

        pairs = []
        for geom_name in geom_names:
            try:
                geom_idx = self._model.get_geom_index(geom_name)
            except BaseException:
                continue
            for ground_idx in ground_geoms:
                pairs.append([geom_idx, ground_idx])
        self.undesired_contact = np.asarray(pairs, dtype=np.uint32) if pairs else np.zeros((0, 2), dtype=np.uint32)
        self.num_undesired_contact_check = int(self.undesired_contact.shape[0])

    def _safe_get_sensor_value(self, sensor_name: str, data: mtx.SceneData):
        if sensor_name in self._missing_contact_sensors:
            return None
        try:
            return self._model.get_sensor_value(sensor_name, data)
        except BaseException:
            self._missing_contact_sensors.add(sensor_name)
            return None

    @staticmethod
    def _sensor_value_to_contact(sensor_value, num_envs: int, threshold: float = 0.01) -> np.ndarray:
        if sensor_value is None:
            return np.zeros(num_envs, dtype=bool)
        value = np.asarray(sensor_value, dtype=np.float32)
        if value.ndim == 0:
            return np.full(num_envs, np.abs(float(value)) > threshold, dtype=bool)
        if value.shape[0] != num_envs:
            flat = value.reshape(-1)
            hit = np.max(np.abs(flat)) > threshold if flat.size > 0 else False
            return np.full(num_envs, hit, dtype=bool)
        if value.ndim == 1:
            return np.abs(value) > threshold
        return np.linalg.norm(value, axis=1) > threshold

    def _aggregate_contact_group(self, sensor_names: list[str], data: mtx.SceneData, threshold: float = 0.01):
        contact = np.zeros((data.shape[0],), dtype=bool)
        for sensor_name in sensor_names:
            sensor_value = self._safe_get_sensor_value(sensor_name, data)
            contact = np.logical_or(contact, self._sensor_value_to_contact(sensor_value, data.shape[0], threshold))
        return contact

    def _get_foot_contacts(self, data: mtx.SceneData) -> np.ndarray:
        return np.stack([self._aggregate_contact_group(group, data) for group in self._foot_contact_sensor_groups], axis=1)

    def _get_foot_positions_world(self, data: mtx.SceneData) -> np.ndarray:
        foot_pos = []
        for geom in self._foot_geoms:
            if geom is None:
                foot_pos.append(np.zeros((data.shape[0], 3), dtype=np.float32))
            else:
                foot_pos.append(geom.get_pose(data)[:, :3].astype(np.float32))
        return np.stack(foot_pos, axis=1)

    def _get_base_contacts(self, data: mtx.SceneData) -> np.ndarray:
        if self.num_termination_check > 0:
            cquerys = self._model.get_contact_query(data)
            contact = cquerys.is_colliding(self.termination_contact)
            contact = contact.reshape((data.shape[0], self.num_termination_check))
            return contact.any(axis=1)
        return self._aggregate_contact_group(self._base_contact_sensors, data)

    def _get_undesired_contacts(self, data: mtx.SceneData) -> np.ndarray:
        if self.num_undesired_contact_check <= 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        cquerys = self._model.get_contact_query(data)
        contact = cquerys.is_colliding(self.undesired_contact)
        contact = contact.reshape((data.shape[0], self.num_undesired_contact_check))
        return contact.any(axis=1).astype(np.float32)

    def _sensor_value_to_force_vec(self, sensor_value, num_envs: int) -> np.ndarray:
        if sensor_value is None:
            return np.zeros((num_envs, 3), dtype=np.float32)
        value = np.asarray(sensor_value, dtype=np.float32)
        if value.ndim == 0:
            return np.tile(np.array([[0.0, 0.0, float(value)]], dtype=np.float32), (num_envs, 1))
        if value.ndim == 1:
            if value.shape[0] == num_envs:
                return np.stack([np.zeros_like(value), np.zeros_like(value), value], axis=1).astype(np.float32)
            if num_envs == 1 and value.shape[0] >= 3:
                return value[:3].reshape(1, 3).astype(np.float32)
            scalar = float(value.reshape(-1)[0]) if value.size > 0 else 0.0
            return np.tile(np.array([[0.0, 0.0, scalar]], dtype=np.float32), (num_envs, 1))
        if value.shape[0] != num_envs:
            flat = value.reshape(-1)
            sample = np.zeros(3, dtype=np.float32)
            if flat.size >= 3:
                sample = flat[:3].astype(np.float32)
            elif flat.size > 0:
                sample[2] = float(flat[0])
            return np.tile(sample.reshape(1, 3), (num_envs, 1))
        if value.shape[1] >= 3:
            return value[:, :3].astype(np.float32)
        if value.shape[1] == 1:
            return np.concatenate([np.zeros((num_envs, 2), dtype=np.float32), value], axis=1).astype(np.float32)
        return np.zeros((num_envs, 3), dtype=np.float32)

    def _get_foot_contact_force(self, data: mtx.SceneData, root_quat: np.ndarray) -> np.ndarray:
        force_list = []
        for sensor_group in self._foot_force_sensor_groups:
            group_force = np.zeros((data.shape[0], 3), dtype=np.float32)
            group_norm = np.zeros((data.shape[0],), dtype=np.float32)
            for sensor_name in sensor_group:
                force_vec = self._sensor_value_to_force_vec(self._safe_get_sensor_value(sensor_name, data), data.shape[0])
                force_norm = np.linalg.norm(force_vec, axis=1)
                take = force_norm > group_norm
                if np.any(take):
                    group_force[take] = force_vec[take]
                    group_norm[take] = force_norm[take]
            force_list.append(group_force)
        contact_force_world = np.concatenate(force_list[:4], axis=1).astype(np.float32)
        reshaped = contact_force_world.reshape(data.shape[0], 4, 3)
        rotated = [Quaternion.rotate_inverse(root_quat, reshaped[:, i, :]) for i in range(4)]
        return np.concatenate(rotated, axis=1).astype(np.float32)

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)

    def _compute_torques(self, actions: np.ndarray, data: mtx.SceneData) -> np.ndarray:
        target_pos = self.default_angles + actions * self._action_scale
        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)
        torques = self._kp * (target_pos - current_pos) - self._kd * current_vel
        torque_limits = np.asarray([17.0, 17.0, 34.0] * 4, dtype=np.float32)
        return np.clip(torques, -torque_limits, torque_limits)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = np.clip(actions, -1.0, 1.0).astype(np.float32)
        state.data.actuator_ctrls = self._compute_torques(state.info["current_actions"], state.data)
        return state

    def _rough_zone_mask(self, base_xy: np.ndarray) -> np.ndarray:
        return np.logical_and(base_xy[:, 1] >= self.ROUGH_Y_RANGE[0], base_xy[:, 1] <= self.ROUGH_Y_RANGE[1])

    def _terrain_height_at_xy(self, sample_xy: np.ndarray) -> np.ndarray:
        """返回任意形状 XY 采样点的真实赛道高度，含崎岖 hfield 插值。"""
        sample_xy = np.asarray(sample_xy, dtype=np.float32)
        sample_x = sample_xy[..., 0]
        sample_y = sample_xy[..., 1]
        h = np.zeros_like(sample_y, dtype=np.float32)

        x_min, y_min, _, x_max, y_max, _ = self._terrain_hfield_bound
        rough = np.logical_and(sample_y >= y_min, sample_y <= y_max)
        if np.any(rough):
            x_coord = np.clip((sample_x - x_min) / (x_max - x_min), 0.0, 1.0)
            y_coord = np.clip((sample_y - y_min) / (y_max - y_min), 0.0, 1.0)
            x_pos = x_coord * (self._terrain_hfield_matrix.shape[1] - 1)
            y_pos = y_coord * (self._terrain_hfield_matrix.shape[0] - 1)
            x0 = np.floor(x_pos).astype(np.int32)
            y0 = np.floor(y_pos).astype(np.int32)
            x1 = np.minimum(x0 + 1, self._terrain_hfield_matrix.shape[1] - 1)
            y1 = np.minimum(y0 + 1, self._terrain_hfield_matrix.shape[0] - 1)
            tx = x_pos - x0
            ty = y_pos - y0
            h00 = self._terrain_hfield_matrix[y0, x0]
            h10 = self._terrain_hfield_matrix[y0, x1]
            h01 = self._terrain_hfield_matrix[y1, x0]
            h11 = self._terrain_hfield_matrix[y1, x1]
            hfield_height = (
                h00 * (1.0 - tx) * (1.0 - ty)
                + h10 * tx * (1.0 - ty)
                + h01 * (1.0 - tx) * ty
                + h11 * tx * ty
            )
            h = np.where(rough, hfield_height, h)

        h = np.where(
            np.logical_and(sample_y >= TERRAIN_SLOPE_Y_START, sample_y < TERRAIN_SLOPE_Y_END),
            section01_terrain_height(sample_y),
            h,
        )
        h = np.where(sample_y >= TERRAIN_SLOPE_Y_END, np.float32(TERRAIN_PLATFORM_HEIGHT), h)
        return h.astype(np.float32)

    def _get_foot_height_above_terrain(self, data: mtx.SceneData) -> np.ndarray:
        foot_pos = self._get_foot_positions_world(data)
        terrain_height = self._terrain_height_at_xy(foot_pos[:, :, :2])
        return (foot_pos[:, :, 2] - terrain_height).astype(np.float32)

    def _kinematic_gait_events(
        self,
        foot_height: np.ndarray,
        previous_height: np.ndarray,
        previous_airborne: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """由足端相对地面的高度和速度生成带滞回的离地/落地事件。"""
        foot_height = np.asarray(foot_height, dtype=np.float32)
        previous_height = np.asarray(previous_height, dtype=np.float32)
        previous_airborne = np.asarray(previous_airborne, dtype=np.bool_)
        foot_vertical_velocity = (foot_height - previous_height) / max(self._cfg.ctrl_dt, 1e-6)
        airborne = np.where(
            previous_airborne,
            foot_height > self._foot_lift_off_height,
            foot_height >= self._foot_lift_on_height,
        )
        first_liftoff = np.logical_and.reduce(
            (
                ~previous_airborne,
                airborne,
                foot_vertical_velocity >= self._foot_lift_min_velocity,
            )
        )
        first_touchdown = np.logical_and(previous_airborne, ~airborne)
        return (
            airborne.astype(np.bool_),
            first_liftoff.astype(np.bool_),
            first_touchdown.astype(np.bool_),
            foot_vertical_velocity.astype(np.float32),
        )

    def _per_leg_swing_score(self, info: dict, active_move: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按四腿中最弱的一腿评分，并让过期步态记录逐渐失效。"""
        num_envs = active_move.shape[0]
        current_air_time = info.get("feet_air_time")
        if current_air_time is None or current_air_time.shape != (num_envs, 4):
            current_air_time = np.zeros((num_envs, 4), dtype=np.float32)
        last_air_time = info.get("_leg_last_air_time")
        if last_air_time is None or last_air_time.shape != (num_envs, 4):
            last_air_time = np.zeros((num_envs, 4), dtype=np.float32)
        stale_time = info.get("_leg_stale_time")
        if stale_time is None or stale_time.shape != (num_envs, 4):
            stale_time = np.zeros((num_envs, 4), dtype=np.float32)

        effective_air_time = np.maximum(current_air_time, last_air_time)
        duration_score = np.clip(
            effective_air_time / self._feet_air_time_target,
            0.0,
            1.0,
        )
        stale_span = max(self.LEG_STALE_FULL - self.LEG_STALE_GRACE, 1e-6)
        freshness = 1.0 - np.clip(
            (stale_time - self.LEG_STALE_GRACE) / stale_span,
            0.0,
            1.0,
        )
        score_by_leg = duration_score * freshness
        score = np.min(score_by_leg, axis=1) * active_move
        return score.astype(np.float32), score_by_leg.astype(np.float32)

    def _slope_hip_opening_penalty(self, dof_pos: np.ndarray, base_y: np.ndarray) -> np.ndarray:
        """只惩罚真实坡段内超过 0.2 rad 的髋关节向外张开。"""
        if len(self.hip_indices) == 0:
            return np.zeros((dof_pos.shape[0],), dtype=np.float32)
        hip_idx = np.asarray(self.hip_indices, dtype=np.int64)
        outward_sign = np.asarray(self.hip_outward_signs, dtype=np.float32)
        hip_deviation = dof_pos[:, hip_idx] - self.default_angles[hip_idx]
        outward_opening = hip_deviation * outward_sign
        opening_excess = np.clip(outward_opening - 0.2, 0.0, None)
        on_slope = np.logical_and(
            base_y >= TERRAIN_SLOPE_Y_START,
            base_y < TERRAIN_SLOPE_Y_END,
        )
        return (np.sum(np.square(opening_excess), axis=1) * on_slope).astype(np.float32)

    def _rear_support_progress(
        self,
        forward_speed: np.ndarray,
        contacts: np.ndarray,
        active_move: np.ndarray,
    ) -> np.ndarray:
        """用后腿支撑时的实际前进速度代替 stale 作为坡上驱动信号。"""
        rear_contact_ratio = np.mean(contacts[:, 2:4].astype(np.float32), axis=1)
        reference_speed = max(float(self._vel_high[0]), 1e-3)
        speed_ratio = np.clip(forward_speed / reference_speed, 0.0, 1.0)
        return (speed_ratio * rear_contact_ratio * active_move).astype(np.float32)

    def _get_terrain_scan(self, base_xy: np.ndarray) -> np.ndarray:
        """前方地形高度采样（真实 hfield 双线性插值与解析赛道剖面）。"""
        sample_y = base_xy[:, 1:2] + self.TERRAIN_SCAN_OFFSETS.reshape(1, -1)
        sample_x = np.broadcast_to(base_xy[:, 0:1], sample_y.shape)
        sample_xy = np.stack([sample_x, sample_y], axis=-1)
        h = self._terrain_height_at_xy(sample_xy)
        return (h / np.float32(TERRAIN_PLATFORM_HEIGHT)).astype(np.float32)

    def _command_limits(self, base_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.tile(self._vel_low, (base_xy.shape[0], 1))
        high = np.tile(self._vel_high, (base_xy.shape[0], 1))
        y = base_xy[:, 1]

        # ── 区域1：崎岖区本身（Y -1.65~1.85）──────────────────────────────────
        # 已验证的行为，不动。
        rough = self._rough_zone_mask(base_xy)
        if np.any(rough):
            low[rough] = self._rough_vel_low
            high[rough] = self._rough_vel_high

        # ── 区域2：崎岖出口落差段（Y 1.85~2.5）────────────────────────────────
        # 从地形碰撞文件精确算出：崎岖区(hfield)出口在Y=1.5,该处地面最高0.277m,
        # 一过Y=1.5立刻是Z=0平地 —— Y=1.5是最高0.28m的下落坎,最易摔。
        # 在落差线前后(1.3~1.8)前进上限减半,给狗时间让前后腿一起过、别让前腿冲下去。
        drop_zone = np.logical_and(y > 1.3, y <= 1.8)
        if np.any(drop_zone):
            low[drop_zone] = self._rough_vel_low
            high[drop_zone] = self._rough_vel_high
            high[drop_zone, 0] = self._rough_vel_high[0] * 0.5  # 前进上限减半

        # ── 区域3：上坡衔接+坡道（Y 1.8~6.8）──────────────────────────────────
        # 关键修复:落差降速到1.8结束、坡道原从2.5才开始,Y1.8~2.5是速度空档,
        # 而平地→15°坡交界正好在Y=2.0 —— 狗在此全速冲上坡折角,惯性带头往前栽。
        # 现从1.8起接管(衔接落差不留空档),前进上限7折,慢着稳着踏上折角再爬。
        slope_zone = np.logical_and(y > 1.8, y <= 6.8)
        if np.any(slope_zone):
            low[slope_zone] = self._vel_low
            high[slope_zone] = self._vel_high
            high[slope_zone, 0] = self._vel_high[0] * 0.7  # 前进上限7折,别冲折角

        return low, high

    def _update_waypoint_commands(self, data: mtx.SceneData, info: dict):
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_xy = pose[:, :2]
        num_envs = data.shape[0]

        goal_idx = info["goal_idx"]
        wait_steps = info["waypoint_wait_steps_left"]
        commands_smoothed = info["commands_smoothed"]
        goal_done = info["goal_done"]

        targets = info["waypoints"][np.arange(num_envs), goal_idx]
        delta_world_xy = targets - base_xy
        dist = np.linalg.norm(delta_world_xy, axis=1)
        reached = np.logical_and(~goal_done, np.logical_and(wait_steps <= 0, dist <= self._reach_threshold))
        if np.any(reached):
            info["goal_reached_this_step"][reached] = True
            if self._waypoint_wait_steps <= 0:
                last_idx = info["num_waypoints"] - 1
                finished = np.logical_and(reached, goal_idx >= last_idx)
                advance = np.logical_and(reached, ~finished)
                goal_idx[advance] += 1
                goal_done[finished] = True
                targets = info["waypoints"][np.arange(num_envs), goal_idx]
                delta_world_xy = targets - base_xy
                dist = np.linalg.norm(delta_world_xy, axis=1)
            else:
                wait_steps[reached] = self._waypoint_wait_steps

        waiting = wait_steps > 0
        command_raw = np.zeros((num_envs, 3), dtype=np.float32)
        active = np.logical_and(~waiting, ~goal_done)
        if np.any(active):
            delta_world = np.zeros((num_envs, 3), dtype=np.float32)
            delta_world[:, :2] = delta_world_xy
            delta_body = Quaternion.rotate_inverse(root_quat, delta_world)[:, :2]
            current_yaw = self._quat_to_yaw(root_quat)
            desired_yaw = np.arctan2(delta_world_xy[:, 1], delta_world_xy[:, 0]).astype(np.float32)
            yaw_error = self._wrap_to_pi(desired_yaw - current_yaw)
            command_raw[:, 0] = self._waypoint_lin_kp * delta_body[:, 0]
            command_raw[:, 1] = self._waypoint_lin_kp * delta_body[:, 1]
            command_raw[:, 2] = self._waypoint_ang_kp * yaw_error
            low, high = self._command_limits(base_xy)
            command_raw = np.clip(command_raw, low, high).astype(np.float32)
            command_raw[~active] = 0.0

        commands_smoothed += self._command_smooth_alpha * (command_raw - commands_smoothed)
        low, high = self._command_limits(base_xy)
        commands_smoothed = np.clip(commands_smoothed, low, high).astype(np.float32)
        commands_smoothed[goal_done] = 0.0

        wait_before = wait_steps.copy()
        if self._waypoint_wait_steps > 0 and np.any(waiting):
            wait_steps[waiting] = np.maximum(wait_steps[waiting] - 1, 0)
        switch = np.logical_and(wait_before > 0, wait_steps == 0)
        if np.any(switch):
            last_idx = info["num_waypoints"] - 1
            finished = np.logical_and(switch, goal_idx >= last_idx)
            advance = np.logical_and(switch, ~finished)
            goal_idx[advance] += 1
            commands_smoothed[switch] = 0.0
            goal_done[finished] = True

        info["goal_idx"] = goal_idx
        info["waypoint_wait_steps_left"] = wait_steps
        info["commands_smoothed"] = commands_smoothed
        info["commands"] = commands_smoothed.copy()
        info["goal_done"] = goal_done
        info["target_xy"] = targets.astype(np.float32)
        info["distance_to_waypoint"] = dist.astype(np.float32)

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_lin_vel_world = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        local_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel_world)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        local_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        joint_pos_rel = dof_pos - self.default_angles

        obs_base = np.hstack(
            [
                local_lin_vel * self._cfg.normalization.lin_vel,
                gyro * self._cfg.normalization.ang_vel,
                local_gravity,
                joint_pos_rel * self._cfg.normalization.dof_pos,
                dof_vel * self._cfg.normalization.dof_vel,
                info["current_actions"],
                info["commands"] * self.commands_scale,
            ]
        )
        assert obs_base.shape == (data.shape[0], 48)
        foot_contact_force = self._get_foot_contact_force(data, root_quat)
        terrain_scan = self._get_terrain_scan(pose[:, :2])
        obs = np.concatenate([obs_base, foot_contact_force, terrain_scan], axis=1)
        assert obs.shape == (data.shape[0], self.OBS_DIM)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _compute_terminated(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        pose = self._body.get_pose(data)
        base_lin_vel_world = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        base_contact = self._get_base_contacts(data)
        speed_overflow = np.sum(np.square(base_lin_vel_world[:, :2]), axis=1) > 1e8
        invalid = np.isnan(pose).any(axis=1) | np.isnan(self.get_dof_vel(data)).any(axis=1)
        # 我们的终止策略：基座触地即终止；超时由 NpEnv 的 truncation 处理。
        # 数值异常(NaN/速度溢出)作为兜底保护一并终止。
        terminated = base_contact.copy()
        terminated = np.logical_or(terminated, speed_overflow)
        terminated = np.logical_or(terminated, invalid)
        terminated = np.logical_or(terminated, info["goal_done"])
        info["base_contact"] = base_contact
        info["fall_like_termination"] = base_contact.copy()
        return terminated

    def _update_goal_success_metrics(self, info: dict, terminated: np.ndarray):
        done_now = terminated.astype(bool)
        if self._cfg.max_episode_steps:
            done_now = np.logical_or(done_now, (info["steps"] + 1) >= self._cfg.max_episode_steps)
        if np.any(done_now):
            success = np.logical_and(done_now, info["goal_done"])
            self._goal_done_episodes += int(np.sum(done_now))
            self._goal_success_episodes += int(np.sum(success))
            for value in success[done_now].astype(np.float32):
                if self._goal_success_window_count >= self.GOAL_SUCCESS_WINDOW_SIZE:
                    self._goal_success_window_sum -= float(self._goal_success_window[self._goal_success_window_head])
                else:
                    self._goal_success_window_count += 1
                self._goal_success_window[self._goal_success_window_head] = value
                self._goal_success_window_sum += float(value)
                self._goal_success_window_head = (self._goal_success_window_head + 1) % self.GOAL_SUCCESS_WINDOW_SIZE

        success_rate = (
            float(self._goal_success_episodes) / float(self._goal_done_episodes)
            if self._goal_done_episodes > 0
            else 0.0
        )
        success_window = (
            float(self._goal_success_window_sum) / float(self._goal_success_window_count)
            if self._goal_success_window_count > 0
            else 0.0
        )
        metrics = info.get("metrics", {})
        metrics["goal_success_rate"] = np.full((terminated.shape[0],), success_rate, dtype=np.float32)
        metrics["goal_success_rate_window200"] = np.full((terminated.shape[0],), success_window, dtype=np.float32)
        info["metrics"] = metrics

    def _diagnostic_metrics(
        self,
        data: mtx.SceneData,
        info: dict,
        pose: np.ndarray,
        projected_gravity: np.ndarray,
        dof_acc: np.ndarray,
    ) -> dict:
        """纯观测用的诊断指标 —— 不参与奖励，只为了让失败在 TensorBoard 上可见。

        motrix_rl/skrl/{torch,jax}/train/ppo.py 的 record_transition 会把
        info["metrics"] 里的每个 (num_envs,) 数组自动写成三条 TB 曲线
        （max / min / mean），所以这里只管往字典里放，RL 侧不需要任何改动。

        每一项都是为了回答一个现有指标答不了的问题，见 docs/section01/01-baseline-audit.md。
        """
        base_y = pose[:, 1]
        terrain_z = section01_terrain_height(base_y)

        # 传感器口径的基座触地真值。不改终止逻辑，只把它记录下来 ——
        # 它和 metrics["undesired_contacts"] / Reward 里 base_contact 的差距，
        # 就是 BUG-001 被画成的曲线。
        belly_contact = self._aggregate_contact_group(self._base_contact_sensors, data).astype(np.float32)

        # 每条腿"距上次离地事件有多久"。G3 的事件来自足端运动学，
        # 不再依赖训练中长期保持为真的接触传感器。
        stale = info.get("_leg_stale_time")
        if stale is None or stale.shape != (data.shape[0], 4):
            stale = np.zeros((data.shape[0], 4), dtype=np.float32)
        contacts = info.get("contacts", np.zeros((data.shape[0], 4), dtype=np.bool_))
        sensor_contacts = info.get("sensor_contacts", np.zeros((data.shape[0], 4), dtype=np.bool_))
        airborne = info.get("_kinematic_airborne", np.zeros((data.shape[0], 4), dtype=np.bool_))
        liftoff = info.get("first_liftoff", np.zeros((data.shape[0], 4), dtype=np.bool_))
        foot_height = info.get(
            "_foot_height_above_terrain",
            np.zeros((data.shape[0], 4), dtype=np.float32),
        )
        foot_vertical_velocity = info.get(
            "_foot_vertical_velocity",
            np.zeros((data.shape[0], 4), dtype=np.float32),
        )

        root_quat = pose[:, 3:7]
        torques = np.clip(
            np.nan_to_num(data.actuator_ctrls, nan=0.0, posinf=0.0, neginf=0.0),
            -200.0,
            200.0,
        ).reshape(data.shape[0], 4, 3)
        joint_vel = np.clip(
            np.nan_to_num(self.get_dof_vel(data), nan=0.0, posinf=0.0, neginf=0.0),
            -500.0,
            500.0,
        ).reshape(data.shape[0], 4, 3)
        joint_power = np.abs(torques * joint_vel)
        foot_force = self._get_foot_contact_force(data, root_quat).reshape(data.shape[0], 4, 3)
        foot_force_norm = np.linalg.norm(foot_force, axis=2)

        metrics = {
            # 现有的 goal_idx 只有 7 个量化档，distance_to_waypoint 每过一门就重置，
            # 两者都画不出"狗到底走了多远"。base_y 才是赛道进度的直接读数。
            "base_y": base_y.astype(np.float32),
            # base_height 是绝对 z，被 1.29m 的坡道高差污染，不能当摔倒检测用。
            # 减掉真实地面高度之后才是"离地多高"。
            "height_above_terrain": (pose[:, 2] - terrain_z).astype(np.float32),
            # BUG-001 的判官
            "belly_contact_any": belly_contact,
            # 贴合坡面 vs 往前栽的区分量（BUG-005）
            "pitch_rel_slope": (projected_gravity[:, 0] - section01_expected_pitch(base_y)).astype(np.float32),
            # 最懒的那条腿已经多久没抬过了（BUG-002 想要的那个量）
            "max_leg_stale_time": np.max(stale, axis=1).astype(np.float32),
            # 未加权的原始 dof_acc 量级 —— 让"正则项量纲失控"看得见（BUG-007）
            "dof_acc_raw_mean": np.mean(np.square(dof_acc.astype(np.float64)), axis=1).astype(np.float32),
            "rear_torque_abs": np.mean(np.abs(torques[:, 2:]), axis=(1, 2)).astype(np.float32),
            "rear_power_abs": np.mean(joint_power[:, 2:], axis=(1, 2)).astype(np.float32),
            "rear_contact_force_norm": np.mean(foot_force_norm[:, 2:], axis=1).astype(np.float32),
        }
        for leg_idx, leg_name in enumerate(("fr", "fl", "rr", "rl")):
            metrics[f"leg_stale_time_{leg_name}"] = stale[:, leg_idx].astype(np.float32)
            metrics[f"leg_contact_{leg_name}"] = contacts[:, leg_idx].astype(np.float32)
            metrics[f"leg_sensor_contact_{leg_name}"] = sensor_contacts[:, leg_idx].astype(np.float32)
            metrics[f"leg_airborne_{leg_name}"] = airborne[:, leg_idx].astype(np.float32)
            metrics[f"leg_liftoff_{leg_name}"] = liftoff[:, leg_idx].astype(np.float32)
            metrics[f"leg_clearance_{leg_name}"] = foot_height[:, leg_idx].astype(np.float32)
            metrics[f"leg_vertical_velocity_{leg_name}"] = foot_vertical_velocity[:, leg_idx].astype(np.float32)
            last_air_time = info.get("_leg_last_air_time", np.zeros_like(stale))
            metrics[f"leg_last_air_time_{leg_name}"] = last_air_time[:, leg_idx].astype(np.float32)
            metrics[f"leg_torque_abs_{leg_name}"] = np.mean(np.abs(torques[:, leg_idx]), axis=1).astype(np.float32)
            metrics[f"leg_power_abs_{leg_name}"] = np.mean(joint_power[:, leg_idx], axis=1).astype(np.float32)
            metrics[f"leg_contact_force_norm_{leg_name}"] = foot_force_norm[:, leg_idx].astype(np.float32)
        return metrics

    def _compute_reward(self, data: mtx.SceneData, info: dict, terminated: np.ndarray) -> np.ndarray:
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_lin_vel_world = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        local_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel_world)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        commands = info["commands"]

        lin_vel_error = np.sum(np.square(commands[:, :2] - local_lin_vel[:, :2]), axis=1)
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        command_speed_xy = np.linalg.norm(commands[:, :2], axis=1)
        body_speed_xy = np.linalg.norm(local_lin_vel[:, :2], axis=1)
        speed_deficit = np.clip(command_speed_xy - body_speed_xy, 0.0, None)
        active_move = command_speed_xy > 0.05
        cmd_dir = commands[:, :2] / np.maximum(command_speed_xy[:, None], 1e-6)
        forward_speed = np.sum(local_lin_vel[:, :2] * cmd_dir, axis=1)
        forward_progress = np.clip(forward_speed, 0.0, 1.5) * active_move

        target_xy = info.get("target_xy", pose[:, :2])
        target_rel_world = np.zeros((data.shape[0], 3), dtype=np.float32)
        target_rel_world[:, :2] = target_xy - pose[:, :2]
        distance_to_waypoint = np.linalg.norm(target_rel_world[:, :2], axis=1)
        prev_distance = info.get("prev_distance_to_waypoint", distance_to_waypoint)
        target_progress = np.clip(prev_distance - distance_to_waypoint, -0.2, 0.2)
        target_rel_body = Quaternion.rotate_inverse(root_quat, target_rel_world)[:, :2]
        target_dir_body = target_rel_body / (np.linalg.norm(target_rel_body, axis=1, keepdims=True) + 1e-6)
        tracking_goal_vel = np.minimum(np.sum(target_dir_body * local_lin_vel[:, :2], axis=1), commands[:, 0])
        tracking_goal_vel = tracking_goal_vel / (np.abs(commands[:, 0]) + 1e-5)
        tracking_goal_vel = np.clip(tracking_goal_vel, -1.0, 1.0) * active_move

        current_yaw = self._quat_to_yaw(root_quat)
        desired_yaw = np.arctan2(target_rel_world[:, 1], target_rel_world[:, 0]).astype(np.float32)
        tracking_yaw = np.exp(-np.abs(self._wrap_to_pi(desired_yaw - current_yaw))) * active_move

        rough = self._rough_zone_mask(pose[:, :2]).astype(np.float32)
        terrain_scan = self._get_terrain_scan(pose[:, :2])
        contacts = info.get("contacts", np.zeros((data.shape[0], 4), dtype=np.bool_))
        foot_pos = self._get_foot_positions_world(data)
        foot_z = foot_pos[:, :, 2]
        foot_height_above_terrain = self._get_foot_height_above_terrain(data)
        foot_clearance = np.clip(
            foot_height_above_terrain - self._foot_lift_off_height,
            0.0,
            None,
        )
        c = contacts.astype(np.float32)  # [n,4] = FR,FL,RR,RL
        airborne = info.get("_kinematic_airborne")
        if airborne is None or airborne.shape != contacts.shape:
            airborne = ~contacts
        rear_airborne_balance = np.abs(airborne[:, 2].astype(np.float32) - airborne[:, 3].astype(np.float32))
        rear_clearance_balance = np.clip(
            np.abs(foot_height_above_terrain[:, 2] - foot_height_above_terrain[:, 3])
            / max(self._foot_swing_target_height, 1e-4),
            0.0,
            1.0,
        )

        dt = max(self._cfg.ctrl_dt, 1e-6)
        torques = np.clip(np.nan_to_num(data.actuator_ctrls, nan=0.0, posinf=0.0, neginf=0.0), -200.0, 200.0)
        dof_vel_safe = np.clip(np.nan_to_num(dof_vel, nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)
        last_dof_vel = np.clip(np.nan_to_num(info["last_dof_vel"], nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)
        dof_acc = np.clip((last_dof_vel - dof_vel_safe) / dt, -5000.0, 5000.0)
        undesired_contacts = self._get_undesired_contacts(data)
        expected_pitch = section01_expected_pitch(pose[:, 1])
        pitch_err = projected_gravity[:, 0] - expected_pitch
        roll_err = projected_gravity[:, 1]

        reward_terms = {
            "tracking_lin_vel": np.exp(-lin_vel_error / self._tracking_sigma),
            "tracking_ang_vel": np.exp(-ang_vel_error / self._tracking_sigma),
            "tracking_goal_vel": tracking_goal_vel,
            "tracking_yaw": tracking_yaw,
            "forward_progress": forward_progress,
            "target_progress": target_progress,
            "reach_goal": info["goal_reached_this_step"].astype(np.float32),
            "reach_all_goal": info["goal_done"].astype(np.float32),
            "lin_vel_z": np.square(local_lin_vel[:, 2]),
            "ang_vel_xy": np.sum(np.square(gyro[:, :2]), axis=1),
            "orientation": np.square(pitch_err) + np.square(roll_err),
            "torques": np.sum(np.square(torques.astype(np.float64)), axis=1).astype(np.float32),
            "dof_vel": np.sum(np.square(dof_vel_safe.astype(np.float64)), axis=1).astype(np.float32),
            "dof_acc": np.sum(np.square(dof_acc.astype(np.float64)), axis=1).astype(np.float32),
            "action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "dof_pos_limits": np.sum(
                np.clip(dof_pos - self.soft_joint_upper_limits, 0.0, None)
                + np.clip(self.soft_joint_lower_limits - dof_pos, 0.0, None),
                axis=1,
            ),
            "anti_stall": np.clip(speed_deficit - 0.15, 0.0, None) * active_move,
            "undesired_contacts": undesired_contacts,
            "base_contact": info.get("base_contact", np.zeros((data.shape[0],), dtype=bool)).astype(np.float32),
            "termination": np.logical_and(
                terminated,
                ~info["goal_done"],
            ).astype(np.float32),
        }

        # 只在落地瞬间给分。小于最短有效摆动时间不计分，达到目标时间即封顶，
        # 避免旧公式 (air_time - target) 把正常的 0.2--0.3 秒步态变成负奖励。
        landing_score = np.clip(
            (info["air_time_before_contact"] - self._feet_air_time_min)
            / (self._feet_air_time_target - self._feet_air_time_min),
            0.0,
            1.0,
        )
        landing_count = np.sum(info["first_contact"], axis=1)
        feet_air_time_reward = np.sum(
            landing_score * info["first_contact"],
            axis=1,
        ) / np.maximum(landing_count, 1.0)
        feet_air_time_reward *= active_move
        reward_terms["feet_air_time"] = feet_air_time_reward

        # ===== 步态对称奖励 (gait_symmetry) =====
        # 标准 trot 要求两个对角腿组反相：一组支撑时另一组摆动。
        # 四脚全着地或四脚全腾空时两组状态相同，因此不应获得步态奖励。
        diag_a = 0.5 * (c[:, 0] + c[:, 3])  # FR 与 RL
        diag_b = 0.5 * (c[:, 1] + c[:, 2])  # FL 与 RR
        gait_symmetry = np.abs(diag_a - diag_b) * active_move
        reward_terms["gait_symmetry"] = gait_symmetry
        reward_terms["rear_airborne_balance"] = rear_airborne_balance * active_move
        reward_terms["rear_clearance_balance"] = rear_clearance_balance * active_move

        # ===== 越障抬腿高度奖励 (swing_foot_height) =====
        # 崎岖区和坡道都需要摆动腿抬高，不能在崎岖区出口硬切断奖励。
        # foot_z 是足部世界高度；摆动腿 = 未接触的脚(contacts==0)。
        in_rough = self._rough_zone_mask(pose[:, :2])
        in_slope = np.logical_and(
            pose[:, 1] >= TERRAIN_SLOPE_Y_START,
            pose[:, 1] < TERRAIN_SLOPE_Y_END,
        )
        needs_clearance = np.logical_or(in_rough, in_slope).astype(np.float32)
        swing_mask = (1.0 - contacts.astype(np.float32))  # [n,4] 摆动腿=1
        # 每只摆动腿的抬腿高度已扣除各足下方的真实地形，并减掉足端几何固定偏置。
        # 只在确实有摆动腿时给分，并按摆动腿数量归一化，避免四脚着地时 exp(0)=1。
        swing_count = np.sum(swing_mask, axis=1)
        swing_height_err = np.sum(
            swing_mask * np.abs(foot_clearance - self._foot_swing_target_height),
            axis=1,
        ) / np.maximum(swing_count, 1.0)
        swing_foot_height = (
            np.exp(-swing_height_err)
            * (swing_count > 0.0)
            * needs_clearance
            * active_move
        )
        reward_terms["swing_foot_height"] = swing_foot_height

        # ===== 逼每条腿都迈步 (per_leg_swing) —— 治"后腿不动迈不过去" =====
        # 问题：狗只用前腿迈步、后腿搭地不动，遇到坎/起伏后腿迈不过去就摔。
        # G3 的 stale 只在足端高度越过离地阈值且向上运动时清零，避免接触传感器
        # 长期为真导致离地事件永远为零。
        leg_stale = info.get("_leg_stale_time")
        if leg_stale is None or leg_stale.shape != (data.shape[0], 4):
            leg_stale = np.zeros((data.shape[0], 4), dtype=np.float32)
        stale_span = max(self.LEG_STALE_FULL - self.LEG_STALE_GRACE, 1e-6)
        stale_penalty_by_leg = np.clip(
            (leg_stale - self.LEG_STALE_GRACE) / stale_span,
            0.0,
            1.0,
        )
        leg_stale_penalty = np.max(stale_penalty_by_leg, axis=1) * active_move
        rear_leg_stale_penalty = np.max(stale_penalty_by_leg[:, 2:4], axis=1) * active_move
        reward_terms["leg_stale_penalty"] = leg_stale_penalty
        reward_terms["rear_leg_stale_penalty"] = rear_leg_stale_penalty

        per_leg_swing, per_leg_swing_score = self._per_leg_swing_score(info, active_move)
        reward_terms["per_leg_swing"] = per_leg_swing

        rear_leg_stale = np.maximum(leg_stale[:, 2], leg_stale[:, 3])
        rear_activity = 1.0 - np.clip(
            (rear_leg_stale - self.LEG_STALE_GRACE) / stale_span,
            0.0,
            1.0,
        )

        # ===== 崎岖出口落差点处理(真实落差线 Y=1.5) =====
        # 现象:前腿先迈过Y=1.5踏上低0.28m平地,后腿还卡崎岖区高处没跟上->前低后高栽倒。
        # 关键是后腿在落差点要及时跟上。落差段(Y1.3~1.8)额外加大腿部活跃度要求。
        drop_zone_r = np.logical_and(pose[:, 1] > 1.3, pose[:, 1] <= 1.8).astype(np.float32)
        reward_terms["drop_leg_catchup"] = rear_activity * drop_zone_r
        reward_terms["drop_pitch"] = np.square(projected_gravity[:, 0]) * drop_zone_r

        # ===== 上坡段处理 —— 治"上坡前面栽下去" =====
        # 现象:狗用平地步态冲上15°坡,后腿没在坡上持续蹬地提供动力,前腿够不着更高坡面->前栽。
        # 坡道范围从碰撞体精确量出: C_Adiban_003 实际坡道 Y 2.065~6.894, 平台从 Y 6.83 起。
        #
        # 两段分别处理,避免和落差段交界处理(到1.8)之间留空档:
        #  - 后腿驱动(slope_leg_drive): 从 1.8 起接上交界处理,一路到坡顶6.9,
        #    过渡坎和坡道上都持续逼后腿蹬地,衔接连续不放手。
        #  - 前倾引导(slope_pitch): 只在真正的15°坡道(2.0~6.9)生效,
        #    过渡坎(1.8~2.0)还是平的,不强行要求15°前倾,避免误导。
        leg_drive_zone = np.logical_and(
            pose[:, 1] >= TERRAIN_SLOPE_Y_START,
            pose[:, 1] < TERRAIN_SLOPE_Y_END,
        ).astype(np.float32)
        rear_support_progress = self._rear_support_progress(forward_speed, contacts, active_move)
        reward_terms["slope_leg_drive"] = rear_support_progress * leg_drive_zone
        # 前腿上坡驱动 (slope_front_drive):与后腿对称,专门促使前腿(FR/FL,索引0/1)上坡积极迈步出力。
        # 单独保证前腿出力,使上坡时前后腿协同驱动。
        # 这里取前两腿(FR/FL)各自 stale 时间，奖励前腿在坡上积极抬迈。
        front_leg_stale = np.maximum(leg_stale[:, 0], leg_stale[:, 1])
        front_activity = 1.0 - np.clip(
            (front_leg_stale - self.LEG_STALE_GRACE) / stale_span,
            0.0,
            1.0,
        )
        reward_terms["slope_front_drive"] = front_activity * leg_drive_zone
        # 上坡身体应顺坡微前倾(约15°),而非保持水平或后仰。
        # projected_gravity[:,0]是重力在机身x轴投影,反映俯仰。坡上目标俯仰对应约sin(15°)=0.26。
        # 现象:踏上坡折角(Y=2.0)头朝坡上栽(前倾过头)。原目标0.26在折角处鼓励大幅前倾,
        # 叠加冲劲->栽头。改小到0.13(约7.5°):只需轻微前倾,配合-1.5权重压过度前倾。
        SLOPE_TARGET_PITCH = 0.13  # 温和前倾目标,防折角栽头
        slope_pitch_zone = np.logical_and(
            pose[:, 1] >= TERRAIN_SLOPE_Y_START,
            pose[:, 1] < TERRAIN_SLOPE_Y_END,
        ).astype(np.float32)
        reward_terms["slope_pitch"] = np.square(projected_gravity[:, 0] - SLOPE_TARGET_PITCH) * slope_pitch_zone

        # ===== 上坡髋关节张开惩罚 (slope_hip) —— 治"前腿髋关节突然张开、腿软倒" =====
        # 现象:狗上坡时前腿髋关节突然张开(向外岔)、腿一软就塌倒。
        # 上坡受力大时髋关节容易被撑开导致腿外岔,这里用 hip_indices 对髋关节张开做约束。
        # 这里只在上坡段(Y2.0~6.9)惩罚髋关节偏离默认收拢角度,逼四条腿(尤其前腿)髋保持收拢、
        # 腿在身体正下方稳稳撑住,不向外张开软倒。只在上坡生效,不影响前面已走好的崎岖/落差。
        reward_terms["slope_hip"] = self._slope_hip_opening_penalty(dof_pos, pose[:, 1])

        reward = np.zeros((data.shape[0],), dtype=np.float32)
        reward_contrib = {}
        for key, scale in self._reward_scales.items():
            term = reward_terms.get(key)
            if term is None:
                continue
            term = np.nan_to_num(term.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
            contrib = float(scale) * term
            reward += contrib
            reward_contrib[key] = contrib.astype(np.float32)

        info["Reward"] = reward_contrib
        metrics = info.get("metrics", {})
        metrics.update(
            {
                "goal_idx": info["goal_idx"].astype(np.float32),
                "distance_to_waypoint": info["distance_to_waypoint"].astype(np.float32),
                "target_progress": target_progress.astype(np.float32),
                "forward_progress": forward_progress.astype(np.float32),
                "base_height": pose[:, 2].astype(np.float32),
                "body_speed_xy": body_speed_xy.astype(np.float32),
                "command_speed_xy": command_speed_xy.astype(np.float32),
                "rough_mask": rough.astype(np.float32),
                "terrain_scan_max": np.max(terrain_scan, axis=1).astype(np.float32),
                "terrain_scan_front": terrain_scan[:, 0].astype(np.float32),
                "max_foot_z": np.max(foot_z, axis=1).astype(np.float32),
                "undesired_contacts": undesired_contacts.astype(np.float32),
                "rear_airborne_balance": rear_airborne_balance.astype(np.float32),
                "rear_clearance_balance": rear_clearance_balance.astype(np.float32),
                "min_leg_swing_score": np.min(per_leg_swing_score, axis=1).astype(np.float32),
                "rear_support_progress": (rear_support_progress * leg_drive_zone).astype(np.float32),
            }
        )
        metrics.update(self._diagnostic_metrics(data, info, pose, projected_gravity, dof_acc))
        info["metrics"] = metrics
        info["prev_distance_to_waypoint"] = distance_to_waypoint.astype(np.float32)
        return np.clip(reward, -100.0, 1000.0).astype(np.float32)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        info = state.info
        info["goal_reached_this_step"][:] = False
        sensor_contacts = self._get_foot_contacts(data)
        foot_height = self._get_foot_height_above_terrain(data)
        previous_height = info.get("_previous_foot_height_above_terrain")
        if previous_height is None or previous_height.shape != foot_height.shape:
            previous_height = foot_height.copy()
        previous_airborne = info.get("_kinematic_airborne")
        if previous_airborne is None or previous_airborne.shape != foot_height.shape:
            previous_airborne = np.zeros_like(foot_height, dtype=np.bool_)
        airborne, first_liftoff, first_touchdown, foot_vertical_velocity = self._kinematic_gait_events(
            foot_height,
            previous_height,
            previous_airborne,
        )
        contacts = ~airborne

        previous_swing_active = info.get("_leg_swing_active")
        if previous_swing_active is None or previous_swing_active.shape != airborne.shape:
            previous_swing_active = np.zeros_like(airborne, dtype=np.bool_)
        first_contact = np.logical_and(first_touchdown, previous_swing_active)
        swing_active = np.logical_and(
            np.logical_or(previous_swing_active, first_liftoff),
            ~first_touchdown,
        )

        air_time_before_contact = info["feet_air_time"].copy()
        info["first_contact"] = first_contact
        info["air_time_before_contact"] = air_time_before_contact
        info["feet_air_time"] = np.where(
            swing_active,
            air_time_before_contact + self._cfg.ctrl_dt,
            0.0,
        ).astype(np.float32)
        info["sensor_contacts"] = sensor_contacts
        info["contacts"] = contacts
        info["first_liftoff"] = first_liftoff
        info["first_touchdown"] = first_touchdown
        info["_kinematic_airborne"] = airborne
        info["_leg_swing_active"] = swing_active
        info["_foot_height_above_terrain"] = foot_height
        info["_foot_vertical_velocity"] = foot_vertical_velocity
        info["_previous_foot_height_above_terrain"] = foot_height.copy()
        stale = info.get("_leg_stale_time")
        if stale is None or stale.shape != (data.shape[0], 4):
            stale = np.zeros((data.shape[0], 4), dtype=np.float32)
        info["_leg_stale_time"] = np.where(
            first_liftoff,
            0.0,
            stale + self._cfg.ctrl_dt,
        ).astype(np.float32)
        last_air_time = info.get("_leg_last_air_time")
        if last_air_time is None or last_air_time.shape != (data.shape[0], 4):
            last_air_time = np.zeros((data.shape[0], 4), dtype=np.float32)
        info["_leg_last_air_time"] = np.where(
            first_contact,
            air_time_before_contact,
            last_air_time,
        ).astype(np.float32)

        self._update_waypoint_commands(data, info)
        terminated = self._compute_terminated(data, info)
        self._update_goal_success_metrics(info, terminated)
        reward = self._compute_reward(data, info, terminated)
        obs = self._get_obs(data, info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        num_envs = data.shape[0]
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        spawn_xy = self._spawn_center[:2] + np.random.uniform(
            low=self._spawn_low,
            high=self._spawn_high,
            size=(num_envs, 2),
        ).astype(np.float32)
        spawn_xyz = np.column_stack(
            [
                spawn_xy,
                np.full((num_envs,), float(self._spawn_center[2]), dtype=np.float32),
            ]
        )

        dof_pos[:, 3:6] = spawn_xyz
        yaw = np.full((num_envs,), 0.5 * np.pi, dtype=np.float32)
        yaw += np.random.uniform(-0.15, 0.15, size=(num_envs,)).astype(np.float32)
        dof_pos[:, 6:10] = self._quat_from_yaw(yaw)

        # 我们的绝对坐标路径点路线。本任务的 cfg 只保留最终目标点:
        # (-2.7, 8.0)，即 2026 平台。
        waypoints = np.tile(self._waypoint_targets[None, :, :], (num_envs, 1, 1)).astype(np.float32)
        final_targets = waypoints[:, -1, :].copy()

        joint_noise = np.random.uniform(
            low=-self._joint_noise_scale,
            high=self._joint_noise_scale,
            size=(num_envs, self._num_action),
        ).astype(np.float32)
        joint_dof_pos = self.default_angles + joint_noise

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._body.set_dof_pos(data, joint_dof_pos, include_floatingbase=False)
        self._body.set_dof_vel(data, np.zeros((num_envs, self._num_action), dtype=np.float32), include_floatingbase=False)
        self._model.forward_kinematic(data)

        # First three global DOFs are target marker x/y/yaw in vbot.xml.
        marker_dof = data.dof_pos.copy()
        marker_dof[:, 0] = final_targets[:, 0]
        marker_dof[:, 1] = final_targets[:, 1]
        marker_dof[:, 2] = 0.5 * np.pi
        data.set_dof_pos(marker_dof, self._model)
        self._model.forward_kinematic(data)
        initial_foot_height = self._get_foot_height_above_terrain(data)

        info = {
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "commands": np.zeros((num_envs, 3), dtype=np.float32),
            "commands_smoothed": np.zeros((num_envs, 3), dtype=np.float32),
            "contacts": np.zeros((num_envs, 4), dtype=np.bool_),
            "sensor_contacts": np.zeros((num_envs, 4), dtype=np.bool_),
            "feet_air_time": np.zeros((num_envs, 4), dtype=np.float32),
            "first_contact": np.zeros((num_envs, 4), dtype=np.bool_),
            "first_liftoff": np.zeros((num_envs, 4), dtype=np.bool_),
            "first_touchdown": np.zeros((num_envs, 4), dtype=np.bool_),
            "air_time_before_contact": np.zeros((num_envs, 4), dtype=np.float32),
            "_kinematic_airborne": np.zeros((num_envs, 4), dtype=np.bool_),
            "_leg_swing_active": np.zeros((num_envs, 4), dtype=np.bool_),
            "_foot_height_above_terrain": initial_foot_height.copy(),
            "_foot_vertical_velocity": np.zeros((num_envs, 4), dtype=np.float32),
            "_previous_foot_height_above_terrain": initial_foot_height.copy(),
            "spawn_z": spawn_xyz[:, 2].astype(np.float32).copy(),
            "waypoints": waypoints,
            "num_waypoints": np.full((num_envs,), waypoints.shape[1], dtype=np.int32),
            "goal_idx": np.zeros((num_envs,), dtype=np.int32),
            "goal_done": np.zeros((num_envs,), dtype=np.bool_),
            "goal_reached_this_step": np.zeros((num_envs,), dtype=np.bool_),
            "waypoint_wait_steps_left": np.zeros((num_envs,), dtype=np.int32),
            "target_xy": waypoints[:, 0, :].copy(),
            "distance_to_waypoint": np.zeros((num_envs,), dtype=np.float32),
            "prev_distance_to_waypoint": np.zeros((num_envs,), dtype=np.float32),
            "base_contact": np.zeros((num_envs,), dtype=np.bool_),
            "fall_like_termination": np.zeros((num_envs,), dtype=np.bool_),
            # 每条腿距上次离地的时间，供 max_leg_stale_time 诊断指标使用。
            # 必须在这里播种：NpEnv._reset_done_envs 只会按 done 掩码重置
            # reset() 返回的 info 里已有的 key，漏掉的累加器会跨 episode 一直累积。
            "_leg_stale_time": np.zeros((num_envs, 4), dtype=np.float32),
            "_leg_last_air_time": np.zeros((num_envs, 4), dtype=np.float32),
            "metrics": {},
        }
        self._update_waypoint_commands(data, info)
        info["prev_distance_to_waypoint"] = info["distance_to_waypoint"].astype(np.float32).copy()
        obs = self._get_obs(data, info)
        return obs, info
