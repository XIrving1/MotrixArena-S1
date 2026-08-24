# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import VBotSection001EnvCfg


@registry.env("vbot_navigation_section001", "np")
class VBotSection001Env(NpEnv):
    """
    VBot Section001 平地 locomotion 训练环境
    完全参考亚军团队 vbot_section011_base_np.py 实现
    """

    _cfg: VBotSection001EnvCfg

    # 4阶段命令课程（完全参考亚军）
    SECTION001_COMMAND_CURRICULUM = (
        {
            "name": "stage1",
            "vel_low": (-0.2, -0.05, -0.2),
            "vel_high": (0.3, 0.05, 0.2),
            "stand_prob": 0.2,
            "min_speed_xy": 0.08,
            "spawn_y_range": (-3.0, 2.0),
            "yaw_range": (-0.8, 0.8),
            "upgrade_min_total_steps": 3_000_000,
            "upgrade_forward_progress_ema": 0.12,
            "upgrade_fall_ratio_ema": 0.08,
            "upgrade_episode_len_ema": 1200.0,
        },
        {
            "name": "stage2",
            "vel_low": (-0.3, -0.15, -0.4),
            "vel_high": (0.5, 0.15, 0.4),
            "stand_prob": 0.10,
            "min_speed_xy": 0.12,
            "spawn_y_range": (-3.0, 4.5),
            "yaw_range": (-1.6, 1.6),
            "upgrade_min_total_steps": 6_000_000,
            "upgrade_forward_progress_ema": 0.18,
            "upgrade_fall_ratio_ema": 0.06,
            "upgrade_episode_len_ema": 1400.0,
        },
        {
            "name": "stage3",
            "vel_low": (-0.5, -0.3, -0.6),
            "vel_high": (0.7, 0.3, 0.6),
            "stand_prob": 0.05,
            "min_speed_xy": 0.18,
            "spawn_y_range": (-3.0, 7.0),
            "yaw_range": (-3.1415926, 3.1415926),
            "upgrade_min_total_steps": 10_000_000,
            "upgrade_forward_progress_ema": 0.24,
            "upgrade_fall_ratio_ema": 0.04,
            "upgrade_episode_len_ema": 1600.0,
        },
        {
            "name": "stage4",
            "vel_low": (-0.6, -0.4, -0.8),
            "vel_high": (0.8, 0.4, 0.8),
            "stand_prob": 0.05,
            "min_speed_xy": 0.2,
            "spawn_y_range": (-3.0, 9.0),
            "yaw_range": (-3.1415926, 3.1415926),
            "upgrade_min_total_steps": 0,
            "upgrade_forward_progress_ema": 0.0,
            "upgrade_fall_ratio_ema": 1.0,
            "upgrade_episode_len_ema": 0.0,
        },
    )

    # 奖励scales（完全参考亚军）
    SECTION001_LOCO_REWARD_SCALES = {
        "termination": -10.0,
        "tracking_lin_vel": 1.2,
        "tracking_ang_vel": 0.6,
        "forward_progress": 1.0,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "torques": -0.00001,
        "dof_vel": -0.0001,
        "dof_acc": -2.5e-7,
        "action_rate": -0.01,
        "feet_air_time": 0.2,
        "stand_still": -0.1,
        "dof_pos_limits": 0.0,
        "hip_pos": -0.2,
        "calf_pos": -0.1,
        "joint_pos": -0.03,
        "anti_stall": -0.8,
        "base_height_drop": -1.5,
    }

    def __init__(self, cfg: VBotSection001EnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._body = self._model.get_body(cfg.asset.body_name)

        self._num_action = self._model.num_actuators
        self._num_dof_vel = self._model.num_dof_vel
        self._num_dof_pos = self._model.num_dof_pos

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_action,), dtype=np.float32)
        # obs: base 48 + foot contact force 12 = 60
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)
        self._base_quat_start = 6
        self._base_quat_end = 10
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
            dtype=np.float32
        )

        # 默认关节角度 + hip/calf索引 + 关节限位
        self.default_angles = np.zeros((self._num_action,), dtype=np.float32)
        self.hip_indices = []
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

        # 软限位（90%硬限位，参考亚军）
        soft_limit_ratio = 0.9
        limit_center = 0.5 * (self.joint_lower_limits + self.joint_upper_limits)
        half_range = 0.5 * (self.joint_upper_limits - self.joint_lower_limits)
        soft_half_range = half_range * soft_limit_ratio
        self.soft_joint_lower_limits = limit_center - soft_half_range
        self.soft_joint_upper_limits = limit_center + soft_half_range

        # 命令重采样步数范围（从cfg读取，参考亚军）
        command_min_t, command_max_t = cfg.commands.resample_time_range
        self._command_resample_min_steps = max(1, int(round(float(command_min_t) / max(cfg.ctrl_dt, 1e-6))))
        self._command_resample_max_steps = max(1, int(round(float(command_max_t) / max(cfg.ctrl_dt, 1e-6))))

        # 出生点配置（参考亚军）
        self._spawn_points = np.asarray(cfg.init_state.spawn_points, dtype=np.float32).reshape(-1, 3)
        if self._spawn_points.shape[0] == 0:
            self._spawn_points = np.asarray([cfg.init_state.pos], dtype=np.float32)
        self._spawn_xy_noise = float(cfg.init_state.spawn_xy_noise)
        self._spawn_yaw_min = float(cfg.init_state.yaw_range[0])
        self._spawn_yaw_max = float(cfg.init_state.yaw_range[1])
        self._joint_noise_scale = float(cfg.init_state.joint_noise_scale)
        spawn_y_range = getattr(cfg.init_state, "spawn_y_range", (-1e9, 1e9))
        self._spawn_y_min = float(spawn_y_range[0])
        self._spawn_y_max = float(spawn_y_range[1])

        # 脚部接触传感器分组（从cfg读取，参考亚军）
        sensor_cfg = cfg.sensor
        self._foot_contact_sensor_groups = [list(group) for group in list(sensor_cfg.foot_contact_sensor_groups)]
        self._base_contact_sensors = list(sensor_cfg.base_contact_sensors)
        self._missing_contact_sensors = set()

        # 终止接触检测初始化（参考亚军）
        self._init_termination_contact()

        # 奖励scales（从cfg读取后用SECTION001_LOCO_REWARD_SCALES覆盖，参考亚军）
        reward_scales = dict(getattr(cfg.reward_config, "scales", {}))
        reward_scales.update(self.SECTION001_LOCO_REWARD_SCALES)
        self._reward_scales = reward_scales
        self._tracking_sigma = max(float(cfg.reward_config.tracking_sigma), 1e-4)
        self._feet_air_time_target = float(cfg.reward_config.feet_air_time_target)

        # landing protection（参考亚军）
        landing_cfg = getattr(cfg, "landing_protection", None)
        self._landing_protection_enable = bool(getattr(landing_cfg, "enable", False))
        self._zero_command_before_landing = bool(getattr(landing_cfg, "zero_command_before_landing", True))
        self._landing_min_contact_feet = max(1, int(getattr(landing_cfg, "min_contact_feet", 1)))

        # 课程学习运行时状态（参考亚军）
        self._curriculum_stage_idx = 0
        self._curriculum_total_steps = 0
        self._curriculum_ema_alpha = 0.02
        self._curriculum_forward_progress_ema = None
        self._curriculum_fall_ratio_ema = None
        self._curriculum_episode_len_ema = None
        self._episode_step_counter = np.zeros((self._num_envs,), dtype=np.int32)
        self._last_forward_progress = np.zeros((self._num_envs,), dtype=np.float32)
        self._last_fall_like_termination = np.zeros((self._num_envs,), dtype=bool)
        self._curriculum_auto_upgrade = True

        # 从cfg读取手动固定stage（参考亚军）
        manual_stage = int(getattr(cfg.commands, "curriculum_stage", 0))
        manual_stage = int(np.clip(manual_stage, 0, len(self.SECTION001_COMMAND_CURRICULUM)))
        if manual_stage > 0:
            self._curriculum_stage_idx = manual_stage - 1
            self._curriculum_auto_upgrade = False
            self._print_curriculum_stage(prefix="[Info] 手动固定速度课程")
        else:
            self._print_curriculum_stage(prefix="[Info] 初始化速度课程")

    def _init_termination_contact(self):
        """初始化终止接触检测（完全参考亚军实现）"""
        termination_contact_names = list(getattr(self._cfg.asset, "terminate_after_contacts_on", []))
        ground_subtree = getattr(self._cfg.asset, "ground_subtree", "C_")
        if isinstance(ground_subtree, str):
            ground_prefixes = [ground_subtree]
        else:
            ground_prefixes = [str(p) for p in ground_subtree]

        ground_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is None:
                continue
            if any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        ground_geoms = list(dict.fromkeys(ground_geoms))

        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except BaseException as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")

        if termination_contact_list:
            self.termination_contact = np.asarray(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = int(self.termination_contact.shape[0])
            print(f"[Info] 初始化终止接触检测: {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")

    def _safe_get_sensor_value(self, sensor_name: str, data: mtx.SceneData):
        """安全获取传感器值（参考亚军）"""
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
            return np.zeros((num_envs,), dtype=bool)
        norms = np.linalg.norm(sensor_value, axis=-1)
        return norms > threshold

    def _get_base_contacts(self, data: mtx.SceneData) -> np.ndarray:
        """获取基座接触（备用方案，参考亚军）"""
        num_envs = data.shape[0]
        base_contact = np.zeros((num_envs,), dtype=bool)
        for sensor_name in self._base_contact_sensors:
            val = self._safe_get_sensor_value(sensor_name, data)
            base_contact |= self._sensor_value_to_contact(val, num_envs)
        return base_contact

    def _get_foot_contacts(self, data: mtx.SceneData) -> np.ndarray:
        """获取脚部接触（参考亚军）"""
        num_envs = data.shape[0]
        num_feet = len(self._foot_contact_sensor_groups)
        contacts = np.zeros((num_envs, num_feet), dtype=bool)
        for i, group in enumerate(self._foot_contact_sensor_groups):
            for sensor_name in group:
                val = self._safe_get_sensor_value(sensor_name, data)
                contacts[:, i] |= self._sensor_value_to_contact(val, num_envs)
        return contacts

    def _get_foot_contact_force(self, data: mtx.SceneData, root_quat: np.ndarray) -> np.ndarray:
        """获取脚部接触力并转到机体坐标系（参考亚军）"""
        num_envs = data.shape[0]
        foot_contact_force = np.zeros((num_envs, 12), dtype=np.float32)
        for i, group in enumerate(self._foot_contact_sensor_groups):
            group_force = np.zeros((num_envs, 3), dtype=np.float32)
            for sensor_name in group:
                val = self._safe_get_sensor_value(sensor_name, data)
                if val is not None:
                    val = np.clip(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)
                    group_force += val
            cf_body = Quaternion.rotate_inverse(root_quat, group_force)
            foot_contact_force[:, i*3:(i+1)*3] = cf_body / 500.0
        return foot_contact_force

    def _current_curriculum(self) -> dict:
        idx = int(np.clip(self._curriculum_stage_idx, 0, len(self.SECTION001_COMMAND_CURRICULUM) - 1))
        return self.SECTION001_COMMAND_CURRICULUM[idx]

    def _print_curriculum_stage(self, prefix: str):
        stage = self._current_curriculum()
        print(
            f"{prefix}: {stage['name']} "
            f"vel=({stage['vel_low']} -> {stage['vel_high']}), "
            f"stand_prob={stage['stand_prob']}, min_speed_xy={stage['min_speed_xy']}"
        )

    def _ema_update(self, old_value, value: float) -> float:
        if old_value is None:
            return float(value)
        alpha = self._curriculum_ema_alpha
        return (1.0 - alpha) * float(old_value) + alpha * float(value)

    def _try_upgrade_curriculum(self):
        """尝试升级课程（完全参考亚军）"""
        if not self._curriculum_auto_upgrade:
            return
        if self._curriculum_stage_idx >= len(self.SECTION001_COMMAND_CURRICULUM) - 1:
            return
        stage = self._current_curriculum()
        if self._curriculum_total_steps < int(stage["upgrade_min_total_steps"]):
            return
        if self._curriculum_forward_progress_ema is None or self._curriculum_fall_ratio_ema is None:
            return
        if self._curriculum_episode_len_ema is None:
            return
        if self._curriculum_forward_progress_ema < float(stage["upgrade_forward_progress_ema"]):
            return
        if self._curriculum_fall_ratio_ema > float(stage["upgrade_fall_ratio_ema"]):
            return
        if self._curriculum_episode_len_ema < float(stage["upgrade_episode_len_ema"]):
            return

        self._curriculum_stage_idx += 1
        print(
            "[Info] 升级条件满足: "
            f"forward_progress_ema={self._curriculum_forward_progress_ema:.3f}, "
            f"fall_ratio_ema={self._curriculum_fall_ratio_ema:.3f}, "
            f"episode_len_ema={self._curriculum_episode_len_ema:.1f}, "
            f"total_steps={self._curriculum_total_steps}"
        )
        self._print_curriculum_stage(prefix="[Info] 速度课程升级")

    def _sample_commands(self, num_envs: int) -> np.ndarray:
        """随机采样速度命令（完全参考亚军）"""
        stage = self._current_curriculum()
        low = np.asarray(stage["vel_low"], dtype=np.float32)
        high = np.asarray(stage["vel_high"], dtype=np.float32)
        commands = np.random.uniform(low=low, high=high, size=(num_envs, 3)).astype(np.float32)

        min_speed_xy = float(stage["min_speed_xy"])
        stand_prob = float(stage["stand_prob"])
        stand_mask = np.random.random(size=(num_envs,)) < stand_prob if stand_prob > 0 else np.zeros(num_envs, dtype=bool)

        if min_speed_xy > 0.0:
            move_idx = np.where(~stand_mask)[0]
            if move_idx.size > 0:
                for _ in range(5):
                    speed = np.linalg.norm(commands[move_idx, :2], axis=1)
                    weak = move_idx[speed < min_speed_xy]
                    if weak.size == 0:
                        break
                    commands[weak] = np.random.uniform(low=low, high=high, size=(weak.size, 3)).astype(np.float32)
                speed = np.linalg.norm(commands[move_idx, :2], axis=1)
                still_weak = move_idx[speed < min_speed_xy]
                if still_weak.size > 0:
                    h = np.random.uniform(-np.pi, np.pi, size=(still_weak.size,))
                    commands[still_weak, 0] = np.clip(min_speed_xy * np.cos(h), low[0], high[0])
                    commands[still_weak, 1] = np.clip(min_speed_xy * np.sin(h), low[1], high[1])

        commands[stand_mask] = 0.0
        return commands

    def _sample_command_steps(self, num_envs: int) -> np.ndarray:
        """采样命令持续步数（参考亚军）"""
        return np.random.randint(
            self._command_resample_min_steps,
            self._command_resample_max_steps + 1,
            size=(num_envs,)
        ).astype(np.int32)

    def _resample_commands(self, info: dict, env_mask=None):
        """重采样速度命令（参考亚军）"""
        if env_mask is None:
            env_indices = np.arange(info["commands"].shape[0])
        else:
            env_indices = np.where(env_mask)[0]
        if env_indices.size == 0:
            return
        info["commands"][env_indices] = self._sample_commands(env_indices.size)
        info["command_steps_left"][env_indices] = self._sample_command_steps(env_indices.size)

    @staticmethod
    def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
        half = 0.5 * yaw
        quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
        quat[:, 2] = np.sin(half)
        quat[:, 3] = np.cos(half)
        return quat

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """应用动作（参考亚军）"""
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions: np.ndarray, data: mtx.SceneData) -> np.ndarray:
        """PD控制器（参考亚军）"""
        action_scaled = actions * float(self._cfg.control_config.action_scale)
        target_pos = self.default_angles + action_scaled
        current_pos = self.get_dof_pos(data)
        current_vel = self.get_dof_vel(data)
        kp = float(getattr(self._cfg.control_config, "stiffness", 80.0))
        kd = float(getattr(self._cfg.control_config, "damping", 6.0))
        torques = kp * (target_pos - current_pos) - kd * current_vel
        if self._num_action == 12:
            torque_limits = np.asarray([17.0, 17.0, 34.0] * 4, dtype=np.float32)
        else:
            torque_limits = np.full((self._num_action,), 34.0, dtype=np.float32)
        return np.clip(torques, -torque_limits, torque_limits)

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """60维观测（完全参考亚军）"""
        cfg = self._cfg
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_lin_vel_world = self._model.get_sensor_value(cfg.sensor.base_linvel, data)
        local_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel_world)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        local_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        joint_pos_rel = dof_pos - self.default_angles

        obs_base = np.concatenate([
            local_lin_vel * cfg.normalization.lin_vel,
            gyro * cfg.normalization.ang_vel,
            local_gravity,
            joint_pos_rel * cfg.normalization.dof_pos,
            dof_vel * cfg.normalization.dof_vel,
            info["current_actions"],
            info["commands"] * self.commands_scale,
        ], axis=-1)
        assert obs_base.shape == (data.shape[0], 48)

        foot_contact_force = self._get_foot_contact_force(data, root_quat)
        obs = np.concatenate([obs_base, foot_contact_force], axis=-1)
        assert obs.shape == (data.shape[0], 60)
        return obs.astype(np.float32)

    def _compute_terminated(self, data: mtx.SceneData) -> np.ndarray:
        """终止条件（完全参考亚军）"""
        num_envs = data.shape[0]
        fall_like_termination = np.zeros((num_envs,), dtype=bool)

        # 优先用collision-pair检测（参考亚军）
        if getattr(self, "num_termination_check", 0) > 0 and self.termination_contact is not None:
            cquerys = self._model.get_contact_query(data)
            termination_check = cquerys.is_colliding(self.termination_contact)
            termination_check = termination_check.reshape((num_envs, self.num_termination_check))
            base_contact = termination_check.any(axis=1)
            fall_like_termination = np.logical_or(fall_like_termination, base_contact)
        else:
            # 回退：用base_contact传感器
            fall_like_termination = np.logical_or(fall_like_termination, self._get_base_contacts(data))

        pose = self._body.get_pose(data)
        base_lin_vel_world = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        speed_overflow = np.sum(np.square(base_lin_vel_world[:, :2]), axis=1) > 1e8
        invalid = np.isnan(pose).any(axis=1) | np.isnan(self.get_dof_vel(data)).any(axis=1)

        terminated = np.logical_or(fall_like_termination, speed_overflow)
        terminated = np.logical_or(terminated, invalid)
        self._last_fall_like_termination = fall_like_termination.copy()
        return terminated

    def _compute_reward(self, data: mtx.SceneData, info: dict, terminated: np.ndarray) -> np.ndarray:
        """奖励函数（完全参考亚军）"""
        cfg = self._cfg
        num_envs = data.shape[0]

        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_lin_vel_world = self._model.get_sensor_value(cfg.sensor.base_linvel, data)
        local_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel_world)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        dof_pos = self.get_dof_pos(data)
        dof_vel = self.get_dof_vel(data)
        commands = info["commands"]

        lin_vel_error = np.sum(np.square(commands[:, :2] - local_lin_vel[:, :2]), axis=1)
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        command_speed_xy = np.linalg.norm(commands[:, :2], axis=1)
        body_speed_xy = np.linalg.norm(local_lin_vel[:, :2], axis=1)
        speed_deficit = np.clip(command_speed_xy - body_speed_xy, 0.0, None)
        active_move = command_speed_xy > 0.1
        cmd_dir = commands[:, :2] / np.maximum(command_speed_xy[:, None], 1e-6)
        forward_speed = np.sum(local_lin_vel[:, :2] * cmd_dir, axis=1)
        forward_progress = np.clip(forward_speed, 0.0, 1.5) * active_move

        spawn_z = info.get("spawn_z", np.full((num_envs,), 0.5, dtype=np.float32))
        base_height_drop = np.clip((spawn_z - pose[:, 2]) - 0.12, 0.0, None)

        dt = max(cfg.ctrl_dt, 1e-6)
        torques_p = np.clip(np.nan_to_num(data.actuator_ctrls, nan=0.0, posinf=0.0, neginf=0.0), -200.0, 200.0)
        dof_vel_p = np.clip(np.nan_to_num(dof_vel, nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)
        last_dof_vel_p = np.clip(np.nan_to_num(info["last_dof_vel"], nan=0.0, posinf=0.0, neginf=0.0), -500.0, 500.0)
        dof_acc_p = np.clip((last_dof_vel_p - dof_vel_p) / dt, -5000.0, 5000.0)

        first_contact = info.get("first_contact", np.zeros((num_envs, 4), dtype=np.bool_))
        air_time = info.get("air_time_before_contact", np.zeros((num_envs, 4), dtype=np.float32))
        feet_air_time_reward = np.sum((air_time - self._feet_air_time_target) * first_contact, axis=1)
        feet_air_time_reward *= command_speed_xy > 0.1

        reward_terms = {
            "tracking_lin_vel": np.exp(-lin_vel_error / self._tracking_sigma),
            "tracking_ang_vel": np.exp(-ang_vel_error / self._tracking_sigma),
            "forward_progress": forward_progress,
            "lin_vel_z": np.square(local_lin_vel[:, 2]),
            "ang_vel_xy": np.sum(np.square(gyro[:, :2]), axis=1),
            "orientation": np.sum(np.square(projected_gravity[:, :2]), axis=1),
            "torques": np.sum(np.square(torques_p.astype(np.float64)), axis=1).astype(np.float32),
            "dof_vel": np.sum(np.square(dof_vel_p.astype(np.float64)), axis=1).astype(np.float32),
            "dof_acc": np.sum(np.square(dof_acc_p.astype(np.float64)), axis=1).astype(np.float32),
            "action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "stand_still": np.sum(np.abs(dof_pos - self.default_angles), axis=1) * (np.linalg.norm(commands, axis=1) < 0.1),
            "dof_pos_limits": np.sum(
                np.clip(dof_pos - self.soft_joint_upper_limits, 0.0, None)
                + np.clip(self.soft_joint_lower_limits - dof_pos, 0.0, None), axis=1),
            "joint_pos": np.sum(np.square(dof_pos - self.default_angles), axis=1) * active_move,
            "anti_stall": speed_deficit * active_move,
            "base_height_drop": base_height_drop,
            "termination": terminated.astype(np.float32),
            "feet_air_time": feet_air_time_reward,
            "hip_pos": (0.8 - np.abs(commands[:, 1])) * np.sum(
                np.square(dof_pos[:, self.hip_indices] - self.default_angles[self.hip_indices]), axis=1
            ) if len(self.hip_indices) > 0 else np.zeros(num_envs, dtype=np.float32),
            "calf_pos": (0.8 - np.abs(commands[:, 1])) * np.sum(
                np.square(dof_pos[:, self.calf_indices] - self.default_angles[self.calf_indices]), axis=1
            ) if len(self.calf_indices) > 0 else np.zeros(num_envs, dtype=np.float32),
        }

        reward = np.zeros(num_envs, dtype=np.float32)
        for key, scale in self._reward_scales.items():
            term = reward_terms.get(key)
            if term is not None:
                reward += float(scale) * term.astype(np.float32)

        self._last_forward_progress = forward_progress.astype(np.float32)
        return np.clip(reward, -100.0, 1000.0).astype(np.float32)

    def _update_command_curriculum_metrics(self, info: dict, terminated: np.ndarray):
        """更新课程学习EMA指标（完全参考亚军）"""
        num_envs = terminated.shape[0]
        if num_envs == 0:
            return

        self._curriculum_total_steps += num_envs
        self._episode_step_counter += 1

        forward_mean = float(np.mean(self._last_forward_progress)) if self._last_forward_progress.size > 0 else 0.0
        fall_ratio = float(np.mean(self._last_fall_like_termination)) if self._last_fall_like_termination.size > 0 else 0.0

        self._curriculum_forward_progress_ema = self._ema_update(self._curriculum_forward_progress_ema, forward_mean)
        self._curriculum_fall_ratio_ema = self._ema_update(self._curriculum_fall_ratio_ema, fall_ratio)

        if self._cfg.max_episode_steps:
            will_truncate = (info["steps"] + 1) >= self._cfg.max_episode_steps
        else:
            will_truncate = np.zeros((num_envs,), dtype=bool)
        done_now = np.logical_or(terminated, will_truncate)
        if np.any(done_now):
            lens = self._episode_step_counter[done_now].astype(np.float32)
            if len(lens) > 0:
                self._curriculum_episode_len_ema = self._ema_update(self._curriculum_episode_len_ema, float(np.mean(lens)))
            self._episode_step_counter[done_now] = 0

        self._try_upgrade_curriculum()

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """更新环境状态（完全参考亚军）"""
        data = state.data
        info = state.info

        # 更新脚部接触
        contacts = self._get_foot_contacts(data)
        air_time_before_contact = info["feet_air_time"].copy()
        info["first_contact"] = np.logical_and(air_time_before_contact > 0.0, contacts)
        info["air_time_before_contact"] = air_time_before_contact
        info["feet_air_time"] = np.where(contacts, 0.0, air_time_before_contact + self._cfg.ctrl_dt)
        info["contacts"] = contacts

        # 更新步数
        info["steps"] = info.get("steps", np.zeros(data.shape[0], dtype=np.int32)) + 1

        # 计算终止、奖励、观测
        terminated = self._compute_terminated(data)
        reward = self._compute_reward(data, info, terminated)
        self._update_command_curriculum_metrics(info, terminated)
        obs = self._get_obs(data, info)

        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """重置环境（完全参考亚军）"""
        num_envs = data.shape[0]

        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        # 根据当前stage选择出生范围（参考亚军）
        stage = self._current_curriculum()
        stage_y_min, stage_y_max = stage.get("spawn_y_range", (self._spawn_y_min, self._spawn_y_max))
        stage_y_min = max(self._spawn_y_min, float(stage_y_min))
        stage_y_max = min(self._spawn_y_max, float(stage_y_max))
        stage_yaw_min, stage_yaw_max = stage.get("yaw_range", (self._spawn_yaw_min, self._spawn_yaw_max))

        valid_mask = np.logical_and(
            self._spawn_points[:, 1] >= stage_y_min,
            self._spawn_points[:, 1] <= stage_y_max
        )
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            valid_indices = np.arange(self._spawn_points.shape[0], dtype=np.int64)

        spawn_indices = np.random.choice(valid_indices, size=(num_envs,), replace=True)
        spawn_xyz = self._spawn_points[spawn_indices].copy()
        spawn_xyz[:, :2] += np.random.uniform(
            low=-self._spawn_xy_noise,
            high=self._spawn_xy_noise,
            size=(num_envs, 2)
        ).astype(np.float32)

        # yaw随机化
        yaw = np.random.uniform(stage_yaw_min, stage_yaw_max, size=(num_envs,)).astype(np.float32)

        # 设置位置（完全参考官方原始文件）
        dof_pos[:, 3:6] = spawn_xyz

        # 设置朝向（完全参考官方原始文件）
        quat_yaw = self._quat_from_yaw(yaw)
        dof_pos[:, self._base_quat_start:self._base_quat_end] = quat_yaw

        # 关节噪声（参考亚军）
        joint_noise = np.random.uniform(
            low=-self._joint_noise_scale,
            high=self._joint_noise_scale,
            size=(num_envs, self._num_action)
        ).astype(np.float32)
        joint_dof_pos = self.default_angles + joint_noise

        # 先reset恢复初始状态，再设置位置和朝向
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # 单独设置关节角度（不影响freejoint四元数）
        self._body.set_dof_pos(data, joint_dof_pos, include_floatingbase=False)
        self._body.set_dof_vel(
            data,
            np.zeros((num_envs, self._num_action), dtype=np.float32),
            include_floatingbase=False
        )
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "commands": np.zeros((num_envs, 3), dtype=np.float32),
            "command_steps_left": np.zeros((num_envs,), dtype=np.int32),
            "contacts": np.zeros((num_envs, len(self._foot_contact_sensor_groups)), dtype=np.bool_),
            "feet_air_time": np.zeros((num_envs, len(self._foot_contact_sensor_groups)), dtype=np.float32),
            "first_contact": np.zeros((num_envs, len(self._foot_contact_sensor_groups)), dtype=np.bool_),
            "air_time_before_contact": np.zeros((num_envs, len(self._foot_contact_sensor_groups)), dtype=np.float32),
            "spawn_z": spawn_xyz[:, 2].astype(np.float32).copy(),
            "steps": np.zeros((num_envs,), dtype=np.int32),
        }

        # 采样初始命令（参考亚军：reset时采样，整个episode不变）
        self._resample_commands(info)

        obs = self._get_obs(data, info)
        print(f"obs.shape:{obs.shape}")
        return obs, info