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

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class ControlConfig:
    # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
    # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
    action_scale = 0.25  # 平地navigation使用0.25
    # torque_limit[N*m] 使用XML forcerange参数

@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.5]  
    
    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

    # the default angles for all joints. key = joint name, value = target angle [rad]
    # 使用locomotion的关节角度配置
    default_joint_angles = {
        "FR_hip_joint": -0.0,     # 右前髋关节
        "FR_thigh_joint": 0.9,    # 右前大腿
        "FR_calf_joint": -1.8,    # 右前小腿
        "FL_hip_joint": 0.0,      # 左前髋关节
        "FL_thigh_joint": 0.9,    # 左前大腿
        "FL_calf_joint": -1.8,    # 左前小腿
        "RR_hip_joint": -0.0,     # 右后髋关节
        "RR_thigh_joint": 0.9,    # 右后大腿
        "RR_calf_joint": -1.8,    # 右后小腿
        "RL_hip_joint": 0.0,      # 左后髋关节
        "RL_thigh_joint": 0.9,    # 左后大腿
        "RL_calf_joint": -1.8,    # 左后小腿
    }

@dataclass
class Commands:
    # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 相对机器人初始位置的偏移（米）
    # yaw: 目标绝对朝向（弧度），水平方向随机
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

@dataclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05

@dataclass
class Asset:
    body_name = "base"
    foot_names = ["FR", "FL", "RR", "RL"]
    terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
    ground_subtree = "C_"  # 地形根节点，用于subtree接触检测
   
@dataclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"
    feet = ["FR", "FL", "RR", "RL"]  # 足部接触力传感器名称

@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # ===== 导航任务核心奖励 =====
            "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
            "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
            "approach_reward": 1.0,         # 接近目标奖励（鼓励接近目标）
            "reach_bonus": 8.0,             # 到达目标奖励（鼓励接近目标）
            "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
            "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
            # ===== Locomotion稳定性奖励（保持但降低权重） =====
            "orientation": -0.05,           # 姿态稳定（降低权重）
            "lin_vel_z": -0.5,              # 垂直速度惩罚
            "ang_vel_xy": -0.05,            # XY轴角速度惩罚
            "torques": -1e-5,               # 扭矩惩罚
            "dof_vel": -5e-5,               # 关节速度惩罚
            "dof_acc": -2.5e-7,             # 关节加速度惩罚
            "action_rate": -0.01,           # 动作变化率惩罚
            
            # ===== 终止惩罚 =====
            "termination": -200.0,          # 终止惩罚
        }
    )

@dataclass
class NavigationConfig:
    # True: use body frame for velocity/position error/command; False: world frame
    use_body_frame: bool = True

@registry.envcfg("vbot_navigation_flat")
@dataclass
class VBotEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 10
    max_episode_steps: int = 1000
    sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


@registry.envcfg("vbot_navigation_stairs")
@dataclass
class VBotStairsEnvCfg(VBotEnvCfg):
    """VBot在楼梯地形上的导航配置，继承flat配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
    max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
    max_episode_steps: int = 2000
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
    control_config: ControlConfig = field(default_factory=ControlConfig)


@registry.envcfg("VBotStairsMultiTarget-v0")
@dataclass
class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯多目标导航配置，继承单目标配置"""
    max_episode_seconds: float = 60.0  # 多目标需要更长时间
    max_episode_steps: int = 6000


@registry.envcfg("vbot_navigation_stairs_obstacles")
@dataclass
class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
    """VBot楼梯地形带障碍球的导航配置"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
    max_episode_seconds: float = 20.0
    max_episode_steps: int = 2000

@registry.envcfg("vbot_navigation_long_course")
@dataclass
class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    """VBot三段地形完整导航配置（比赛任务）- 使用world.xml统一地图"""
    # 使用scene_world.xml作为完整的三段地形地图（集成了world.xml）
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_world.xml"
    max_episode_seconds: float = 60.0  # 优化：减少到60秒，加快训练速度
    max_episode_steps: int = 6000  # 对应60秒 @ 100Hz
    
    @dataclass
    class InitState:
        # 起始位置：section01的中心位置
        pos = [0.0, 0.0, 1.8]  # 高台中心，高度1.8m
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    
    @dataclass
    class Commands:
        # 目标范围：覆盖整个30米路线（section01:0-10m, section02:10-20m, section03:20-30m）
        pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
    @dataclass
    class ControlConfig:
        action_scale = 0.25  # 与stairs保持一致
    
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section001")
#通过 @registry.envcfg("vbot_navigation_section001") 注册
@dataclass
class VBotSection001EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section001.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, -2.4, 0.5]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section01")
#通过 @registry.envcfg("vbot_navigation_section01") 注册
@dataclass
class VBotSection01EnvCfg(VBotStairsEnvCfg):
    """Section01 越障导航配置：机身坐标系平滑速度指令、足端接触观测、腾空时间塑形、前方地形采样观测与基座触地终止，沿一组路径点逐段引导到达 2026 平台。"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section01.xml"
    render_spacing: float = 0.0
    max_episode_seconds: float = 40.0
    max_episode_steps: int = 4000

    @dataclass
    class InitState:
        # Section01 起点区域。
        pos: tuple[float, float, float] = (0.0, -2.4, 0.5)
        pos_randomization_range: tuple[float, float, float, float] = (-0.5, -0.5, 0.5, 0.5)
        yaw_range: tuple[float, float] = (-0.15, 0.15)
        joint_noise_scale: float = 0.03

        default_joint_angles: dict[str, float] = field(
            default_factory=lambda: {
                "FR_hip_joint": -0.0,
                "FR_thigh_joint": 0.9,
                "FR_calf_joint": -1.8,
                "FL_hip_joint": 0.0,
                "FL_thigh_joint": 0.9,
                "FL_calf_joint": -1.8,
                "RR_hip_joint": -0.0,
                "RR_thigh_joint": 0.9,
                "RR_calf_joint": -1.8,
                "RL_hip_joint": 0.0,
                "RL_thigh_joint": 0.9,
                "RL_calf_joint": -1.8,
            }
        )

    @dataclass
    class Commands:
        # 沿路线布置的分段路径点。密集的路径点让 reach_goal 在训练中持续提供信号；
        # 它们是"途经门"而非停留点，机器人依次穿过即可，不需要在每个点停下。
        command_mode: str = "waypoint_nav"
        waypoint_targets: tuple[tuple[float, float], ...] = (
            (0.0, -0.60),
            (0.0, 1.20),
            (0.0, 2.25),
            (0.0, 4.00),
            (0.0, 6.00),
            (0.0, 7.00),
            (0.0, 7.80),
        )
        waypoint_reach_threshold: float = 0.45
        waypoint_wait_seconds: float = 0.0
        waypoint_lin_kp: float = 0.8
        waypoint_ang_kp: float = 1.0
        waypoint_cmd_smooth_tau: float = 0.25
        vel_limit: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-0.6, -0.35, -1.0),
            (0.9, 0.35, 1.0),
        )
        rough_vel_limit: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-0.25, -0.25, -0.8),
            (0.45, 0.25, 0.8),  # 前进上限0.65->0.45：崎岖区更慢更稳，减少高速绊倒
        )
        pose_command_range: tuple[float, float, float, float, float, float] = (0.0, 7.8, 0.0, 0.0, 7.8, 0.0)

    @dataclass
    class ControlConfig:
        # VBot 是较小的机器人，这里在保证稳定的前提下采用较大的动作缩放，
        # 使摆动幅度更大、步幅更舒展。
        action_scale: float = 0.5  # 试验：0.45->0.5，步幅更大，观察是否改善后腿步态
        stiffness: float = 80.0
        damping: float = 6.0

    @dataclass
    class RewardConfig:
        scales: dict[str, float] = field(
            default_factory=lambda: {
                # 奖励核心：速度跟踪 + 紧凑的正则项 + 接触塑形。
                "termination": -10.0,
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
                "torques": -0.00001,
                "dof_vel": -0.0001,
                "dof_acc": -2.5e-7,
                "action_rate": -0.01,
                "feet_air_time": 1.0,  # 0.8->1.0，更鼓励抬腿摆动，增大步幅
                "anti_stall": -0.8,
                "dof_pos_limits": -0.5,
                "undesired_contacts": -1.0,
                "base_contact": -10.0,
                # ===== 我的新增奖励项 =====
                "per_leg_swing": 3.0,      # 逼每条腿都迈步，治后腿不动（核心，权重给大）
                # G3：stale 由足端高度+速度离地事件重置，不再读取不可靠的接触跳变。
                "leg_stale_penalty": -0.5,
                "rear_leg_stale_penalty": -1.0,
                "gait_symmetry": 0.2,      # 鼓励对角腿同步的对角(trot)步态
                "rear_airborne_balance": -0.8,  # G4：RR/RL 离地占比不能长期分叉
                "rear_clearance_balance": -0.4,  # G4：RR/RL 足端高度保持接近
                "swing_foot_height": 1.5,  # 崎岖区抬腿高度奖励，治崎岖区绊倒
                "drop_leg_catchup": 4.0,   # 核心：落差点(Y=1.5)额外逼后腿跟上，治"前腿出去后腿没出"
                "drop_pitch": -2.0,        # 落差段压身体前倾，辅助
                # ===== 上坡段处理(Y 1.5~7.0)，治"上坡前面栽下去" =====
                "slope_leg_drive": 3.0,    # 上坡后腿接触支撑且机身前进时给分，避免只抬腿不发力
                "slope_pitch": 0.0,        # 姿态已由 terrain-relative orientation 统一约束
                "slope_hip": -1.0,         # 上坡髋关节张开惩罚，治前腿髋张开软倒
                "slope_front_drive": 3.0,  # 前腿上坡驱动，逼前腿积极迈步出力
            }
        )
        tracking_sigma: float = 0.2
        feet_air_time_min: float = 0.12
        feet_air_time_target: float = 0.30
        foot_lift_on_height: float = 0.055
        foot_lift_off_height: float = 0.035
        foot_lift_min_velocity: float = 0.05
        foot_swing_target_height: float = 0.12

    @dataclass
    class Asset:
        body_name: str = "base"
        foot_names: list[str] = field(default_factory=lambda: ["FR", "FL", "RR", "RL"])
        terminate_after_contacts_on: list[str] = field(
            default_factory=lambda: [
                "collision_middle_box",
                "collision_head_box",
            ]
        )
        undesired_contacts_on: list[str] = field(
            default_factory=lambda: [
                "collision_mainlink_box",
                "collision_motor_cylinder",
                "collision_revo_left_box",
                "collision_revo_right_box",
                "collision_upper_box",
                "collision_lower_box",
                "collision_backprotect_box",
                "collision_low_bar_FL_box",
                "collision_low_bar_FR_box",
                "collision_low_bar_RL_box",
                "collision_low_bar_RR_box",
            ]
        )
        ground_subtree: list[str] = field(default_factory=lambda: ["C1_", "C2_", "C3_"])

    @dataclass
    class Sensor:
        base_linvel: str = "base_linvel"
        base_gyro: str = "base_gyro"
        feet: list[str] = field(default_factory=lambda: ["FR", "FL", "RR", "RL"])
        foot_force_sensors: list[str] = field(
            default_factory=lambda: [
                "FR_foot_contact_1",
                "FL_foot_contact_1",
                "RR_foot_contact_1",
                "RL_foot_contact_1",
            ]
        )
        foot_contact_sensor_groups: list[list[str]] = field(
            default_factory=lambda: [
                ["FR_foot_contact_1", "FR_foot_contact_2", "FR_foot_contact_3"],
                ["FL_foot_contact_1", "FL_foot_contact_2", "FL_foot_contact_3"],
                ["RR_foot_contact_1", "RR_foot_contact_2", "RR_foot_contact_3"],
                ["RL_foot_contact_1", "RL_foot_contact_2", "RL_foot_contact_3"],
            ]
        )
        base_contact_sensors: list[str] = field(default_factory=lambda: ["base_contact_1", "base_contact_2", "base_contact_3"])

    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)

@registry.envcfg("vbot_navigation_section011")
#通过 @registry.envcfg("vbot_navigation_section011") 注册
@dataclass
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, 7.5, 2.0]  # 中心位置
        pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section012")
#通过 @registry.envcfg("vbot_navigation_section012") 注册
@dataclass
class VBotSection012EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section012.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [-2.5, 15.0, 3.3]  # 中心位置
        pos_randomization_range = [-0., -0., 0., 0.]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)

@registry.envcfg("vbot_navigation_section013")
#通过 @registry.envcfg("vbot_navigation_section013") 注册
@dataclass
class VBotSection013EnvCfg(VBotStairsEnvCfg):
    """VBot Section01单独训练配置 - 高台楼梯地形"""
    model_file: str = os.path.dirname(__file__) + "/xmls/scene_section013.xml"
    max_episode_seconds: float = 40.0  # 拉长一倍：从20秒增加到40秒
    max_episode_steps: int = 4000  # 拉长一倍：从2000步增加到4000步
    @dataclass
    class InitState:
        # 起始位置：随机化范围内生成
        pos = [0.0, 26.0, 3.3]  # 中心位置
        pos_randomization_range = [-0., -0., 0., 0.]  # X±0.5m, Y±0.5m随机

        default_joint_angles = {
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
        }
    @dataclass
    class Commands:
        # 目标位置：缩短距离，固定目标点
        # 起始位置Y=-2.4, 目标Y=3.6, 距离=6米（与vbot_np相近）
        # pose_command_range = [0.0, 3.6, 0.0, 0.0, 3.6, 0.0]
        # 原始配置（已注释）：
        # 目标位置：固定在终止角范围远端（完全无随机化）
        # 固定目标点: X=0, Y=10.2, Z=2 (Z通过XML控制)
        # 起始位置Y=-2.4, 目标Y=10.2, 距离=12.6米
        pose_command_range = [0.0, 10.2, 0.0, 0.0, 10.2, 0.0]
    @dataclass
    class ControlConfig:
        action_scale = 0.25
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    control_config: ControlConfig = field(default_factory=ControlConfig)
