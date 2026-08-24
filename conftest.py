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

"""MotrixArena S1 测试套的共享 fixture。

放在仓库根而不是各自的 tests/ 下，是为了让 motrix_envs/tests 与 motrix_rl/tests
共用同一个 session 级模型 —— section01 场景加载约 10 秒 / 265MB 网格，
加载两次会让整套测试的时间翻倍。
"""

import os

import numpy as np
import pytest

# section01 场景的 DOF 布局（见 xmls/vbot.xml，实测 num_dof_pos == 36）：
#   0:3   目标标记 x / y / yaw（可视化用）
#   3:6   基座 xyz（freejoint）
#   6:10  基座四元数 xyzw
#   10:22 12 个腿关节
#   22:29 / 29:36  两个朝向箭头的 freejoint
DOF_BASE_XYZ = slice(3, 6)
DOF_BASE_QUAT = slice(6, 10)

TEST_NUM_ENVS = 32


@pytest.fixture(scope="session")
def section01_model():
    """加载 section01 的 SceneModel。

    必须传**绝对路径**：mtx.load_model() 的 assetdir 按 CWD 解析，
    相对路径会被二次拼接成 .../xmls/motrix_envs/src/.../assets/xxx.obj 而报错。
    """
    import motrixsim as mtx

    from motrix_envs.navigation.vbot.cfg import VBotSection01EnvCfg

    return mtx.load_model(os.path.abspath(VBotSection01EnvCfg().model_file))


@pytest.fixture(scope="session")
def section01_env():
    """构建 32 个并行环境的 section01 env（session 级，全套测试共用一份）。"""
    from motrix_envs import registry

    return registry.make("vbot_navigation_section01", sim_backend="np", num_envs=TEST_NUM_ENVS)


@pytest.fixture(scope="session")
def section01_env_stepped(section01_env):
    """已经初始化并步进过一次的 env —— 需要读 state.info 的测试用这个。"""
    env = section01_env
    if env.state is None:
        env.init_state()
    env.step(np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32))
    return env


def teleport_envs_along_course(env, y_values: np.ndarray) -> None:
    """把各个 env 的基座沿 Y 铺开到赛道各处，用于扫描"某奖励项在赛道上是否曾非零"。

    只改基座平移，不动关节角与朝向，这样各 env 之间唯一的差异就是位置。
    """
    data = env.state.data
    dof_pos = np.array(data.dof_pos, dtype=np.float32, copy=True)
    dof_pos[:, DOF_BASE_XYZ.start + 1] = y_values.astype(np.float32)
    # 抬到该 Y 处真实地面之上，避免直接生成在坡道内部
    from motrix_envs.navigation.vbot.vbot_section01_np import section01_terrain_height

    dof_pos[:, DOF_BASE_XYZ.start + 2] = section01_terrain_height(y_values) + 0.42
    data.set_dof_pos(dof_pos, env.model)
    env.model.forward_kinematic(data)
