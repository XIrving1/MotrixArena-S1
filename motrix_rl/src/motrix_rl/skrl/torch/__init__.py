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


def _inherits_from(cls, base_class_name):
    """Check if cls inherits from a class with the given base_class_name."""
    return any(base.__name__ == base_class_name for base in cls.__mro__)


def wrap_env(env, enable_render: bool = False):
    """Wrap the environment based on its type."""
    if _inherits_from(env.__class__, "NpEnv"):
        from motrix_rl.skrl.torch.wrap_np import SkrlNpWrapper

        return SkrlNpWrapper(env, enable_render=enable_render)
    else:
        raise ValueError(f"Unsupported environment type: {env.__class__.__name__}")
