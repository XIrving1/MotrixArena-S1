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

from dataclasses import dataclass


@dataclass
class DeviceSupports:
    torch: bool = False
    torch_gpu: bool = False
    jax: bool = False
    jax_gpu: bool = False


def _check_gpu_available_for_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros((1,)).cuda().numpy(force=True)
        return True
    except Exception:
        return False


def get_device_supports() -> DeviceSupports:
    supports = DeviceSupports()
    try:
        import torch  # noqa: F401

        supports.torch = True
        supports.torch_gpu = _check_gpu_available_for_torch()
    except ImportError:
        pass

    try:
        import jax  # noqa: F401

        supports.jax = True
        from jax.lib import xla_bridge

        platform = xla_bridge.get_backend().platform
        if platform == "gpu":
            supports.jax_gpu = True
    except ImportError:
        pass

    return supports
