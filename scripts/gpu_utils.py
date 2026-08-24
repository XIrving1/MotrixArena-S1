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

import jax.numpy as jnp
import pynvml


def monitor_gpu_utilization(stop_event, gpu_index=0, interval=1.0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    utilization_samples = []

    while not stop_event.is_set():
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization_samples.append(util.gpu)
        stop_event.wait(interval)

    pynvml.nvmlShutdown()

    if utilization_samples:
        data = jnp.array(utilization_samples)
        print(f"GPU utilization statistics over {len(data)} samples:")
        print(f"  Mean: {jnp.mean(data):.2f}%")
        print(f"  Max : {jnp.max(data):.2f}%")
        print(f"  Min : {jnp.min(data):.2f}%")
        print(f"  Median : {jnp.median(data):.2f}%")
    else:
        print("No GPU utilization samples recorded.")
