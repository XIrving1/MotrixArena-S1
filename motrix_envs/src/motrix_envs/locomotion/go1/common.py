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

import numpy as np


def generate_repeating_array(N, L, i):
    """
    Generate an array of length L with values repeating in [0, N],
    starting with the value at index i of the period.

    Parameters:
    N: maximum value in the period (0 to N)
    L: length of the output array
    i: starting index within the period

    Returns:
    numpy array with the repeating pattern
    """
    # Create the base period [0, 1, 2, ..., N-1]
    period = np.arange(N)

    # Create the starting sequence by rolling the period to start at index i
    start_from_i = np.roll(period, -i)

    # Calculate how many full periods we need
    full_periods = L // len(start_from_i)
    remainder = L % len(start_from_i)

    # Create the array by repeating the rolled period
    result = np.tile(start_from_i, full_periods)

    # Add the remaining elements if needed
    if remainder > 0:
        result = np.concatenate([result, start_from_i[:remainder]])

    return result
