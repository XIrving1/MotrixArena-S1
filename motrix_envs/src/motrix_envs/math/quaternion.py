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


class Quaternion:
    def multiply(q1, q2):
        """
        Quaternion multiply, with [w, x, y, z] format
        """
        qx1, qy1, qz1, qw1 = q1[0], q1[1], q1[2], q1[3]
        qx2, qy2, qz2, qw2 = q2[0], q2[1], q2[2], q2[3]

        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

        return np.array([qx, qy, qz, qw], dtype=np.float32)

    def from_euler(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
        """
        Euler convert to quaternion, with [w, x, y, z] format
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.stack([qx, qy, qz, qw], dtype=np.float32, axis=-1)

    def rotate_vector(quats: np.ndarray, v: np.ndarray):
        """
        Rotate a list vectors v by a list of quaternions using a vectorized approach. v could be a simple shape (3,)
        vector, or a shape (N,3) vector array with quats shape (N,4)

        Parameters:
            quats (np.ndarray): Array of quaternions of shape (N, 4). Each quaternion is in [w, x, y, z] format.
            v (np.ndarray): Fixed vector of shape (3,) to be rotated.

        Returns:
            np.ndarray: Array of rotated vectors of shape (N, 3).
        """
        # Normalize the quaternions to ensure they are unit quaternions

        # Extract the scalar (w) and vector (x, y, z) parts of the quaternions
        w = quats[:, -1]  # Shape (N,)
        im = quats[:, :3]  # Shape (N, 3)

        t = 2 * np.cross(im, v)
        return v + w.reshape(-1, 1) * t + np.cross(im, t)

    def rotate_inverse(quats, v):
        """
        Rotate a list of vectors v by a list of inverse quaternions using a vectorized approach.

        Parameters:
            quats (np.ndarray): Array of quaternions of shape (N, 4). Each quaternion is in [w, x, y, z] format.
            v (np.ndarray): Fixed vector of shape (3,) to be rotated.

        Returns:
            np.ndarray: Array of rotated vectors of shape (N, 3).
        """
        # Normalize the quaternions to ensure they are unit quaternions

        # Extract the scalar (w) and vector (x, y, z) parts of the quaternions
        w = quats[:, -1]  # Shape (N,)
        im = quats[:, :3]  # Shape (N, 3)

        # Compute the cross product between the imaginary part of each quaternion and the fixed vector v.
        # np.cross broadcasts v to match each row in im, resulting in an array of shape (N, 3)
        cross_im_v = np.cross(im, v)

        term = cross_im_v - w.reshape(-1, 1) * v

        # Final result: v' = v + 2 * r × term
        v_rotated = v + 2 * np.cross(im, term)

        return v_rotated

    def similarity(q_current, q_target):
        """
        Use NumPy to compute attitude alignment reward between two batches of quaternions.

        Parameters:
            q_current (np.ndarray): Quaternion of current pose, shape (num_envs, 4).
            q_target (np.ndarray): Quaternion of target pose, shape (num_envs, 4) or (4,).
                                    If (4,), it will be broadcast to all environments.

        Returns:
            np.ndarray: Reward value for each environment, shape (num_envs,). Reward value range is [-1, 1].
        """
        # Ensure input is float array
        q_current = q_current.astype(np.float32)
        q_target = q_target.astype(np.float32)

        # If q_target is a single quaternion, broadcast to all environments
        if q_target.ndim == 1:
            # Use np.tile for broadcasting
            q_target = np.tile(q_target, (q_current.shape[0], 1))

        # Step 1: Compute conjugate of q_current
        # Conjugate of quaternion (w, x, y, z) is (w, -x, -y, -z)
        q_current_conj = np.copy(q_current)
        q_current_conj[..., 1:] *= -1  # Negate x, y, z components

        # Step 2: Compute relative quaternion q_rel = q_target * q_current_conj
        # Unpack quaternion components for computation
        w1, x1, y1, z1 = q_target[..., 0], q_target[..., 1], q_target[..., 2], q_target[..., 3]
        w2, x2, y2, z2 = q_current_conj[..., 0], q_current_conj[..., 1], q_current_conj[..., 2], q_current_conj[..., 3]

        # Apply quaternion multiplication formula
        w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        # For numerical stability, clamp w_rel to [-1.0, 1.0] range
        w_rel_clamped = np.clip(w_rel, -1.0, 1.0)

        # Step 3: Compute rotation angle theta
        theta = 2.0 * np.arccos(w_rel_clamped)

        # Step 4: Compute reward
        reward = np.cos(theta)

        return reward

    def get_yaw(quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # Compute yaw angle (rotation around Z axis)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)
