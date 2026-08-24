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

"""B0 基线 checkpoint 的结构校验。

为什么值得测：SKRL 的 checkpoint 里除了策略网络，还有 optimizer 状态与两个
RunningStandardScaler（观测/价值的归一化统计量）。**少了 state_preprocessor
就无法正确推理** —— 观测归一化统计量丢失会让策略看到完全不同尺度的输入。
这是"权重能加载但表现莫名其妙很差"最常见的原因。

权重本体在 gitignore 的 artifacts/ 下，不在仓库里；文件缺失时整组 skip，
保证新克隆下测试套仍然是绿的。sha256 记录在 docs/section01/checkpoints.sha256。
"""

import hashlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "B0"
CHECKSUM_FILE = REPO_ROOT / "docs" / "section01" / "checkpoints.sha256"

# 一个可续训的 SKRL PPO checkpoint 必须包含的模块
REQUIRED_MODULES = ["policy", "value", "optimizer", "state_preprocessor", "value_preprocessor"]


def _load(name: str):
    torch = pytest.importorskip("torch", reason="需要 skrl-torch 后端")
    path = BASELINE_CKPT_DIR / name
    if not path.exists():
        pytest.skip(f"基线权重不存在（artifacts/ 不进 git）：{path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_personal_baseline_contains_full_resume_state():
    """best_agent.pt 必须是可续训的完整状态，而不是只有网络权重。"""
    checkpoint = _load("best_agent.pt")
    assert isinstance(checkpoint, dict), f"checkpoint 顶层不是 dict，而是 {type(checkpoint)}"

    missing = [key for key in REQUIRED_MODULES if key not in checkpoint]
    assert not missing, f"checkpoint 缺少这些模块，无法续训：{missing}"


def test_resume_checkpoint_requires_all_training_modules():
    """两个 RunningStandardScaler 都必须带着统计量，否则推理时观测尺度会错。"""
    checkpoint = _load("agent_15000.pt")

    for key in ("state_preprocessor", "value_preprocessor"):
        preprocessor = checkpoint[key]
        assert preprocessor, f"{key} 是空的"
        # RunningStandardScaler 的状态里应当有 running_mean / running_variance
        keys = set(preprocessor.keys()) if hasattr(preprocessor, "keys") else set()
        assert any("mean" in str(k) for k in keys), f"{key} 里找不到 running_mean，统计量丢失：{sorted(keys)}"


def test_inference_only_checkpoint_is_rejected():
    """只含 policy 的 checkpoint 应当被判定为不可续训。

    这条测的是我们自己的校验逻辑，不依赖磁盘上的文件 ——
    它定义了"什么叫一个完整的 checkpoint"。
    """
    inference_only = {"policy": {"fake": 1}}
    missing = [key for key in REQUIRED_MODULES if key not in inference_only]
    assert missing == ["value", "optimizer", "state_preprocessor", "value_preprocessor"]


def test_baseline_checkpoints_match_recorded_checksums():
    """B0 权重没有被后续训练覆盖过 —— play.py 按 mtime 挑 run，很容易误伤。

    sha256 记录在 docs/section01/checkpoints.sha256（进 git），
    权重本体在 artifacts/（不进 git）。两者对不上就说明基线被动过了。
    """
    if not CHECKSUM_FILE.exists():
        pytest.skip(f"校验和文件不存在：{CHECKSUM_FILE}")

    expected = {}
    for line in CHECKSUM_FILE.read_text().strip().splitlines():
        digest, name = line.split()
        expected[name.lstrip("*")] = digest

    assert expected, "校验和文件是空的"

    for name, want in expected.items():
        path = BASELINE_CKPT_DIR / name
        if not path.exists():
            pytest.skip(f"基线权重不存在：{path}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == want, f"{name} 的内容变了！基线可能已被新训练覆盖。期望 {want[:16]}…，实际 {got[:16]}…"
