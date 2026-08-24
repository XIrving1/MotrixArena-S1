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

"""Section01 策略的无头量化评估。

scripts/play.py 是给人看的：单个环境、60fps、无限渲染。它回答不了
"checkpoint 12000 比 15000 好吗"这种问题。本脚本是它的量化孪生：
几百个环境并行跑满一整个 episode，输出可比较的 JSON 指标。

用法：
    # 评估单个 checkpoint
    uv run scripts/eval.py --policy artifacts/checkpoints/B0/agent_15000.pt \\
        --num-envs 256 --out artifacts/eval/B0_agent15000.json

    # 扫一整个 run 的所有 checkpoint，输出 checkpoint-性能曲线
    uv run scripts/eval.py --policy-glob 'artifacts/checkpoints/B0/agent_*.pt' \\
        --num-envs 128 --out artifacts/eval/B0_sweep.json

    # 不指定 policy 时自动找最新 run 的 best_agent（同 play.py 的规则）
    uv run scripts/eval.py --num-envs 256

排序键 (success_rate, max_waypoint, final_y, -time_to_finish)：
成功率为 0 时第一个键退化，此时靠 waypoint 与 Y 分出高下 ——
这正是被删掉的 test_evaluation.py::test_checkpoint_ranking_uses_success_waypoint_y_and_time
的名字所记录的排序逻辑。
"""

import glob as globlib
import json
import pathlib
import time

import numpy as np
from absl import app, flags

from motrix_envs import registry as env_registry
from motrix_envs.navigation.vbot.vbot_section01_np import (
    TERRAIN_ROUGH_Y_END,
    TERRAIN_SLOPE_Y_END,
    TERRAIN_SLOPE_Y_START,
    TERRAIN_START_PLATFORM_Y_END,
)

_ENV = flags.DEFINE_string("env", "vbot_navigation_section01", "要评估的环境")
_POLICY = flags.DEFINE_string("policy", None, "checkpoint 路径（不指定则自动找最新 run 的 best_agent）")
_POLICY_GLOB = flags.DEFINE_string("policy-glob", None, "批量评估多个 checkpoint 的 glob 模式")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 256, "并行环境数 = 评估的 episode 数")
_MAX_STEPS = flags.DEFINE_integer("max-steps", 0, "每个 episode 最多跑多少步（0 = 用 cfg 的上限）")
_SEED = flags.DEFINE_integer("seed", 42, "随机种子")
_OUT = flags.DEFINE_string("out", None, "结果 JSON 的输出路径")

# 按 docs 里核算过的**真实**赛道边界分桶（不用代码里那套近似值）
FAILURE_ZONES = [
    ("start_platform Y<-1.5", -np.inf, TERRAIN_START_PLATFORM_Y_END),
    ("rough_hfield -1.5..1.5", TERRAIN_START_PLATFORM_Y_END, TERRAIN_ROUGH_Y_END),
    ("drop_lip 1.5..2.0", TERRAIN_ROUGH_Y_END, TERRAIN_SLOPE_Y_START),
    ("ramp 2.0..6.83", TERRAIN_SLOPE_Y_START, TERRAIN_SLOPE_Y_END),
    ("finish_platform >6.83", TERRAIN_SLOPE_Y_END, np.inf),
]


def find_best_policy(env_name: str) -> str:
    """找最新一次训练的最佳权重。

    这是 scripts/play.py:48-107 的同款逻辑，**复制**而非 import ——
    play.py 在模块级注册了同名的 absl flag（--env/--num-envs/--seed），
    import 它会和本脚本的 flag 冲突。20 行重复换取互不干扰，值得。
    """
    from motrix_rl.skrl import get_log_dir

    env_dir = pathlib.Path(get_log_dir(env_name))
    if not env_dir.exists():
        raise FileNotFoundError(f"找不到训练结果目录：{env_dir}")

    runs = [d for d in env_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"{env_dir} 下没有任何训练 run")

    # ⚠ 按 mtime 挑最新 —— 一旦开始新训练，这里就不再指向 B0 基线了。
    #   评估基线请显式传 --policy artifacts/checkpoints/B0/...
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    ckpt_dir = latest / "checkpoints"

    best = list(ckpt_dir.glob("best_agent.*"))
    if best:
        return str(best[0])

    candidates = list(ckpt_dir.glob("agent_*.pt")) + list(ckpt_dir.glob("agent_*.pickle"))
    if not candidates:
        raise FileNotFoundError(f"{ckpt_dir} 下没有权重文件")

    def timestep_of(path):
        parts = pathlib.Path(path).stem.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    return str(max(candidates, key=timestep_of))


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None, "min": None}
    return {
        "mean": round(float(np.mean(values)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(np.max(values)), 4),
        "min": round(float(np.min(values)), 4),
    }


def _build_agent(env_name: str, num_envs: int, seed: int, policy_path: str):
    """按 Trainer.play 的同款方式搭出 agent 并加载权重。

    复用 Trainer._make_model / _make_agent（虽是私有方法）而不是重写：
    模型结构必须和训练时**逐层一致**，否则 state_dict 加载会失败或静默错位。
    抄一份 130 行的网络构建代码是更糟的选择。
    """
    import torch
    from skrl import config
    from skrl.utils import set_seed

    from motrix_rl.skrl.torch import wrap_env
    from motrix_rl.skrl.torch.train import ppo as ppo_module

    config.torch.backend = "torch"

    env = env_registry.make(env_name, sim_backend="np", num_envs=num_envs)
    set_seed(seed)
    skrl_env = wrap_env(env, False)

    trainer = ppo_module.Trainer(env_name, sim_backend="np", enable_render=False)
    rlcfg = trainer._rlcfg
    models = trainer._make_model(skrl_env, rlcfg)
    ppo_cfg = ppo_module._get_cfg(rlcfg, skrl_env)  # log_dir=None -> 不写 TB、不存 checkpoint
    agent = trainer._make_agent(models, skrl_env, ppo_cfg)
    agent.load(policy_path)

    return env, skrl_env, agent, torch


def evaluate(env_name: str, policy_path: str, num_envs: int, seed: int, max_steps: int) -> dict:
    """跑 num_envs 个 episode，每个环境贡献恰好一个 episode 的统计。"""
    env, skrl_env, agent, torch = _build_agent(env_name, num_envs, seed, policy_path)

    step_limit = max_steps or env.cfg.max_episode_steps
    ctrl_dt = env.cfg.ctrl_dt

    obs, _ = skrl_env.reset()

    # 每个 env 只记录它的**第一个** episode
    still_recording = np.ones(num_envs, dtype=bool)
    max_y = np.full(num_envs, -np.inf, dtype=np.float64)
    max_waypoint = np.zeros(num_envs, dtype=np.float64)
    final_y = np.full(num_envs, np.nan, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype=np.int64)
    succeeded = np.zeros(num_envs, dtype=bool)
    ended_by_termination = np.zeros(num_envs, dtype=bool)
    ended_by_truncation = np.zeros(num_envs, dtype=bool)

    started = time.monotonic()
    with torch.no_grad():
        for step in range(step_limit):
            outputs = agent.act(obs, timestep=0, timesteps=0)
            # 确定性评估：取分布均值而不是采样
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, infos = skrl_env.step(actions)

            # ── 关键：NpEnv.step 在返回前就调用了 _reset_done_envs()，
            #    所以 done 环境的 obs / info["goal_idx"] / data 都已经是**重置后**的值。
            #    但 info["Reward"] 与 info["metrics"] 不在 reset() 返回的 info 里，
            #    replace_dict_values 不会碰它们 —— 于是这两个字典保留着重置前的真实值。
            #    评估必须走这条通道，否则每个"最终 Y"都会是出生点 -2.4。
            metrics = infos["metrics"]
            step_y = np.asarray(metrics["base_y"], dtype=np.float64)
            step_waypoint = np.asarray(metrics["goal_idx"], dtype=np.float64)
            reached_all = np.asarray(infos["Reward"]["reach_all_goal"], dtype=np.float64) != 0.0

            # wrapper 返回的是 torch tensor，且可能在 GPU 上
            term = terminated.detach().cpu().numpy().astype(bool).reshape(-1)
            trunc = truncated.detach().cpu().numpy().astype(bool).reshape(-1)
            done = term | trunc

            active = still_recording
            max_y[active] = np.maximum(max_y[active], step_y[active])
            max_waypoint[active] = np.maximum(max_waypoint[active], step_waypoint[active])
            episode_len[active] += 1

            newly_done = active & done
            if np.any(newly_done):
                final_y[newly_done] = step_y[newly_done]
                succeeded[newly_done] = reached_all[newly_done]
                # goal_done 也会置 terminated，所以先判成功再归因失败
                ended_by_termination[newly_done] = term[newly_done] & ~reached_all[newly_done]
                ended_by_truncation[newly_done] = trunc[newly_done] & ~term[newly_done]
                still_recording[newly_done] = False

            if not np.any(still_recording):
                break

    # 跑满步数上限仍未结束的，按截断处理
    unfinished = still_recording
    if np.any(unfinished):
        final_y[unfinished] = max_y[unfinished]
        ended_by_truncation[unfinished] = True

    wall_clock = time.monotonic() - started

    zone_hist = {}
    valid_final_y = final_y[~np.isnan(final_y)]
    for label, low, high in FAILURE_ZONES:
        in_zone = np.logical_and(valid_final_y >= low, valid_final_y < high)
        zone_hist[label] = round(float(np.mean(in_zone)), 4) if valid_final_y.size else 0.0

    finish_times = episode_len[succeeded] * ctrl_dt

    return {
        "checkpoint": str(policy_path),
        "env": env_name,
        "num_envs": num_envs,
        "seed": seed,
        "step_limit": int(step_limit),
        "success_rate": round(float(np.mean(succeeded)), 4),
        "max_waypoint_reached": _percentiles(max_waypoint),
        "final_y": _percentiles(valid_final_y),
        "max_y": _percentiles(max_y[np.isfinite(max_y)]),
        "time_to_finish_s": _percentiles(finish_times) if finish_times.size else None,
        "episode_len_steps": {
            **_percentiles(episode_len.astype(np.float64)),
            "frac_truncated": round(float(np.mean(ended_by_truncation)), 4),
        },
        "failure_zone_hist": zone_hist,
        "death_by": {
            "termination": round(float(np.mean(ended_by_termination)), 4),
            "truncation": round(float(np.mean(ended_by_truncation)), 4),
            "goal_done": round(float(np.mean(succeeded)), 4),
        },
        "eval_wall_clock_s": round(wall_clock, 1),
    }


def ranking_key(result: dict):
    """(成功率, waypoint 中位数, final_y 中位数, -完赛耗时) —— 越大越好。"""
    finish = result.get("time_to_finish_s")
    return (
        result["success_rate"],
        result["max_waypoint_reached"]["p50"] or 0.0,
        result["final_y"]["p50"] or 0.0,
        -(finish["p50"] if finish else 1e9),
    )


def print_report(result: dict) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {pathlib.Path(result['checkpoint']).name}   ({result['num_envs']} episodes, "
          f"seed {result['seed']}, {result['eval_wall_clock_s']}s)")
    print("=" * 74)
    print(f"  成功率              : {result['success_rate']:.1%}")
    wp = result["max_waypoint_reached"]
    print(f"  到达的最远路径点    : p50={wp['p50']:.0f}  p95={wp['p95']:.0f}  max={wp['max']:.0f}  (共 7 个)")
    fy = result["final_y"]
    print(f"  最终 Y              : p50={fy['p50']:.2f}  p95={fy['p95']:.2f}  max={fy['max']:.2f}  (终点 7.80)")
    ep = result["episode_len_steps"]
    print(f"  episode 长度        : mean={ep['mean']:.0f}/{result['step_limit']}  截断占比={ep['frac_truncated']:.1%}")
    if result["time_to_finish_s"]:
        print(f"  完赛耗时(秒)        : p50={result['time_to_finish_s']['p50']:.1f}")

    print("\n  失败位置分布（按真实赛道边界分桶）：")
    for label, frac in result["failure_zone_hist"].items():
        bar = "█" * int(round(frac * 40))
        print(f"    {label:26s} {frac:6.1%} {bar}")

    print("\n  结束原因：")
    for label, frac in result["death_by"].items():
        print(f"    {label:26s} {frac:6.1%}")


def main(argv):
    del argv

    if _POLICY_GLOB.value:
        policies = sorted(globlib.glob(_POLICY_GLOB.value))
        if not policies:
            raise SystemExit(f"--policy-glob 没匹配到任何文件：{_POLICY_GLOB.value}")
    elif _POLICY.value:
        policies = [_POLICY.value]
    else:
        policies = [find_best_policy(_ENV.value)]
        print(f"未指定 --policy，自动选中最新 run 的：{policies[0]}")

    results = []
    for policy in policies:
        print(f"\n评估 {policy} …")
        result = evaluate(_ENV.value, policy, _NUM_ENVS.value, _SEED.value, _MAX_STEPS.value)
        print_report(result)
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 74}\n  排名 (success, waypoint, final_y, -time)\n{'=' * 74}")
        for rank, result in enumerate(sorted(results, key=ranking_key, reverse=True), 1):
            print(
                f"  {rank:2d}. {pathlib.Path(result['checkpoint']).name:22s}"
                f"  success={result['success_rate']:.1%}"
                f"  wp_p50={result['max_waypoint_reached']['p50']:.0f}"
                f"  y_p50={result['final_y']['p50']:.2f}"
            )

    if _OUT.value:
        out_path = pathlib.Path(_OUT.value)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = results[0] if len(results) == 1 else {"results": results}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n已写入 {out_path}")

    print()


if __name__ == "__main__":
    app.run(main)
