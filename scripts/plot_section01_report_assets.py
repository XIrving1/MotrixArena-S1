#!/usr/bin/env python3
"""Generate Section01 report figures from archived runs and eval JSON files."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "vbot_navigation_section01"
EVAL = ROOT / "artifacts" / "eval"
OUT = ROOT / "docs" / "section01" / "media"

FONT = "Microsoft YaHei, Noto Sans CJK SC, sans-serif"
INK = "#172b4d"
MUTED = "#64748b"
GRID = "#dbe4ee"
BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
RED = "#dc2626"
PURPLE = "#7c3aed"

RUN_DIRS = {
    "B0": "26-08-04_12-21-58-225910_PPO",
    "A1": "26-08-11_18-44-15-659466_PPO",
    "G5": "26-08-19_10-26-06-165590_PPO",
    "G6": "26-08-19_16-06-56-576397_PPO",
    "G7": "26-08-19_19-02-14-566423_PPO",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 24,
    weight: int = 400,
    anchor: str = "start",
    fill: str = INK,
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>'
    )


def svg_start(title: str, subtitle: str, width: int = 1600, height: int = 900) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(60, 66, title, size=38, weight=700),
        text(60, 105, subtitle, size=22, fill=MUTED),
    ]


def write_svg(name: str, parts: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts) + "\n", encoding="utf-8")


def event_series(run: str, tag: str) -> tuple[list[float], list[float]]:
    run_dir = RUNS / RUN_DIRS[run]
    event = next(run_dir.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    return [float(v.step) for v in values], [float(v.value) for v in values]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def polyline(points: list[tuple[float, float]], color: str, width: int = 5) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def line_panel(
    parts: list[str],
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title_value: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float], str]],
    background_bands: list[tuple[float, float, str, str]] | None = None,
) -> None:
    left, right, top, bottom = 88, 28, 62, 72
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom

    def sx(value: float) -> float:
        return px0 + (value - x_min) / (x_max - x_min) * (px1 - px0)

    def sy(value: float) -> float:
        return py1 - (value - y_min) / (y_max - y_min) * (py1 - py0)

    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="{GRID}" stroke-width="2"/>')
    parts.append(text(x0 + 28, y0 + 40, title_value, size=26, weight=700))

    if background_bands:
        for low, high, color, label in background_bands:
            band_top = sy(min(high, y_max))
            band_bottom = sy(max(low, y_min))
            parts.append(f'<rect x="{px0}" y="{band_top:.1f}" width="{px1-px0:.1f}" height="{max(0, band_bottom-band_top):.1f}" fill="{color}" opacity="0.38"/>')
            parts.append(text(px1 - 8, (band_top + band_bottom) / 2 + 7, label, size=17, anchor="end", fill=MUTED))

    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        label = f"{tick:.0f}" if abs(tick) >= 10 or tick.is_integer() else f"{tick:.1f}"
        parts.append(text(px0 - 12, y + 7, label, size=18, anchor="end", fill=MUTED))

    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py1}" stroke="{GRID}" stroke-width="1"/>')
        label = f"{tick/1000:.0f}k" if tick >= 1000 else f"{tick:.0f}"
        parts.append(text(x, py1 + 30, label, size=18, anchor="middle", fill=MUTED))

    parts.append(f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="{INK}" stroke-width="2"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="{INK}" stroke-width="2"/>')
    parts.append(text((px0 + px1) / 2, y0 + height - 20, x_label, size=19, anchor="middle", fill=MUTED))
    parts.append(text(x0 + 24, (py0 + py1) / 2, y_label, size=19, anchor="middle", fill=MUTED, rotate=-90))

    legend_x = px0 + 8
    for name, xs, ys, color in series:
        pts = [(sx(x), sy(max(y_min, min(y_max, y)))) for x, y in zip(xs, ys)]
        parts.append(polyline(pts, color))
        parts.append(f'<line x1="{legend_x}" y1="{y0+76}" x2="{legend_x+34}" y2="{y0+76}" stroke="{color}" stroke-width="5"/>')
        parts.append(text(legend_x + 42, y0 + 83, name, size=18, fill=INK))
        legend_x += 150


def plot_baseline_progress() -> None:
    data = json.loads((EVAL / "B0_sweep.json").read_text(encoding="utf-8"))["results"]

    def checkpoint_step(item: dict) -> int:
        match = re.search(r"agent_(\d+)\.pt$", item["checkpoint"])
        if not match:
            raise ValueError(item["checkpoint"])
        return int(match.group(1))

    data.sort(key=checkpoint_step)
    steps = [checkpoint_step(item) for item in data]
    final_y = [float(item["final_y"]["p50"]) for item in data]
    trunc = [100.0 * float(item["episode_len_steps"]["frac_truncated"]) for item in data]

    parts = svg_start(
        "B0 baseline：训练在进步，但任务仍未完成",
        "来源：B0 checkpoint sweep，64 env，seed 42；15 个 checkpoint 的 success 均为 0%",
    )
    line_panel(
        parts,
        x0=60,
        y0=145,
        width=720,
        height=660,
        title_value="最终位置中位数",
        x_min=1000,
        x_max=15000,
        y_min=-2.5,
        y_max=7.5,
        x_ticks=[1000, 5000, 10000, 15000],
        y_ticks=[-2.0, 0.0, 2.0, 4.0, 6.0],
        x_label="trainer timesteps",
        y_label="final Y p50 (m)",
        series=[("B0", steps, final_y, BLUE)],
        background_bands=[
            (-2.5, -1.5, "#e2e8f0", "起步平台"),
            (-1.5, 1.5, "#fde68a", "崎岖区"),
            (1.5, 2.0, "#fed7aa", "落差段"),
            (2.0, 6.83, "#bfdbfe", "坡道"),
            (6.83, 7.5, "#bbf7d0", "终点平台"),
        ],
    )
    line_panel(
        parts,
        x0=820,
        y0=145,
        width=720,
        height=660,
        title_value="截断率呈 U 形",
        x_min=1000,
        x_max=15000,
        y_min=0,
        y_max=100,
        x_ticks=[1000, 5000, 10000, 15000],
        y_ticks=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        x_label="trainer timesteps",
        y_label="truncation (%)",
        series=[("B0", steps, trunc, RED)],
    )
    parts.append(text(60, 855, "读图：final Y p50 从 -1.90 m 升到 6.23 m，但坡上摔倒未被终止检测覆盖，最终截断率回升到 81.2%。", size=22, fill=INK))
    write_svg("b0_baseline_progress.svg", parts)


def plot_training_curves() -> None:
    b0_x, b0_y = event_series("B0", "Reward / Total reward (mean)")
    a1_x, a1_y = event_series("A1", "Reward / Total reward (mean)")
    g5_x, g5_y = event_series("G5", "metrics / base_y (mean)")
    g6_x, g6_y = event_series("G6", "metrics / base_y (mean)")
    g7_x, g7_y = event_series("G7", "metrics / base_y (mean)")

    parts = svg_start(
        "训练曲线：终止修复与奖励连续性",
        "曲线来自 run 内 TensorBoard event；A1 日志分辨率由每 1000 步提升到每 100 步",
    )
    line_panel(
        parts,
        x0=60,
        y0=145,
        width=720,
        height=660,
        title_value="B0 → A1：相同 15k 预算",
        x_min=0,
        x_max=15000,
        y_min=-1500,
        y_max=6000,
        x_ticks=[0, 5000, 10000, 15000],
        y_ticks=[-1000.0, 0.0, 2000.0, 4000.0, 6000.0],
        x_label="trainer timesteps",
        y_label="total reward mean",
        series=[
            ("B0", b0_x, moving_average(b0_y, 2), RED),
            ("A1", a1_x, moving_average(a1_y, 8), BLUE),
        ],
    )
    line_panel(
        parts,
        x0=820,
        y0=145,
        width=720,
        height=660,
        title_value="G5 → G6/G7：硬阈值移除后恢复推进",
        x_min=0,
        x_max=45000,
        y_min=-3.0,
        y_max=3.5,
        x_ticks=[0, 15000, 30000, 45000],
        y_ticks=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        x_label="trainer timesteps",
        y_label="base Y mean (m)",
        series=[
            ("G5", g5_x, moving_average(g5_y, 15), RED),
            ("G6", g6_x, moving_average(g6_y, 15), GREEN),
            ("G7", g7_x, moving_average(g7_y, 15), PURPLE),
        ],
    )
    parts.append(text(60, 855, "注意：G 系列奖励函数不同，因此右图使用物理位置 base Y，而不是直接比较总奖励。", size=22, fill=INK))
    write_svg("training_curves.svg", parts)


def plot_eval_outcomes() -> None:
    names = ["B0", "A1", "A3", "G3", "G5", "G6", "G7"]
    files = {
        "B0": "B0_agent15000.json",
        "A1": "A1.json",
        "A3": "A3.json",
        "G3": "G3.json",
        "G5": "G5.json",
        "G6": "G6.json",
        "G7": "G7.json",
    }
    records = {name: json.loads((EVAL / files[name]).read_text(encoding="utf-8")) for name in names}

    parts = svg_start(
        "评估结果：成功率提升与失败位置迁移",
        "B0 为历史 64-env 口径；A1 及后续为 256-env、确定性动作、最多 4000 步",
    )
    x0, y0, width, height = 60, 145, 720, 660
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="{GRID}" stroke-width="2"/>')
    parts.append(text(x0 + 28, y0 + 40, "成功率与截断率", size=26, weight=700))
    px0, px1, py0, py1 = x0 + 80, x0 + width - 30, y0 + 85, y0 + height - 80
    for tick in [0, 20, 40, 60, 80, 100]:
        y = py1 - tick / 100 * (py1 - py0)
        parts.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(text(px0 - 12, y + 7, tick, size=18, anchor="end", fill=MUTED))
    group_w = (px1 - px0) / len(names)
    bar_w = 26
    for index, name in enumerate(names):
        center = px0 + group_w * (index + 0.5)
        success = 100 * records[name]["success_rate"]
        trunc = 100 * records[name]["episode_len_steps"]["frac_truncated"]
        for value, color, offset in [(success, BLUE, -bar_w), (trunc, ORANGE, 2)]:
            bar_h = value / 100 * (py1 - py0)
            parts.append(f'<rect x="{center+offset:.1f}" y="{py1-bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
        parts.append(text(center, py1 + 31, name, size=19, anchor="middle"))
    parts.append(text(px0 - 55, (py0 + py1) / 2, "rate (%)", size=19, anchor="middle", fill=MUTED, rotate=-90))
    parts.append(f'<rect x="{px0+8}" y="{y0+58}" width="18" height="18" fill="{BLUE}"/><rect x="{px0+130}" y="{y0+58}" width="18" height="18" fill="{ORANGE}"/>')
    parts.append(text(px0 + 34, y0 + 74, "success", size=18))
    parts.append(text(px0 + 156, y0 + 74, "truncation", size=18))

    x0, y0, width, height = 820, 145, 720, 660
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="{GRID}" stroke-width="2"/>')
    parts.append(text(x0 + 28, y0 + 40, "episode 最终位置分布", size=26, weight=700))
    px0, px1, py0 = x0 + 155, x0 + width - 35, y0 + 105
    row_h, bar_h = 65, 34
    zones = [
        ("start_platform Y<-1.5", "起步", "#94a3b8"),
        ("rough_hfield -1.5..1.5", "崎岖", "#eab308"),
        ("drop_lip 1.5..2.0", "落差", "#f97316"),
        ("ramp 2.0..6.83", "坡道", "#ef4444"),
        ("finish_platform >6.83", "终点", "#22c55e"),
    ]
    legend_x = px0
    for _, label, color in zones:
        parts.append(f'<rect x="{legend_x}" y="{y0+62}" width="16" height="16" fill="{color}"/>')
        parts.append(text(legend_x + 23, y0 + 77, label, size=17))
        legend_x += 88
    for index, name in enumerate(names):
        y = py0 + index * row_h
        parts.append(text(px0 - 20, y + 25, name, size=20, anchor="end", weight=700 if name in {"B0", "G5", "G7"} else 400))
        cursor = px0
        hist = records[name]["failure_zone_hist"]
        for key, _, color in zones:
            value = float(hist[key])
            segment = value * (px1 - px0)
            parts.append(f'<rect x="{cursor:.1f}" y="{y}" width="{segment:.1f}" height="{bar_h}" fill="{color}"/>')
            if value >= 0.08:
                parts.append(text(cursor + segment / 2, y + 25, f"{100*value:.0f}%", size=16, anchor="middle", fill="#ffffff", weight=700))
            cursor += segment
        parts.append(f'<rect x="{px0}" y="{y}" width="{px1-px0}" height="{bar_h}" fill="none" stroke="{GRID}" stroke-width="1"/>')
    parts.append(text((px0 + px1) / 2, y0 + height - 25, "episode fraction", size=19, anchor="middle", fill=MUTED))
    parts.append(text(60, 855, "关键变化：B0 的 78.12% episode 结束在坡道；A3/G3 坡道失败为 0；G5 的失败重新集中到崎岖区与坡道。", size=22, fill=INK))
    write_svg("eval_outcomes.svg", parts)


def main() -> None:
    plot_baseline_progress()
    plot_training_curves()
    plot_eval_outcomes()
    for path in sorted(OUT.glob("*.svg")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
