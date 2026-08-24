# MotrixArena S1 · 结营作业

MotrixArena S1 导航任务的工程归档，包含环境与强化学习代码、实验配置、评估结果、代表性模型检查点，以及用于展示训练效果的图表。

## 目录

- `motrix_envs/`：仿真环境、机器人与场景资源
- `motrix_rl/`：强化学习训练与推理代码
- `scripts/`：训练、评估和报告图表脚本
- `artifacts/`：评估 JSON、代表性检查点与 TensorBoard 事件文件
- `media/`：训练曲线、评估结果与汇总图片

## 环境与运行

项目使用 `uv` 管理 Python 依赖：

```bash
uv sync
```

训练、评估和回放入口请查看 `scripts/`。代表性模型包括 B0、A1、A3、G3、G5、G6、G7，路径为 `artifacts/checkpoints/<实验编号>/`。

查看训练曲线：

```bash
uv run tensorboard --logdir artifacts/tensorboard
```
## License

遵循仓库中的 `LICENSE` 与 `NOTICE`。
