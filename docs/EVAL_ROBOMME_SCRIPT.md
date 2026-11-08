# eval_robomme.py 使用说明

更新日期：2026-11-08（Asia/Shanghai）。本文是统一评测脚本的命令参考；环境版本、完整执行链路与历史成绩见 [统一评测链路](UNIFIED_ROBOMME_EVALUATION.md)。

## 入口与运行条件

- 仓库：`/mnt/cpfs/heping/projects/robomme_policy_learning`，分支 `main`。
- 脚本：[scripts/eval_robomme.py](../scripts/eval_robomme.py)。从仓库根目录用 `.venv/bin/python` 启动。
- 本服务器部署配置：[configs/eval/robomme_server.json](../configs/eval/robomme_server.json)。模型使用项目 `.venv`，仿真使用 `simplememvla-robomme` Conda 环境，脚本自动启动各自进程。
- checkpoint 内需要有完整的 `params/`、`assets/robomme/norm_stats.json` 和已完成提交的 Orbax `_CHECKPOINT_METADATA`。模型变体从 checkpoint 父目录 `history_config.txt` 恢复；无该文件时按原项目的 baseline 路径加载。
- 默认使用物理 GPU 0–3。执行前检查所选 GPU 的当前占用；下面示例不会自动停止已有任务。

## 完整评测示例

下面的命令放在同一个 Bash 会话中执行。先定义公共参数并检查配置；`--dry-run` 不启动模型或仿真进程，也不创建评测输出目录。

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
CKPT_ROOT=/mnt/cpfs/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/framesamp-modul-lerobot-repro-s42
EVAL_DIR=/root/robomme-evaluations/framesamp-modul-new-run

EVAL_ARGS=(
  run
  --runtime-config configs/eval/robomme_server.json
  --checkpoint "$CKPT_ROOT/60000"
  --checkpoint "$CKPT_ROOT/70000"
  --checkpoint "$CKPT_ROOT/79999"
  --seeds 0 42 7
  --gpus 0 1 2 3
  --streams-per-gpu 2
  --variant framesamp-modul-lerobot-repro-s42
  --reference-percent 44.51
  --output-dir "$EVAL_DIR"
)

.venv/bin/python scripts/eval_robomme.py "${EVAL_ARGS[@]}" --dry-run
```

准备开始新一轮评测时，使用同一组参数后台启动。每次完整重测选择新的 `EVAL_DIR`；已有 run 的续跑见下文。

```bash
mkdir -p "$(dirname "$EVAL_DIR")"
nohup .venv/bin/python -u scripts/eval_robomme.py "${EVAL_ARGS[@]}" \
  > "$EVAL_DIR.supervisor.log" 2>&1 < /dev/null &
```

数字 checkpoint 标签自动按 `79999 → 70000 → 60000` 倒序执行，当前 checkpoint 的全部回合完成后才进入下一个。默认完整场景数为 `3 checkpoint × 3 seed × 16 task × 50 episode = 7200`。seed 保持命令行指定顺序。

评测其他变体时，使用它自己的 checkpoint 路径、名称与输出目录，保持相同入口和协议。例如单个 checkpoint、单个 seed：

```bash
.venv/bin/python scripts/eval_robomme.py run \
  --runtime-config configs/eval/robomme_server.json \
  --ckpt /absolute/path/to/another_variant/79999 \
  --seed 7 --gpus 0 1 2 3 --streams-per-gpu 2 \
  --variant another_variant \
  --output-dir /root/robomme-evaluations/another-variant-s7
```

## 参数参考

| 参数 | 默认值 / 要求 | 作用 |
|---|---|---|
| `--checkpoint` / `--ckpt` | 新 run 必填，可重复 | checkpoint 目录；支持 `LABEL=/absolute/path`。标签全为数字时倒序，否则按传参顺序 |
| `--seeds` / `--seed` | `0 42 7` | 一个或多个策略采样 seed；各回合开始时重置 RNG |
| `--output-dir` | 必填 | 独立 run 的输出根目录 |
| `--variant` | 首个 checkpoint 的父目录名 | 报告显示名称，不负责切换网络结构 |
| `--policy-config` | `mme_vla_suite` | 原项目训练配置名；history 变体由 checkpoint 元信息恢复 |
| `--runtime-config` | 不传则使用脚本内通用路径默认值 | 部署 JSON；本服务器传 `configs/eval/robomme_server.json` |
| `--gpus` | `0 1 2 3` | 使用的物理 GPU 编号 |
| `--streams-per-gpu` | `2` | 每卡独立 rollout/policy 流数，支持 1–8；按变体显存需求调整 |
| `--tasks` | 官方全部 16 任务 | 只运行指定任务，名称可从 `run --help` 查看 |
| `--episode-ids` | `0` 至 `49` | 每个任务使用的官方 test episode 编号 |
| `--max-steps` | `1300` | 单回合最大仿真步数 |
| `--execute-steps` | `16` | 每次规划执行动作块的前多少步；不能超过模型预测长度 |
| `--denoising-steps` | `10` | 动作采样的去噪步数 |
| `--vision-batch-size` | `16` | 单流历史帧编码批量 |
| `--reference-percent` | 不设置 | 在报告中计算与指定参照成功率的百分点差值 |
| `--max-attempts` | `3` | 单回合异常中断后的总尝试次数上限 |
| `--episode-timeout` | `900` 秒 | 单回合 watchdog，包含该回合的模型准备时间 |
| `--dry-run` | 关闭 | 验证输入并打印 manifest；不启动 worker |
| `--resume` | 关闭 | 从原 manifest 和 SQLite 恢复；只配合 `--output-dir`，可加 `--dry-run` |

部署 JSON 的路径字段也可以用 `--model-python`、`--sim-python`、`--benchmark-dir`、`--maniskill-dir`、`--asset-dir`、`--graphics-dir`、`--work-dir`、`--openpi-data-home` 覆盖。相对路径按仓库根目录解释。Hugging Face 默认使用 HF-mirror，实际 endpoint 记录在 manifest。

符号记忆还需要以下参数；感知记忆、循环记忆和 baseline 使用默认的 `none`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--subgoal-source` | `none` | 符号记忆必须显式选择 `oracle`、`qwenvl`、`memer` 或 `gemini` |
| `--subgoal-type` | 从模型配置读取 | 可指定 `simple_subgoal` / `grounded_subgoal` 做一致性校验 |
| `--subgoal-adapter` | 不设置 | QwenVL / MemER 必填；需先准备对应权重与依赖 |
| `--subgoal-keep-period` | `1` | 按执行步计的 subgoal 查询周期；在重新规划时检查，预测器还有自己的更新逻辑 |
| `--gemini-model` | `gemini-2.5-pro` | Gemini 预测器模型名；还需其原有 API 配置 |

oracle 与预测器属于不同评测条件，应分开记录。其他变体也应使用对应的论文参照，不要默认沿用 FrameSamp+Modul 的 44.51%。

## 进度、恢复和报告

以下命令继续使用前面定义的 `EVAL_DIR`：

```bash
# 只读查看回合进度
.venv/bin/python scripts/eval_robomme.py status --output-dir "$EVAL_DIR"

# 续跑中断的 run，读取原来的全部参数，不重复传 checkpoint / seed / GPU
.venv/bin/python scripts/eval_robomme.py run --output-dir "$EVAL_DIR" --resume

# 全部完成后重新生成最终报告
.venv/bin/python scripts/eval_robomme.py report --output-dir "$EVAL_DIR"
```

同一个输出目录有控制器锁；恢复时要求旧 worker 已退出，且脚本、关键源码、checkpoint 和环境源码身份校验通过。`--resume` 不接受覆盖 seed、GPU、并发数等参数。已有完成回合不会重跑，异常不会被直接计成任务失败；超过重试上限时状态为 `needs_attention`，需查看日志中的具体错误。

| 文件 | 用途 |
|---|---|
| `manifest.json` | 实际协议、路径、GPU 配置和身份哈希 |
| `episodes.sqlite3` | 唯一回合队列和逐回合结果，恢复依据 |
| `status.json` | 当前进度、活动回合、worker PID 和失败样例 |
| `CHECKPOINTS.md` / `checkpoint_results.json` | 每完成一个 checkpoint 更新一次的逐任务对比 |
| `REPORT.md` / `results.json` | 全部完成后的逐任务、逐 seed、总体成功率及可选参照差值 |
| `episodes.jsonl` | 最终逐回合结果导出 |
| `gpu*-stream*-worker.log` / `gpu*-stream*-policy.log` | 仿真和模型各自的日志 |

## 协议和使用边界

- 完整协议使用官方 `test` 16 任务、每任务 50 条，环境 seed 由官方场景 metadata 决定；命令行 seed 控制策略采样。初始演示进入在线记忆。
- 默认每个模型推理 batch size 为 **1**；四卡各两流代表八个独立 worker。`--vision-batch-size` 是历史帧批量，不是并行环境数量。
- 当前 FrameSamp+Modul 模型预测 20 步、执行前 16 步；入口从实际模型读取预测长度。记忆预算和集成方式沿用 checkpoint 配置。
- `--tasks BinFill --episode-ids 0` 等子集可用于冒烟检查，应使用独立输出目录。子集分数不能作为完整官方 test 成绩。
- 2026-11-05 已完成评测位于 `/root/robomme_framesamp_eval_20261105`。旧 manifest 支持新版 `status` / `report`，不支持新版 `--resume`；重生成旧报告前可先复制其 manifest 和 SQLite 到新目录。
- 入口迁移已通过 17 项 CPU 测试、Ruff、真实 checkpoint dry-run，以及旧 7200 条结果的逐 seed / 逐任务复算。新版 GPU 实跑与其他变体依赖需单独验收；此前已完成成绩见 [统一评测链路](UNIFIED_ROBOMME_EVALUATION.md)。

查看完整命令帮助：

```bash
.venv/bin/python scripts/eval_robomme.py --help
.venv/bin/python scripts/eval_robomme.py run --help
```
