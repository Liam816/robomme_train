# RoboMME 统一评测入口

核实日期：2026-11-08（Asia/Shanghai）。服务器仓库根目录为
`/mnt/cpfs/heping/projects/robomme_policy_learning`。以下仓库路径均相对此目录。

可直接复制的启动命令与完整参数表见 [评测脚本使用说明](EVAL_ROBOMME_SCRIPT.md)。

## 入口与适用范围

统一使用 `scripts/eval_robomme.py`，通过 `--checkpoint`（别名 `--ckpt`）和
`--seeds`（别名 `--seed`）传入权重及评测随机种子。调度器、模型加载、仿真回合与
汇总入口均复用同一套代码。每个变体使用独立 `--output-dir`。

模型加载调用原项目 `create_trained_policy(get_config(...), checkpoint_dir)`，从 checkpoint
父目录的 `history_config.txt` 读取实际配置，归一化统计来自 checkpoint 内的
`assets/robomme/norm_stats.json`，动作预测长度来自恢复模型的 `action_horizon`。
入口没有固定 FrameSamp、Modulation、512-token 预算或三 checkpoint / 三 seed 的数量。
变体的网络、记忆采样、池化和归一化仍由原项目实现。

- π0.5 baseline：父目录没有 `history_config.txt`，原策略以无记忆方式加载。
- 感知记忆 FrameSamp / TokenDrop，以及循环记忆 TTT / RMT：读取对应 YAML，默认 `--subgoal-source none`。
- 符号记忆：同一入口，但还须显式选择 `--subgoal-source oracle|qwenvl|memer|gemini`。
  实际 simple/grounded 类型从 checkpoint 配置读取，可用 `--subgoal-type` 加一致性检查。
  oracle 与预测器是不同实验条件，不能混成一个论文成绩。QwenVL/MemER 还需
  `--subgoal-adapter /absolute/path/to/adapter`，并在 simulator 环境准备其原有依赖与权重。
  Gemini 还需其原有 API 配置，可用 `--gemini-model` 指定模型。

上述是入口覆盖的代码分支；本次迁移以此前 FrameSamp+Modul 为基准进行 CPU 行为测试与
7,200 条记录复算。其他变体的权重、subgoal 依赖与 GPU 闭环运行需各自验收。

## 服务器环境与资源

部署路径单独放在 `configs/eval/robomme_server.json`，可复制修改，也可用同名命令行选项覆盖。
相对路径按仓库根目录解释。此 JSON 只包含部署路径；评测条件保存在每次运行的 manifest。

| 用途 | 本服务器路径 / 版本 |
|---|---|
| 模型进程 | 仓库 `.venv/bin/python`；Python 3.12、JAX 0.5.3、Flax 0.10.2 |
| 仿真进程 | `/mnt/cpfs/heping/miniconda3/envs/simplememvla-robomme/bin/python`；Python 3.10、Torch 2.4.1、SAPIEN 3.0.3 |
| RoboMME benchmark | `/root/robomme-runtime/robomme_benchmark-856bc3a189d4172f3f47dbee4424d585f8d78db3` |
| ManiSkill 源码 | `/mnt/cpfs/heping/SimpleMemVLA/third_party/ManiSkill`；revision `07be6fbc66350ddca200abfb0a11b692f078f7fd` |
| 仿真资产 | `/mnt/cpfs/heping/SimpleMemVLA-data/assets/maniskill` |
| EGL / Vulkan 配置 | `/mnt/cpfs/heping/SimpleMemVLA-data/cache/graphics` |
| Tokenizer 等 OpenPI 缓存 | `/mnt/cpfs/heping/checkpoints`，经 `OPENPI_DATA_HOME` 传给模型进程 |
| 临时 socket / 编译缓存 | `/dev/shm/robomme_eval`；重启后可重新创建 |

benchmark 从已完成评测使用的同一 revision 复制到 `/root/robomme-runtime`，避免依赖带日期的
`/dev/shm/robomme_framesamp_eval_20261105`。资产、权重、运行结果不进入 Git。
新服务器也可初始化仓库 `third_party/robomme_benchmark` 并修改 profile，安装同版本仿真环境。
旧 Conda 环境负责仿真；模型由项目 `.venv` 运行，二者通过带认证的本地 Unix socket 交换
msgpack 数据，不需要另外启动 WebSocket 服务。
Hugging Face 默认继续使用 `https://hf-mirror.com`，其 endpoint 写入 manifest 并传给 worker；
如启动前显式设置 `HF_ENDPOINT`，则记录并使用该值。

## 启动命令

先检查 `nvidia-smi`，待所选卡可用再执行实际评测。迁移验收时物理 GPU 0–3 正被其他训练使用，
本次没有启动新的 GPU 回合。`--dry-run` 检查部署文件、checkpoint、官方 metadata 并打印
解析后的协议，不创建输出目录、不启动 GPU 进程。

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
CKPT_ROOT=/mnt/cpfs/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/framesamp-modul-lerobot-repro-s42
.venv/bin/python scripts/eval_robomme.py run \
  --runtime-config configs/eval/robomme_server.json \
  --checkpoint "$CKPT_ROOT/60000" \
  --checkpoint "$CKPT_ROOT/70000" \
  --checkpoint "$CKPT_ROOT/79999" \
  --seeds 0 42 7 --gpus 0 1 2 3 --streams-per-gpu 2 \
  --variant framesamp-modul-lerobot-repro-s42 \
  --reference-percent 44.51 \
  --output-dir /root/robomme-evaluations/framesamp-modul-recheck \
  --dry-run
```

确认实际运行时去掉 `--dry-run`。后台执行可用同一命令加 `nohup` 并重定向输出：

```bash
nohup .venv/bin/python -u scripts/eval_robomme.py run \
  --runtime-config configs/eval/robomme_server.json \
  --checkpoint "$CKPT_ROOT/60000" --checkpoint "$CKPT_ROOT/70000" --checkpoint "$CKPT_ROOT/79999" \
  --seeds 0 42 7 --gpus 0 1 2 3 --streams-per-gpu 2 \
  --variant framesamp-modul-lerobot-repro-s42 --reference-percent 44.51 \
  --output-dir /root/robomme-evaluations/framesamp-modul-recheck \
  > /root/robomme-evaluations-framesamp-modul-recheck.log 2>&1 < /dev/null &
```

后续变体传它自己的路径、seed、名称和独立输出目录即可：

```bash
.venv/bin/python scripts/eval_robomme.py run \
  --runtime-config configs/eval/robomme_server.json \
  --checkpoint /absolute/path/to/another_variant/79999 \
  --seed 7 --gpus 0 1 2 3 --streams-per-gpu 2 \
  --variant another_variant --output-dir /root/robomme-evaluations/another-variant-s7
```

数字 checkpoint 标签自动按 iter 倒序评测，当前 checkpoint 全部完成后才开始下一个。
可重复 `--checkpoint LABEL=/absolute/path` 为非数字目录命名；只要包含非数字标签，
就保持传参顺序。seeds 保持指定顺序。支持任意数量 checkpoint 和 seed，总条数动态计算。
变体应分开启动；同一命令的 `--policy-config`、subgoal 来源和执行协议作用于全部 checkpoint。

## 默认协议与执行链路

```text
run / CLI 参数
  → manifest.json + SQLite 唯一 episode 队列
  → 每张 GPU 2 个独立 rollout stream
    → Conda simulator worker：官方 EnvRunner / test metadata / BenchmarkEnvBuilder
    ↔ 项目 .venv policy worker：原 create_trained_policy / add_buffer / infer
  → 每条回合写 SQLite
  → checkpoint_results.json / CHECKPOINTS.md
  → results.json / REPORT.md / episodes.jsonl
```

默认使用官方 **test 16 任务 × 每任务 episode IDs 0–49**，`joint_angle` 动作空间，
环境 seed 由该任务的官方 metadata 决定；`--seed(s)` 控制策略采样 RNG，不能改变场景集。
每回合 reset policy 并重新设定该 seed。初始 conditioning demonstration 进入在线记忆。
当前 front/wrist 图像、8 维状态和 prompt 通过原变换交给策略。

默认 `--max-steps 1300 --execute-steps 16 --denoising-steps 10 --vision-batch-size 16`。
本次 FrameSamp 权重预测 20 个动作，每次只执行前 16 个，再收集历史重新规划。
演示视觉编码分块并补齐到 16 帧，编码后去掉补齐帧；真实时间顺序不变。
原发布路径使用 mean pooling，此入口继续调用原实现。

**策略推理 batch size 始终为 1**。`--streams-per-gpu 2` 是每卡两个独立模型/环境进程，
四卡合计八路；`--vision-batch-size` 是单流历史帧编码批量，二者都不是环境 batch size。
CUDA_VISIBLE_DEVICES 在各 worker 中限制到一张物理卡，renderer 用该进程的逻辑 `cuda:0`。
baseline 或其他集成/预算更大的变体显存需求可能不同，可调整 stream 数后创建新 run。

`--tasks`、`--episode-ids` 可缩小为冒烟检查，例如 `--tasks BinFill --episode-ids 0`。
这类结果按实际范围计数，不能作为完整 16×50 协议的成绩。`--reference-percent` 默认不设置；
44.51 只用于本次 FrameSamp+Modul 对照，其他变体需要选择它对应的论文参照。

## 进度、恢复与统计

```bash
.venv/bin/python scripts/eval_robomme.py status --output-dir /absolute/path/to/run
.venv/bin/python scripts/eval_robomme.py report --output-dir /absolute/path/to/run
.venv/bin/python scripts/eval_robomme.py run --output-dir /absolute/path/to/run --resume
```

`status` 只读取进度。`report` 要求覆盖完整 manifest 且所有行完成，重新生成汇总。
正常运行每完成一个 checkpoint 就更新其逐任务表；全部完成时自动生成最终报告。
总体成功率为 success / 全部场景，fail 与 timeout 均记失败；基础设施错误不会被当成任务失败。
完整各组等量，因此各 checkpoint×seed 成功率的算术平均等于所有回合池化成功率。
JSON 还保存各任务、各 checkpoint、各 seed 的成功数/总数与组合间样本标准差。

SQLite 主键为 `(checkpoint, seed, task, episode)`。意外中断仅重新排队未完成回合，
每回合最多 `--max-attempts 3` 次；默认单回合 watchdog 为 `--episode-timeout 900` 秒。
持久失败会使状态成为 `needs_attention` 并非零退出，查看 worker/policy 日志及
`status.json` 的 failed_examples。SIGTERM / Ctrl-C 会清理自身 worker 进程组。

`--resume` 使用原 manifest，拒绝 seed、GPU、并发数、checkpoint 等覆盖，防止静默混合实验条件；
检查入口、关键源码、history YAML、checkpoint 元数据、norm stats、benchmark 与 ManiSkill 源文件身份。
同一输出目录有控制器锁，先前 worker 未退出时也拒绝恢复。更换协议、部署或代码请新建目录。
checkpoint 参数数据以路径与 Orbax 元数据标识，没有逐块哈希全部大权重；运行期间应保持权重只读。

| 产物 | 内容 |
|---|---|
| `manifest.json` | 本次实际路径、协议、变体、GPU/stream、源文件/metadata 哈希 |
| `episodes.sqlite3` | 逐回合状态、尝试次数、env_seed、结果、步数和异常 |
| `status.json` | 进度、当前 checkpoint、活跃回合、worker PID |
| `checkpoint_results.json` / `CHECKPOINTS.md` | 所有已经完整完成的 checkpoint 对比 |
| `results.json` / `REPORT.md` | 全矩阵统计、逐任务表、逐 seed 表与可选参照差值 |
| `episodes.jsonl` | 最终逐回合导出 |
| `gpu*-stream*-worker.log` / `gpu*-stream*-policy.log` | 两个环境各自的输出 |
| `scheduler_history.jsonl` / `supervisor.pid` | 启动/恢复历史与主进程 PID |

## 与已完成评测的关系及验证

原始 run `/root/robomme_framesamp_eval_20261105` 已完成，保留其原脚本、manifest、SQLite
和报告。此次没有重新启动该 run，也没有改写这些原始文件。
统一入口支持其旧 manifest 的 `status` / `report` 读取；旧 run 不支持接入新版 `--resume`。
若要避免重写旧报告，先复制其 manifest 和 SQLite 到独立目录再执行 `report`。

| Checkpoint | 三 seed 成功数 / 回合数 | 成功率 |
|---|---:|---:|
| 79999 | 1054 / 2400 | 43.9167% |
| 70000 | 1054 / 2400 | 43.9167% |
| 60000 | 1044 / 2400 | 43.5000% |
| 九组合计 | 3152 / 7200 | 43.7778% |

原 seed 为 `[0,42,7]`，九组平均比论文主表 44.51% 低 **0.7322 个百分点**。
最初 60000/seed0 的 800 条为单流，之后八组使用两流；核心推理、回合和场景协议一致。
新版数值 checkpoint 默认倒序只改变调度顺序，不回填旧 run 的历史执行顺序。

CPU 验证命令：

迁移验收：17 项 CPU 协议测试与 Ruff 静态检查通过；实际三个 checkpoint 的 dry-run
确认 7,200 条矩阵、倒序、全部官方 metadata 哈希和恢复身份校验；旧 Conda Python 3.10
可以导入新版入口、Null/Oracle predictor，并解析官方 test 场景，无需加载可选 VLM 依赖。

```bash
.venv/bin/python -m unittest discover -s tests -p test_eval_robomme.py -v
```

测试覆盖动态矩阵、倒序与 checkpoint 屏障、重复/遗漏回合、环境 seed 一致性、
动作块/初始演示传递、重试上限、恢复参数拒绝和无 GPU 的调度汇总。
另外用旧 SQLite 的独立副本复算全部 7,200 条，并与原结果逐项核对。
此验证不等于新版已实际完成一次 GPU 闭环；所选 GPU 0–3 正在训练，未为迁移而抢占资源。
