# FrameSamp+Modul：batch 减半对照配置

日期：2026-11-12，Asia/Shanghai。工作目录为 `/mnt/cpfs/heping/projects/robomme_policy_learning`。
原复现 YAML 保持原样。此新实验按用户指定使用物理 GPU 4–7 后台训练，运行状态以 `train.log` 和同目录 `launch_identity.json` 为准。

## 对照设置

新配置：[robomme_framesamp_modul_bs32.yaml](../configs/train/robomme_framesamp_modul_bs32.yaml)。
入口：[train_framesamp_modul_bs32.sh](../scripts/train_framesamp_modul_bs32.sh)，默认只审阅，显式传 `--execute` 才会训练。

| 项目 | 已完成的基线 | 新对照 |
|---|---|---|
| 实验名 | `framesamp-modul-lerobot-repro-s42` | `framesamp-modul-lerobot-bs32-s42` |
| 物理 GPU / FSDP | 0、1、2、3 / 4 | **4、5、6、7 / 4** |
| 全局 batch | 64 | **32** |
| 每卡 batch | 16 | **8** |
| 训练更新数 | 80,000 | **80,000** |
| batch × 更新数 | 5,120,000 | **2,560,000** |
| 训练 seed | 42 | 相同 |
| 学习率 | 10k warmup，再恒定 5e-5 | 相同，不按 batch 缩放 |
| 优化器 / EMA / 冻结范围 | 原配置 | 相同 |
| 模型 / 记忆 | FrameSamp+Modul，512 token | 相同 |
| LeRobot 数据、特征与基座参数 | 原输入 | 只读复用，相同 |
| 保存间隔 | 10,000 iter | 相同 |
| checkpoint 保留 | 万步节点全部保留 | **仅最近 3 个** |
| 新训练产物位置 | `/mnt/cpfs` | **`/mnt/cpfs-standard`** |

这是固定优化器更新次数、减少样本预算的实验。减半 batch 后训练累计处理的样本数也减半，不是相同样本预算的对照。SR 是否保持需要完成训练后闭环评测。

## 保存策略与步编号

新配置设置：

```yaml
batch_size: 32
num_train_steps: 80000
save_interval: 10000
checkpoint_max_to_keep: 3
keep_period: null
```

训练器使用 `range(0,80000)` 并在每个对应循环内更新参数。保存事件沿用原实现：
`10000、20000、30000、40000、50000、60000、70000、79999`。
旧节点随新 checkpoint 完成而滚动删除，正常结束只留下 **`60000、70000、79999`**。
`79999` 是完成 80,000 次更新后的最终目录，仍对应“8w”；没有额外增加一次更新来改目录名。

原 `keep_period=10000` 会保护所有万步节点，不能用于“只保留三个”。为此新增 `TrainConfig.checkpoint_max_to_keep`，
由训练器传给 Orbax `max_to_keep`。该字段默认仍为 1，原配置的周期保护和保存行为保持不变。

## cpfs-standard 上的新产物

| 用途 | 绝对路径 |
|---|---|
| checkpoint 根目录 | `/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/framesamp-modul-lerobot-bs32-s42` |
| JAX 编译缓存 | `/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/jax_cache/framesamp-modul-lerobot-bs32-s42` |
| 归一化资产 | `/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/assets/mme_vla_suite/robomme/norm_stats.json` |
| Tokenizer 缓存 | `/mnt/cpfs-standard/heping/checkpoints/big_vision/paligemma_tokenizer.model` |
| 审阅输出和验证记录 | `/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/profiles/framesamp-modul-lerobot-bs32-s42` |
| 建议训练日志 | `/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/profiles/framesamp-modul-lerobot-bs32-s42/train.log` |

归一化文件与 tokenizer 是原文件的校验一致副本。约 50 GB 的 LeRobot 特征缓存与 π0.5 基座参数继续从已有 `/mnt/cpfs` 路径读取，不重复生成或下载。
代码仍在原仓库；本配置把新训练权重、资产输出及编译缓存改到 `cpfs-standard`。

## 审阅和后续执行

以下默认审阅命令不会分配 GPU 或启动训练：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
PYTHONDONTWRITEBYTECODE=1 bash scripts/train_framesamp_modul_bs32.sh \
  --json-output /mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/profiles/framesamp-modul-lerobot-bs32-s42/review.json
```

审阅 JSON 应显示 `global_batch_size=32`、`samples_per_device=8`、`retained_checkpoint_ids=[60000,70000,79999]`，
以及 `execution_blockers=[]`。原训练目录不会被覆盖；checkpoint 目标目录需要保持为新的独立 run。

等 GPU 4–7 可用且决定开始新 run 时，用同一入口显式执行：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
PROFILE_DIR=/mnt/cpfs-standard/heping/projects/robomme_policy_learning/runs/profiles/framesamp-modul-lerobot-bs32-s42
mkdir -p "$PROFILE_DIR"
nohup env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 bash scripts/train_framesamp_modul_bs32.sh \
  --execute --json-output "$PROFILE_DIR/launch.json" \
  > "$PROFILE_DIR/train.log" 2>&1 < /dev/null &
```

## 速度与 SR 的比较方法

单步 batch 计算量降低不保证总训练时间缩短到一半。FSDP 通信、数据准备、编译、保存和存储 I/O 仍有开销。
此外本次输出存储也改变了，不能把总时长差异全部归因于 batch。
比较时分开记录启动/编译时间、预热后非保存区间的 step 时间，以及实际完整训练墙钟时间。

完成后仍走 [统一评测入口](EVAL_ROBOMME_SCRIPT.md)：这三个 checkpoint 各用 seed **`[0,42,7]`**，
官方 test 16 任务 × 每任务 50 条，九组合计 7,200 条，保持与已完成 batch-64 基线一致。
基线九组平均 SR 为 **43.7778%**，论文 FrameSamp+Modul 参考为 **44.51%**。
先逐 checkpoint、逐任务和总体对照基线，再看与论文差值；目前新配置的速度与 SR 都尚未测量。

## 验证记录

保留策略及 GPU 选择 CPU 检查共 **7 项测试**；新配置实际解析为 batch 32 / 每卡 8、80,000 步，
最终保留列表正确，`execution_blockers=[]`。这些配置和保存策略检查不执行训练。运行时四个物理 GPU 经 `CUDA_VISIBLE_DEVICES=4,5,6,7` 映射为进程内的逻辑设备 0–3。

配置审阅日志：`runs/profiles/framesamp-modul-lerobot-bs32-s42/review.log`，在上表的 `cpfs-standard` 项目目录下。
真实 Orbax 小型 CPU checkpoint 测试日志为同目录 `retention_tests.log`；测试命令：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=src:packages/openpi-client/src .venv/bin/python \
  -m unittest discover -s tests -p test_checkpoint_retention.py -v
```

测试检查实际目录的滚动删除、最终三点保留、原万步保护行为、配置中模型/优化器/学习率等保持一致。
