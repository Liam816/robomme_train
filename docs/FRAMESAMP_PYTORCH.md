# RoboMME FrameSamp + Modul 的 OpenPI PyTorch 复现

这份配置复现 `configs/train/robomme_framesamp_modul_repro.yaml` 对应的 π0.5 模型。模型前向、反向、AdamW、EMA 和四卡 FSDP 由 PyTorch 执行。RoboMME 提供原数据集、变换、历史缓冲区和官方闭环测试。为对齐随机输入，训练时在 CPU 上沿用 JAX 的图像增强、flow noise 和时间采样。

## 训练入口

以下命令在服务器上执行。两个仓库加载同一套训练代码。

```bash
REF=/mnt/cpfs/heping/projects/robomme_policy_learning
TARGET=/mnt/cpfs/heping/pi_wd/robomme_base_repro
PY=/mnt/cpfs/heping/pi_wd/openpi/.venv/bin/python
PROFILE=configs/train/robomme_framesamp_modul_pytorch.yaml
cd "$TARGET"
"$PY" scripts/launch_train.py --config "$PROFILE" --dry-run
"$PY" scripts/launch_train.py --config "$PROFILE" --check
```

`--dry-run` 打印完整配置，`--check` 读取真实样本，检查缓存、归一化和权重路径。下面两条命令分别用于测速和正式训练，独立执行。正式训练默认使用物理 GPU 4–7。切换参考仓库位置时，在目标仓库入口设置 `ROBOMME_REPO`。

```bash
# 稳态速度，16 步预热后计时 100 步
"$PY" scripts/launch_train.py --config "$PROFILE" --benchmark

# 80,000 步正式训练
"$PY" scripts/launch_train.py --config "$PROFILE" --execute

# 在参考仓库中执行同一配置的入口
cd "$REF"
"$PY" scripts/train_framesamp_pytorch.py --config "$PROFILE" --dry-run
```

YAML 通过 `reference_profile` 继承原始配置。也可传入原始 JAX YAML、导出的完整 PyTorch YAML，或通过重复的 `--set section.key=value` 覆盖字段。切换训练 seed 时，用相同 seed 的完整原训练 YAML 运行一次初始化命令，并把新文件写入 `pytorch.memory_init`。服务器已准备好 seed 42 的文件。

```bash
cd "$REF"
"$REF/.venv/bin/python" scripts/prepare_robomme_memory.py \
  --reference-repo "$REF" --config configs/train/robomme_framesamp_modul_repro.yaml \
  --output /path/to/new/framesamp-modul-init-s42.npz
```

| 配置 | 此复现采用的值 |
| --- | --- |
| 模型 | π0.5，Gemma 2B VLM 和 300M 动作专家，BF16，冻结视觉编码器及投影，训练 VLM、动作专家与记忆模块 |
| 输入 | 前视和腕部 RGB，历史使用前视缓存，均匀采样 32 帧，每帧 4×4 mean pooling，共 512 个记忆 token |
| 数据和动作 | 1,600 个训练 episode，768,897 帧，476,857 个可执行样本。沿用 quantile normalization 与关节 delta action，预测 20 步，每步 8 个物理维度，填充到 32 维 |
| 初始化与优化 | 同一份原始 Orbax π0.5 基座与 seed 42 记忆权重，global batch 64，每卡 16，AdamW，10,000 步 warmup 后 LR 5e-5，梯度裁剪 1，EMA 0.999 |
| checkpoint | 每 10,000 步保存与保留，末次保存编号 79,999。Orbax 中写入 EMA 参数和归一化，供 JAX 和 PyTorch 评测共用。此配置未保存 Adam 状态，也不支持从中途恢复训练 |
| 输出 | `/mnt/cpfs-standard/heping/projects/robomme_base_repro/runs/ckpts/mme_vla_suite/framesamp-modul-openpi-pytorch-s42` |

## 闭环评测

要单独衡量换训练后端的影响，对 PyTorch 生成的 checkpoint 使用原 JAX 评测器。要检查 PyTorch 端到端推理，同一组权重也可以由 PyTorch 评测。双方共用官方场景、动作执行、历史缓冲和 SQLite 计分。原全量运行实际使用 seeds `0,42,7`，覆盖 checkpoint `60000,70000,79999`。每组为 16 个任务 × 50 个 test episode，总计 7,200 回合。每次执行预测序列的前 16 步，单回合上限 1,300 步。

```bash
OUT=/mnt/cpfs-standard/heping/projects/robomme_base_repro/runs/eval
cd "$REF"
"$REF/.venv/bin/python" scripts/eval_framesamp_profile.py --config "$PROFILE" \
  --backend jax --output-dir "$OUT/framesamp-modul-s42-jax"

cd "$TARGET"
"$PY" scripts/eval_robomme_pytorch.py --config "$PROFILE" \
  --output-dir "$OUT/framesamp-modul-s42-pytorch"
```

上述命令需要对应的三个 checkpoint。用 `--dry-run` 可打印运行清单，用 `--checkpoint-root` 可指向另一组同保存策略的权重。参考仓库另有 `scripts/eval_framesamp_pytorch.py run`，可像 `scripts/eval_robomme.py run` 一样显式选择单个 checkpoint、任务、episode、seed 与 GPU。每次正式评测使用新的输出目录。

## 已验证的速度与数值

同一节点的物理 GPU 4–7 为四张 RTX 4090。两端顺序执行，从获取真实 batch 开始，到本步设备计算完成为止。两边均包含反向、优化器和 EMA，排除前 16 步与 checkpoint I/O。在原 JAX 环境复测的命令如下。

```bash
cd "$REF"
"$REF/.venv/bin/python" scripts/benchmark_framesamp_jax.py \
  --config configs/train/robomme_framesamp_modul_repro.yaml \
  --gpus 4 5 6 7 --warmup 16 --steps 100 \
  --cache /mnt/cpfs-standard/heping/projects/robomme_base_repro/runs/jax-benchmark-cache
```

| 同卡测量 | 平均秒/步 | 中位秒/步 | P90 秒/步 | 样本/秒 | 第 1–100 步平均 loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 JAX 0.5.3 | 3.166 | 3.156 | 4.084 | 20.22 | 0.0673801 |
| OpenPI PyTorch 2.7.1+cu126 | 2.253 | 2.225 | 2.292 | 28.41 | 0.0673793 |

该轮稳态吞吐比为 1.405×。旧 JAX 训练日志相近窗口约 2.62 秒/步，也反映了运行时负载和数据读取的影响。该比值只适用于以上配置和本次配对窗口。

4 个真实样本的 9 项输入和目标逐项一致。对原 JAX 的 79,999 checkpoint，用同一段真实历史、seed 7 和 10 步去噪比较最终 20×8 动作，7 维关节 RMSE 为 0.0001905 rad，夹爪符号分歧为 0/20。PyTorch 四卡正式保存路径已完成两步更新，产出了原评测器可读的 Orbax EMA。数据、参数映射、采样、checkpoint 和评测协议的回归测试共 35 项通过。

官方 BinFill test episode 0 使用场景 seed 540000、策略 seed 7。加载同一份原 79,999 权重，PyTorch 在 273 步成功，原 JAX 在 276 步成功，历史数据库为 275 步成功。把两步短训的 PyTorch EMA 交给原 JAX 评测器，该回合运行 387 步后判失败，完成了新权重的闭环读取和计分验证。

原 JAX 训练的闭环成功率为 3,152/7,200，即 43.78%。新训练的最终成功率需要同一协议的完整复现。配对测速原始日志和逐项验证保存在服务器 `/mnt/cpfs-standard/heping/projects/robomme_base_repro/validation/framesamp-port-s42`。
