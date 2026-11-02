# FrameSamp+Modul 训练配置审阅

服务器仓库：`/mnt/cpfs/heping/projects/robomme_policy_learning`。本地审阅副本与服务器文件对应；实际运行在服务器进行。本次仅解析配置，没有创建大模型、执行优化步骤或启动训练。

配置入口：[robomme_framesamp_modul_repro.yaml](../configs/train/robomme_framesamp_modul_repro.yaml)。参数快照：[resolved_config.json](../runs/train_profiles/framesamp_modul_repro_review.json)。入口脚本：[run_training_profile.py](../scripts/run_training_profile.py)。

目标为论文 Table 3 的 FrameSamp+Modul，论文报告总体成功率 44.51%。本配置对齐 Appendix B.2 / Table 6 的训练超参数，数据读取按此前选定的 LeRobot 复用方案执行。最终分数仍需训练和论文协议评估，不能由配置解析保证。

| 项目 | 本次配置 |
|---|---|
| GPU | 固定物理 GPU 0、1、2、3 |
| 并行 | FSDP 4，单进程管理四卡，不额外套 torchrun |
| 全局 batch | 64；四卡各处理 16 个样本，无梯度累积 |
| 训练步数 | 80,000 |
| 学习率 | warmup 10,000 步，之后恒定 5e-5 |
| 优化器 | AdamW，β₁=0.9，β₂=0.95，ε=1e-8，weight decay=0 |
| 梯度裁剪 / EMA | global norm 1.0 / 0.999 |
| 基座 | π₀.₅ base，JAX/Orbax 参数目录 |
| 微调方式 | 冻结视觉编码器；VLM、action expert 和新增 memory 参数参与更新；不使用 LoRA |
| 计算 dtype | bfloat16；不声称所有参数、梯度与优化器状态都以 BF16 存储 |
| 当前观测 | front + wrist 两张图；每张 256 个视觉 token |
| 历史来源 | front 一路；全历史含演示段和当前帧 |
| 历史预算 | 最多均匀采样 32 帧，每帧 4×4=16 token，共 512 token；右侧补齐并使用 mask |
| 记忆融合 | modulation；原始图像特征 2048 维、位置编码 768 维，记忆输出宽度 1024 |
| Proprioception | 模型输入不使用；state 仍用于原有动作差分与归一化 |
| 动作 | 7 个关节 + gripper，模型内部补齐到 32 维；预测 chunk=20，评估执行前 16 步 |
| 数据 | 全部 1,600 条演示，768,897 帧；只监督 476,857 个执行段样本 |
| 归一化 | 作者提供的 robomme norm_stats，沿用已验证路径 |
| seed | 训练 seed=42，来自仓库默认值；论文没有列出训练 seed |
| 日志 / 保存 | 每 100 步记录；每 10,000 步保存；保留这些周期 checkpoint 和最终 checkpoint |
| 输出 | `runs/ckpts/mme_vla_suite/framesamp-modul-lerobot-repro-s42` |
| 显存分配 | 四卡各最多预分配 90%；没有实际训练，尚未验证完整训练峰值显存 |

学习率复用了仓库的 `CosineDecaySchedule`，将 peak_lr 和 decay_lr 都设为 5e-5，因此预热后保持常数。实际 CPU 求值：step 0 约 5e-9，step 9,999 约 4.9995e-5，step 10,000 / 40,000 / 79,999 均约 5e-5。保持仓库的非零 warmup 起点。

**请重点 review 下面的差异和取舍。**

1. **Pooling：开源实现与论文文字不一致。** 论文 Appendix B.2 写 max pooling；开源配置、已生成缓存和当前在线实现使用 mean。本配置保留 mean，复用已完成的缓存。若按论文文字改成 max，需修改实际池化路径、重新生成缓存，并核对在线推理；只把 YAML 的字段改成 max 不足以完成切换。
2. **图像：使用 LeRobot AV1 解码结果。** 不是原始 HDF5 RGB；一个时刻与官方 4×4 特征的平均 token 余弦相似度为 0.999405。这是抽样诊断，不能证明全数据等价。动作由 LeRobot float32 读取，已检查样本与原始 pickle 的差异约 1.1e-7。
3. **weight decay：按论文显式设 0。** 上游优化器默认是 1e-10，并有“设 0 可能 OOM”的注释；这条注释未在本次训练中验证。配置不会自动更改 batch、weight decay 或冻结策略。
4. **训练入口：直接调用一次正式 main。** 上游 `scripts/train.py` 的 `__main__` 会先做约 12 次更新的 tentative pass，再调用正式 main。本入口使用 `train.main(config, tentative_run=False)`，不经过这一双调用路径。
5. **W&B：本次关闭。** 上游 entity 硬编码为作者账号。本 profile 仅使用训练控制台日志；需要在线日志时先改成你的 entity，再启用。
6. **续训：当前只配置新训练。** 上游 checkpoint 保存 EMA 参数和 assets，没有保存完整 optimizer / train_state。`resume=false`、`overwrite=false`，已有实验目录会阻止执行；不能把目前的参数加载当成严格的优化器状态续训。

按当前训练循环，checkpoint 用循环索引命名，因此最后三个目录是 **60000、70000、79999**，对应 60001、70001、80000 次更新。本配置保持当前保存代码。论文只规定“最后三个 checkpoint”，没有在给定正文里列出这三个目录编号。

评估计划为 16 个任务 × 每任务 50 个 episode，最大 1,300 步，执行 chunk 前 16 步，最后 3 个 checkpoint × 3 个评估随机种子，共 9 次评估。计划种子为 `[7, 31, 42]`；论文没有公开确切种子值，因此这些是本次固定值，不标称作者原值。训练只规划 seed=42 的一次 run，不把评估的三个种子解释为三次训练。

**文件及执行方式**

服务器配置：`/mnt/cpfs/heping/projects/robomme_policy_learning/configs/train/robomme_framesamp_modul_repro.yaml`。

下面的命令只解析并显示配置，默认使用 CPU，不启动训练：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
bash scripts/train_framesamp_modul_repro.sh
```

未来确认配置且补齐基座后，下面的命令才会启动训练。本次没有执行它：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
bash scripts/train_framesamp_modul_repro.sh --execute
```

执行模式将按 YAML 设置 GPU 0–3，在确认必需路径齐备后，将上游硬编码的 `~/.cache/jax_<exp_name>` 通过新软链接重定向到 CPFS 的 `runs/jax_cache/`，避免编译缓存占用系统盘。审阅模式不创建该软链接。

配置审阅时唯一检测到的启动缺项是：`/mnt/cpfs/heping/checkpoints/openpi-assets/checkpoints/pi05_base/params`。该路径需要 JAX/Orbax π₀.₅ base 参数；现有 PyTorch safetensors 不能直接替代。

本次验证范围：YAML 到实际 `TrainConfig` 的转换、内存配置字段断言、冻结策略、学习率取值、四卡 batch/FSDP 配置、数据/归一化路径、保存编号与 shell/Python 语法。没有进行真实权重加载、优化步骤、显存峰值测量或成功率评估。此前已通过的 LeRobot 数据读取和双 worker 批次检查仍适用。

论文依据：[paper.md](/Users/liam/Projects/release/deepseek_harness_wd/GraphResearch/data/papers/RoboMME/paper.md:1364)，重点为 §5.1、Appendix B.2 和 Table 6。
