# RoboMME perceptual-memory v1 P0 on worker36021 / MUSA

本机适配工作区：`/data/heping/robomme_musa/repo`；分支 `codex/musa-percmem-v1`，基于 `8c9c891c0e37e6b6eeeff18630436bc7ace5c1a3`。原仓库、原始数据和共享 quicforge 环境保持不变。新增依赖仅放在 `/data/heping/robomme_musa/venv`（继承共享环境的 PyTorch/torch_musa）。

## 后端与模型

现有 JAX/CUDA 入口不能使用 MUSA。新增 PyTorch/MUSA P0 后端，保留原 JAX 参数路径和布局、模型宽度与层数、原 seed42 的内存模块初始化、双专家 Gemma、冻结 SigLIP、全历史 MemReader 和 adaRMS 接入。此后端只接受原 native/full-history teacher 配置；P1/P2/P3 仍使用原实现。

原全局 batch64 保留：4 卡 × microbatch2 × 累积8。保留 BF16 主干计算、FP32 可训练权重、冻结视觉 BF16 权重、历史特征 FP32 计算/FP16 保存、原 flow-matching loss、AdamW、全局梯度裁剪、warmup/constant 学习率、EMA，以及原 JAX CPU 增广与随机数。跨框架和硬件存在浮点舍入差异；实际误差以数值验证报告为准。

仅设置物理 `MUSA_VISIBLE_DEVICES=4,5,6,7` 和 `CUDA_VISIBLE_DEVICES=4,5,6,7`，四进程使用逻辑 musa0–3。production 对照只占用物理4。未对0–3的进程执行任何控制操作。

## 数据与资产

源：`/home/mccuseradmin/heping/data/datasets/yinchenghust/robomme_lerobot`。

独立视图 `/data/heping/robomme_musa/data/lerobot_view` 映射 `observation.state→state`、`action→actions`、`frame_index→step_idx`、`is_video_demo→is_demo`、双相机键 `image/wrist_image`。`exec_start_idx` 由经过全量验证的 demo 连续前缀推导；视频使用只读原文件的符号链接。

1600 episodes、768897 行、476857 execution 样本，20fps。严格按 stride10 提取 77562 个关键帧，单轨迹最多142，保留320帧 teacher padding。不能复用此前10fps服务器的154381关键帧缓存。

原适配器生成索引和缓存规划。归一化按全部 execution 样本、20步 episode-clipped horizon、前7维 DeltaActions 和原 RunningStats/quantile 规则重新计算。缓存路径 `/data/heping/robomme_musa/data/percmem_cache`；归一化 `/data/heping/robomme_musa/assets/norm/mme_vla_suite/robomme`。

基座来自公开镜像 `hairuoliu/pi05_base@171d7c8641a306698b2f82eea84325997de0f34f`，每个对象必须与官方 GCS 清单的 MD5 或 CRC32C 一致才可使用。导出保持参数叶子路径，不加载其他策略或随机基座。官方来源：<https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/params/&maxResults=1000>。

## 入口和验收

```bash
cd /data/heping/robomme_musa/repo
bash scripts/run_musa_percmem.sh export
bash scripts/run_musa_percmem.sh validate-cpu
bash scripts/run_musa_percmem.sh validate-production
bash scripts/run_musa_percmem.sh extract
bash scripts/run_musa_percmem.sh train --steps 3 --run-name musa_smoke_verified
```

原始数据视图、索引与 norm 已生成。`index` 和 `norm` 子命令可重新生成相应资产；不要在训练读取缓存时并行重建。

验证门禁：模块前向/梯度/优化器对照 → 真实权重完整模型和原历史编码器对照 → 全量 native cache 验收 → 三步四卡训练、reader 参数更新、冻结权重哈希、checkpoint 固定输入重载一致性。前一阶段失败会停止后续阶段。

各阶段证据和日志位于 `/data/heping/robomme_musa`：`numerical_cpu_validation.json`、`production_validation.json`、`assets/download_complete.json`、`assets/export_complete.json`、`pipeline.status.json`、`export.log`、`validate_production.log`、`extract.log`、`short_train.log`。训练结果在 profile 指定的 `runs/ckpts/percmem-v1-teacher-musa-s42/<run-name>/status.json`。

本文件为实施说明；只有训练 `status.json` 的 `stage=complete` 才表示指定短训练及重载检查通过。checkpoint 是本 MUSA 后端四卡本地分片格式，不能直接交给原 Orbax 策略入口。

`--steps N` 指本次执行的更新数，完整训练总预算仍为80000。可用 `--resume-from /绝对路径/step_000003` 和新的 `--run-name` 继续；恢复要求同一 profile、同一四卡分片布局、完整 checkpoint，恢复优化器、EMA、RNG，并跳过已消费的全局采样批次。周期保存保留原零起始 loop 的触发条件，目录名使用已完成更新数，避免恢复时学习率和随机数偏移一位。

## 2026-11-03 实测验收通过

20个基座对象，共12,441,721,931 bytes，全部匹配官方MD5/CRC32C；tokenizer匹配官方MD5。1600 episodes/77562 keyframes的native缓存全部校验通过。

33项CPU数值检查通过。真实预训练完整模型的FP32 velocity最大绝对误差2.98e-6；BF16相对L2误差0.004215、cosine0.999991。冻结历史视觉编码器、原增广、非零memory接入与连续三个epoch采样顺序对照通过。总参数3,386,067,729，可训练2,968,692,768，与原定义一致。跨框架浮点计算不要求逐位相同。

| 更新 | global mean loss | clip前 grad norm | reader final 相对本次初始值的L2变化 |
| --- | --- | --- | --- |
| 1 | 0.09806338 | 1.16797161 | 1.02345e-5 |
| 2 | 0.08669331 | 1.11107302 | 2.70987e-5 |
| 3 | 0.07732103 | 1.07210636 | 5.20271e-5 |
| 从step3恢复后继续到4 | 0.09005308 | 1.07520640 | 3.45073e-5 |

真实训练仍使用原warmup学习率，前三步依次约5e-9、1e-8、1.5e-8。两次运行均通过checkpoint保存、扰动参数后重载固定输入loss逐位一致、冻结视觉/scorer哈希不变；恢复运行从step3继续，并使用step4对应的采样和学习率。短测只验收执行与状态恢复，不代表策略收敛或环境任务成功率。

短训练结果：`/data/heping/robomme_musa/runs/ckpts/percmem-v1-teacher-musa-s42/musa_smoke_verified/status.json`；checkpoint为其下`step_000003`。恢复验证结果在同级`musa_resume_verified/status.json`，checkpoint为`step_000004`。两个输出目录均保留。CPU诊断中的5e-4学习率仅用于产生可观测的优化器更新，不作用于真实训练。

再次短测可直接运行（默认生成独立run名）：

```bash
cd /data/heping/robomme_musa/repo
bash scripts/run_musa_percmem.sh train --steps 3
```

已有CPU验证报告、字段视图、索引、norm和资产可直接复用。`scripts/musa/pipeline.py` 是本机准备阶段的协调器，等待已校验权重；首次部署需先按本说明准备环境、字段视图、索引/norm，并执行`validate-cpu`。不要把等待资产的后台进程状态误认为训练已完成。
