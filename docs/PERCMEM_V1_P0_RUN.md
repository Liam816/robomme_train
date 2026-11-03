# P0 全历史感知记忆实验运行记录

核实时间：2026-11-03T17:33:45+08:00（Asia/Shanghai）。基线提交 `827c86535fb3383488574ea683552454886fc344`，分支 `codex/pi05-perc-mem-v1`。本文件是提交前快照，实时状态以服务器日志为准。

## 配置与实际产物

- 配置：[teacher profile](../configs/train/robomme_percmem_v1_teacher_repro.yaml)；[记忆网络配置](../src/mme_vla_suite/models/config/robomme/perceptual-learned-adarms-v1-teacher.yaml)。
- 模型：pi05_base、gemma_2b + gemma_300m；全局 batch64、FSDP4、80k steps、seed42、warmup10k 后 LR5e-5、EMA0.999。物理 GPU4–7；GPU0–3 的原 FrameSamp+Modul run 不受影响。
- P0 不运行学习淘汰；读取截至当前时刻已发生的全部 2fps front 关键帧，max_frames320，reader 为8个可学习 query、宽512、2层。W4/Q28与scorer用于后续阶段。
- 缓存：`/mnt/cpfs/heping/data/public/robomme_percmem_v1_cache`；1600轨迹、154381关键帧、最多283帧/轨迹、FP16 8×8×2048。完成标记与完整schema/身份检查通过，不混读旧4×4缓存。
- 训练PID2223168，supervisor PID2185454；已连续完成105次迭代，最新段无异常或OOM。Step0 loss0.0935；Step100日志 loss0.0684、grad_norm0.5992。Step100值是steps1–100的均值，不是单个batch或验证集指标，不能据此宣称收敛或记忆有效性。
- 四卡显存约40.6GiB/卡，含90%预分配。batch之间有计算空档，当前吞吐不是持续满载；尚未进行独立数据管线性能剖析。
- checkpoint目录：`/mnt/cpfs/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/percmem-v1-teacher-lerobot-s42`。尚未到首个10000步checkpoint，尚未完成80k或闭环评测。

完整数字和身份见[运行快照](percmem_v1_p0_run.json)。主日志：

```text
/mnt/cpfs/heping/projects/robomme_policy_learning_worktrees/perc_mem_v1_20261103/261103_train_mem_p0.log
```

## 已修复的真实启动问题

1. **NNX静态graph不一致**：A_feat的局部identity-pad函数与A_cond每次新建的Xavier初始化器造成 `pjit out_shardings.model_def` metadata不同。改为模块级共享初始化函数/对象，保持维度、初始化数学定义不变。回归测试实际走eval_shape→FSDP→jitted初始化。
2. **首步之后的缓冲区释放断言**：首次修复后的run输出step0后，在 `BFCAllocator::DeallocateRawInternal` 触发 `h != kInvalidChunkHandle` 并退出。微型四卡原整state捐赠路径复现；相同5GiB数组单独传输/释放探针通过。现将冻结params/EMA作为独立不捐赠状态，只捐赠可训练params、对应EMA和优化器状态；JIT内外重组完整state，保持冻结EMA精确等于当前冻结参数。CPU与微型四卡连续三步均通过，生产run已持续越过该失败点。XLA内部指针错误未作二进制调试，不泛化为所有BFC错误的根因。

未升级共享venv、切换分配器或调整训练预算。旧integration分支保留原捐赠逻辑。前两次run没有checkpoint，失败元数据目录已保留后从base重新启动，没有伪称恢复优化器进度。旧异常留在同一log，查看当前错误应从最后一个RESTART标记开始。

## 验证与重现入口

提交前CPU运行21项感知记忆测试及6项FrameSamp回归，27项全部通过；另有微型四卡三步回归及生产四卡短程运行。具体范围见[实现说明](PERCMEM_V1_IMPLEMENTATION_REVIEW.md)和[验证报告](percmem_v1_validation.json)。P1标签可靠性、真实P2 scorer、P3蒸馏与闭环成绩仍待实际实验。

在具有相同数据/基座路径、空闲GPU和全新输出目录的环境中，先生成缓存，再显式启动训练：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning_worktrees/perc_mem_v1_20261103
PERCMEM_PYTHON=/mnt/cpfs/heping/projects/robomme_policy_learning/.venv/bin/python
"$PERCMEM_PYTHON" scripts/perc_mem/extract_keyframes.py \
  --execute --devices 4 --config configs/train/robomme_percmem_v1_teacher_repro.yaml
nohup "$PERCMEM_PYTHON" -u scripts/run_percmem_profile.py \
  --execute --config configs/train/robomme_percmem_v1_teacher_repro.yaml \
  >> 261103_train_mem_p0.log 2>&1 < /dev/null &
```

当前已有run正在训练，勿重复执行；runner也会拒绝已有checkpoint输出目录。精确DataLoader/worker续训尚未实现，profile保持resume=false/overwrite=false。

日志、PID、锁、后台supervisor及临时状态JSON是本机运行产物，不纳入本次代码提交；可重现入口为上面的版本化CLI与配置。
