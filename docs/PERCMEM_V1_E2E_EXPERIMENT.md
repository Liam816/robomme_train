# P0 对标：端到端学习式定长感知记忆

核实时间：2026-11-03T22:43:40+08:00。基线8c9c891，独立分支 `codex/pi05-perc-mem-e2e`。服务器目录 `/mnt/cpfs/heping/projects/robomme_policy_learning_worktrees/perc_mem_e2e_20261103`。本次已实现配置和梯度路径，未启动新的训练，原P0 worktree未修改。

[新训练配置](../configs/train/robomme_percmem_v1_e2e_repro.yaml)、[记忆配置](../src/mme_vla_suite/models/config/robomme/perceptual-learned-adarms-v1-e2e.yaml)、[P0参照](../configs/train/robomme_percmem_v1_teacher_repro.yaml)、[逐字段比较](percmem_v1_e2e_comparison.json)。

## 实验因素与初始化

| 项目 | P0 | E2E |
|---|---|---|
| 记忆 | 全部已发生的2fps关键帧 | W4最近窗口+Q28学习式淘汰，固定32帧 |
| Scorer | 存在但不调用、冻结 | 随机初始化，由动作FM loss训练 |
| Reader | 随机初始化并训练，末层零初始化 | 相同种子和初始化，全部训练 |
| A_feat/A_cond | identity-pad/非零Xavier并训练 | 相同 |
| 基座 | 官方pi05_base、VLM/action全参训练 | 相同 |
| 冻结 | 视觉和未使用scorer | 仅视觉 |
| Loss | Flow Matching | 同一FM，无ranking/蒸馏/aux loss |
| 数据 | 1600轨迹，native FP16 8×8缓存，作者归一化 | 相同，不依赖离线bank |
| 优化预算 | batch64/FSDP4/80k、seed42、warmup10k后LR5e-5 | 相同 |
| AdamW/clip/EMA | (.9,.95,eps1e-8,wd0)/1/.999 | 相同 |
| 图像/动作/语言 | front+wrist、H20、内部32维、执行16步、prompt64 | 相同 |

本组因素定义为“全历史记忆→学习式定长记忆”，容量限制与淘汰学习一起引入；P0不是同容量的非学习淘汰对照。profile共同训练设置完全一致，必要的profile/exp/history/cache标识分开。

不加载P0/P2记忆权重。loader只恢复官方基座并保留新模块初始化，遇到带记忆参数的checkpoint会拒绝。随机初始化沿用P0约定：reader末层仍为零、A_feat仍为identity-pad、归一化层保留标准初始化，避免新增初始化实验因素。相同seed小模型全部初始参数逐项一致；生产参数结构也相同。

总参数仍为3,386,067,729；可训练参数从2,968,692,768变为2,971,264,033，差值仅为原scorer的2,571,265参数。[参数审计](percmem_v1_e2e_parameters.json)。

## 可学习硬淘汰

原网络内部尺寸不变：store64、score/read各16 token，feature2560；scorer256宽/4头/2层，reader512宽/8 query/8头/2层，输出2048，再映射到1024维时间condition。π0.5的融合与采样机制不变。

按20Hz逻辑时钟、stride10写入2fps关键帧。强制保留W4；离开W的帧与Q28组成29候选，用原scorer硬argmin丢弃一帧，允许新候选直接被拒绝。队列决策与原writer/部署一致使用deterministic scorer，原dropout0.1结构配置保留但不在写入决策中启用。

前向是精确one-hot硬选择，不把帧混合成soft memory；反向使用softmax(-score/temperature)的straight-through代理梯度，temperature固定1，无Gumbel噪声、退火或其他loss。代理梯度有偏，不是硬argmin的真实导数。[实现](../src/mme_vla_suite/models/representation/perc_memory_e2e.py)。

每个原P0动作锚点，用当前A_feat/scorer按时间重放全部已发生关键帧，再只将最终32帧交给同一reader。训练输入仍为max_frames320的native历史，当前数据最多283帧；reader只读取定长bank。没有截断历史反向，也不复用P3预导出的成员表。

中间状态保存索引、时间及Q×N成员矩阵，避免每次写入保留完整视觉token bank。原scorer帧内编码每帧计算一次，原上下文层每次write仍重新计算；custom JVP前向使用精确gather，反向提供成员关系代理梯度；上下文打分用rematerialization，梯度覆盖全部写入。

由于保留P0 reader输出层零初始化，首步scorer梯度为0，reader输出层更新后才出现FM梯度。实际三步optimizer测试已验证scorer/reader权重更新；训练日志新增scorer_grad_norm、reader_grad_norm与feature_adapter_grad_norm便于验收。

## 部署与边界

训练以native2048重放；部署读取resolved history的副本，仅将E2E input_space改为projected。训练后的A_feat冻结在每帧写入时生成FP16 64×2560 token，在线只保留32帧；模型直接读取此bank，不再次淘汰幸存帧。在线打分池化显式用FP32，与训练重放一致。

CPU已验证硬writer的成员/时间/token一致，以及更新后的reader condition一致。35项E2E和原有回归通过，另1项最终在线精度测试通过；早期淘汰的后续梯度、scorer/reader更新、视觉及EMA冻结、空记忆/未来帧/候选自淘汰、同seed初始化和base-only加载均覆盖。[验证报告](percmem_v1_e2e_validation.json)。

[配置解析](percmem_v1_e2e_review.json)无缺失资产或fresh-run目录冲突，但它不检查显卡空闲。当前仍使用GPU4–7以对标P0，这些卡已有P0作业，本次没有启动E2E。生产四卡编译、峰值与吞吐尚待小步验收，不能把CPU测试写成GPU容量验收。训练仍需读完整历史以重放，不能由reader只读32帧推断训练总内存恒定或数据加载必然更快。

只审阅命令：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning_worktrees/perc_mem_e2e_20261103
PERCMEM_PYTHON=/mnt/cpfs/heping/projects/robomme_policy_learning/.venv/bin/python
"$PERCMEM_PYTHON" scripts/run_percmem_profile.py \
  --config configs/train/robomme_percmem_v1_e2e_repro.yaml
```

未来选择空闲设备并完成容量验收后，在同一入口显式加`--execute`才启动。checkpoint目录为 `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42`；日志/编译缓存/输出标识与P0分开，fresh-run保护不变。
