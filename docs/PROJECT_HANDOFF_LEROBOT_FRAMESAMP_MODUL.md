---
title: RoboMME FrameSamp+Modul：LeRobot 数据适配与训练链路交接
created: 2026-11-03T11:31:00+08:00
updated: 2026-11-08T12:28:13+08:00
tags: [RoboMME, MME-VLA, LeRobot, FrameSamp, Modulation, JAX, 训练复现, 闭环评测]
---

# RoboMME FrameSamp+Modul：LeRobot 数据适配与训练链路交接

本文供接手本项目的 agent 阅读，内容来自服务器实际代码、数据元信息、训练配置和运行记录。数据与训练链路初次核实于 **2026-11-03，Asia/Shanghai**；统一评测入口与完成结果更新于 **2026-11-08**。未另标日期的训练过程指标是 11 月 3 日快照。本文记录本次原生 RoboMME/MME-VLA 复现，不与其他 SimpleMemVLA、Qwen/DiT 或 S5000 实验混用。

**最新接手入口**：[统一评测执行文档](UNIFIED_ROBOMME_EVALUATION.md)。后续变体统一使用 `scripts/eval_robomme.py`，以 `--checkpoint`、`--seeds` 指定权重与评测 seed，部署路径在 `configs/eval/robomme_server.json`。最后三个 checkpoint 的九组合计已完成 7,200 条，成功率 **43.7778%**，比论文主表 44.51% 低 **0.7322 个百分点**。完整原始结果位于 `/root/robomme_framesamp_eval_20261105/REPORT.md`。

## 1. 先读这一页

- 项目目录：`/mnt/cpfs/heping/projects/robomme_policy_learning`。
- 上游仓库：<https://github.com/RoboMME/robomme_policy_learning>。
- 本次实现提交：`827c86535fb3383488574ea683552454886fc344`；基于上游 `ac846a9a4aec841e483b1dc5d620dfe426bde853`。
- 目标是论文主表的 **FrameSamp+Modul，512 个记忆 token**，主表报告总体成功率 44.51%。附录 Table 16 另有 1024-token 的 45.87%，本次配置对应前者，不能称为论文所有预算设置中的绝对最高分。
- 原始训练数据已在 `/mnt/cpfs/heping/data/public/robomme`：LeRobot v3、Parquet 数值数据、AV1 双相机视频，约 9.58 GB。**不需要为了这次训练下载完整的官方预处理大包。**
- 新建缓存位于 `/mnt/cpfs/heping/data/public/robomme_framesamp_cache`：约 50.46 GB，只保存数值索引、位置编码和 front 相机的 4×4 冻结视觉特征，不展开 RGB pickle。
- **训练配置的 `dataset_path` 指向新缓存目录，不是原始 LeRobot 根目录。** 缓存元信息保存原始数据的绝对路径；当前 front/wrist 图像仍回到原始视频解码。迁移时必须同时保留原始数据和缓存。
- 数据读取用 `LeRobotFrameSampDataset`，自动接入原项目变换和训练器。优化使用 JAX/Flax/Optax；PyTorch 在这里主要负责 DataLoader，不是 PyTorch 模型训练。
- 运行环境是项目独立 `.venv`，不是 Conda 的 `simplememvla-robomme`。
- 配置：`configs/train/robomme_framesamp_modul_repro.yaml`；入口：`scripts/train_framesamp_modul_repro.sh`。本文中未写绝对路径的仓库文件，均相对上述项目根目录。
- 已使用物理 GPU **0、1、2、3** 启动训练，全局 batch 64、FSDP 4、目标 80,000 步；单个主进程管理四卡，不使用 `torchrun`。
- 原训练已产出 60000、70000、79999 checkpoint，且这三个权重的完整评测已完成；**接手时先看日志和 PID，不要重复执行启动命令。** 也不要为了重启设置 `overwrite=true`。

## 2. 文件、数据与权重在哪里

| 用途 | 服务器绝对路径 |
|---|---|
| 项目及独立 Python 环境 | `/mnt/cpfs/heping/projects/robomme_policy_learning`、其下 `.venv/bin/python` |
| 已有 LeRobot v3 数据，保持只读使用 | `/mnt/cpfs/heping/data/public/robomme` |
| 新的 FrameSamp 缓存，也是训练 dataset_path | `/mnt/cpfs/heping/data/public/robomme_framesamp_cache` |
| 项目中的缓存快捷软链接 | `/mnt/cpfs/heping/projects/robomme_policy_learning/data/robomme_framesamp_cache` |
| 离线视觉编码器 | `/mnt/cpfs/heping/checkpoints/pi05_vision_encoder/siglip_params.pkl` |
| 完整 π0.5 JAX/Orbax 基座参数 | `/mnt/cpfs/heping/checkpoints/openpi-assets/checkpoints/pi05_base/params` |
| 作者原有、已纳入 Git 的归一化文件 | `/mnt/cpfs/heping/projects/robomme_policy_learning/assets/norm_stats.json` |
| 本 run 实际使用的归一化文件副本 | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/assets/mme_vla_suite/robomme/norm_stats.json` |
| 训练配置 | `/mnt/cpfs/heping/projects/robomme_policy_learning/configs/train/robomme_framesamp_modul_repro.yaml` |
| 标准输出和错误输出，文件名保留用户拼写 | `/mnt/cpfs/heping/projects/robomme_policy_learning/261102_train_framesample_modual.log` |
| checkpoint 根目录 | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/ckpts/mme_vla_suite/framesamp-modul-lerobot-repro-s42` |
| 编译缓存，位于 CPFS | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/jax_cache/framesamp-modul-lerobot-repro-s42` |
| 特征准备与真实数据检查记录 | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/framesamp_preparation` |
| 配置解析、启动身份与 PID 记录 | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/train_profiles` |
| 基座下载与校验记录 | `/mnt/cpfs/heping/projects/robomme_policy_learning/runs/weight_preparation` |
| 本交接文档在服务器的位置 | `/mnt/cpfs/heping/projects/robomme_policy_learning/docs/PROJECT_HANDOFF_LEROBOT_FRAMESAMP_MODUL.md` |

`.gitignore` 排除了 `/data/`、`/runs/` 和日志等运行产物。提交代码不代表数据、权重、归一化运行副本、checkpoint 已经随 Git 迁移。

### 2.1 两套权重的用途不同

**离线视觉编码器**用于提前生成历史记忆特征：

- 来源：`Yinpei/pi05_vision_encoder`，通过 HF-mirror 获取。
- 固定 revision：`59bd9ff4d58ea0638064bda851fd7d477ee9708c`。
- `siglip_params.pkl` SHA-256：`f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293`。

**完整 π0.5 基座**用于初始化 VLM、action expert、视觉主干等模型参数：

- 镜像：`beiyurobotics/pi05_base`，endpoint 为 `https://hf-mirror.com`。
- 固定 revision：`dacf0ff24ac1570e2872be389a4ce5f9a0ebd039`。
- 对应官方源：`gs://openpi-assets/checkpoints/pi05_base/params`。
- 20 个参数/元数据文件，共 **12,441,721,931 bytes**；下载后逐文件通过官方对象的 CRC32C 校验，有相应字段的文件还核对了 MD5 和 HF LFS SHA-256。
- 完整校验清单：`runs/weight_preparation/pi05_base_verification.json`；参数父目录另有 `source_verification.json`。
- 这是 JAX/Orbax/OCDBT 格式。其他项目缓存的 PyTorch `model.safetensors` 不能直接交给当前 `CheckpointWeightLoader`。
- 下载时使用的临时脚本不属于 Git 提交；可复用的是固定 revision、上述路径和持久化校验清单，不依赖 `/tmp` 脚本继续存在。

## 3. 数据集：源数据与训练样本不是同一个数量

本地数据 README 指向 Hugging Face 数据集 `pepijn223/robomme_data_lerobot_video`。本地 `meta/info.json` 的下载缓存元信息记录 revision 为 `92796abbc67db4ea6447e5031d04bf47aa3770bb`；该记录标识此文件的下载版本，不代替全数据内容校验。

| 属性 | 实际值及解释 |
|---|---|
| 格式 | `codebase_version = v3.0`，机器人 `panda` |
| split | `train: 0:1600`；本次使用全部 1600 个 episodes，没有另切 held-out/validation |
| 全部帧数 | 768,897，包含 conditioning demonstration 和机器人执行段 |
| 动作监督样本数 | 476,857，即 `is_demo=false` 的所有行 |
| 演示帧数 | 292,040，由全部帧减执行帧得到；这些帧可进入历史，但不是训练样本起点 |
| 相机 | `image` 为 front，`wrist_image` 为 wrist；两路均 256×256 RGB、10 fps |
| 视频 | AV1、YUV420P，保存在 MP4 容器中，训练时用 PyAV 解码 |
| 数值 | `state[8]`、`actions[8]` 均 float32，包含 7 个关节维度及 gripper 维度 |
| 任务数 | RoboMME 基准有 16 类任务；`info.total_tasks=116` 是语言指令条目数，不能解释成 116 类基准任务 |
| 原始数据占用 | 核对时 9,577,475,630 bytes，包含元数据与下载缓存等文件；约 9.58 GB，采用十进制单位 |

原始目录主要结构：

```text
/mnt/cpfs/heping/data/public/robomme/
├── meta/info.json
├── meta/tasks.parquet
├── meta/episodes/chunk-*/file-*.parquet
├── data/chunk-*/file-*.parquet
├── videos/image/chunk-*/file-*.mp4
├── videos/wrist_image/chunk-*/file-*.mp4
└── README.md
```

数值 Parquet 的关键字段为 `index`、`episode_index`、`step_idx`、`exec_start_idx`、`is_demo`、`task_index`、`state`、`actions`。源数据还包含 `epis_idx` 和多种 subgoal 字段；当前 FrameSamp 适配器只索引所需的数值列，没有丢弃或改写源文件。

### 3.1 三种索引必须区分

- `index`：整个 LeRobot 数据集的全局帧行号。
- `episode_index`：整个数据集的全局 episode 标识，也是 `episode_XXXXXX.npy` 的编号。
- `step_idx`：episode 内局部时间步，从 0 开始。历史采样与时间位置编码使用这个局部时间轴。

不要将这些编号等同于仿真评测中某个任务、某个 split 下的 source episode 编号。当前适配器不需要建立评测场景编号映射。

episode 元数据的 `dataset_from_index`、`dataset_to_index` 定义全局行号的半开区间 `[begin,end)`；各相机的 `chunk_index`、`file_index`、`from_timestamp` 定义其视频文件及片段位置。一个视频文件可以包含多个 episodes，不能假定一个 episode 对应一个独立 MP4。

## 4. 为什么增加约 50 GB 缓存，缓存中是什么

原始 AV1 视频压缩后很小，但训练需要随机访问历史的视觉特征。缓存提前计算被冻结的视觉主干输出，避免每个训练样本重新编码最多 32 张历史图像。它不缓存展开后的原始 RGB，也不重复保存 wrist 历史。

```text
/mnt/cpfs/heping/data/public/robomme_framesamp_cache/
├── meta/
│   ├── lerobot_framesamp.json       # 索引元数据、源路径、episodes、语言指令映射
│   ├── features_complete.json       # 全量准备完成标记
│   ├── position_spatial.npy         # 16 个空间位置，各 512 维
│   └── position_temporal.npy        # episode 局部时间位置，各 256 维
├── index/
│   ├── state.npy                    # [768897,8] float32
│   ├── actions.npy                  # [768897,8] float32
│   ├── episode_index.npy            # [768897] int32
│   ├── step_idx.npy                 # [768897] int32
│   ├── exec_start_idx.npy           # [768897] int32
│   ├── task_index.npy               # [768897] int32
│   ├── is_demo.npy                  # [768897] bool
│   └── sample_indices.npy           # [476857]，执行段对应的全局行号
└── features/
    ├── episode_000000.npy           # [T_episode,16,2048]
    ├── episode_000000.json          # 此文件生成身份与完成记录
    └── ...                         # 共 1600 组 npy/json
```

最占空间的是 `768897 × 16 × 2048 × 2 bytes` 的历史图像特征，单这一项约 50.39 GB；加索引、NPY 头和元数据后，缓存实测 **50,459,805,923 bytes**。因此 50 GB 来自未压缩的特征数组，不代表重复下载了 50 GB 视频。

### 4.1 uint16 文件实际保存 BF16 位模式

`features/*.npy` 的磁盘 dtype 是 `uint16`，内容是 BF16 浮点数的原始位模式，不是整数特征，也不是数值量化后的 uint16。读取时必须：

```python
features = np.load(path, mmap_mode="r")  # uint16
decoded = features[frame_indices].view(ml_dtypes.bfloat16).astype(np.float32)
```

不能直接 `features.astype(np.float32)`。适配器只将选中的最多 32 帧解码到 float32，不把全部 50 GB 缓存读进内存。

### 4.2 缓存与源数据的绑定

- 索引保存源数据的绝对路径和 `meta/info.json` SHA-256。
- 本次源信息 SHA-256：`726d4a2d0d88b8328776c8fab59614a66a15aa276928048d06b26684a2fbffd1`。
- 每个特征 JSON 保存视觉权重 SHA-256、源信息 SHA-256、形状、dtype、输入来源和 pooling 说明。
- 全量验证 1600 个 episode 的形状、dtype 和生成身份后写入 `meta/features_complete.json`；训练默认拒绝缺少此标记的缓存。
- 这些是元信息和结构检查，不代表每次启动都会对所有视频与 50 GB 特征做完整内容哈希。
- 缓存路径迁移后，不能只改训练 YAML；`lerobot_framesamp.json` 内 `source` 仍必须能定位原始视频和元信息。更换内容、权重或 pooling 时应使用独立缓存并重新验证，不手工伪造完成标记。

## 5. LeRobot 读取是如何接入原训练器的

核心文件：`src/mme_vla_suite/training/lerobot_framesamp.py`。原有 pickle 读取器仍保留。

### 5.1 索引阶段：prepare_index

`prepare_index(source, cache)` 做以下事情：

1. 检查 LeRobot v3.0，按全局 `index` 将数值列写入只读训练时可 mmap 的 NPY 文件。
2. 拒绝全局行号越界、重复、缺失；核对 episode 边界、连续 `step_idx`、固定的 `exec_start_idx` 及 demo mask。
3. 从 episode 元数据定位两路原始视频，检查文件存在。
4. 用 `np.flatnonzero(~is_demo)` 生成训练样本索引，因此 `len(dataset)=476857`。
5. 将 `meta/tasks.parquet` 转成 `task_index → language instruction` 映射，写入缓存元数据。

### 5.2 单个样本：LeRobotFrameSampDataset.__getitem__

给定训练样本下标 `i`：

1. `row = sample_indices[i]`，由 `row` 找到全局 episode 和局部 `step_idx=t`。
2. 当前观测：分别从原始 front/wrist 视频读取时刻 `t` 的 RGB 图像。
3. 历史记忆：从同一 episode 的 **0 到 t，包含演示段和当前帧** 采样 front 特征。
4. 历史不足 32 帧时全部使用并在右侧补零；超过时用 `np.linspace(0,t,32,dtype=np.int32)` 均匀选 32 帧。它不是只取最近 32 帧，也不是按 `streaming_obs_horizon=16` 抽训练样本。
5. 每帧 16 个 token：按时间、空间顺序展开到 512-token 预算，并生成逐 token 的有效 mask。
6. 时间位置使用 episode 局部采样时间，256 维；空间位置 512 维。拼接成每个 token 768 维的位置特征。
7. 监督动作读取从 `row` 开始的 20 帧；超出 episode 尾部时重复最后一个动作，绝不跨到下一个 episode。
8. prompt 由 `task_index` 取指令，并转为小写。

视频定位公式为 `episode_video_from_timestamp + step_idx / fps`，然后依据 PTS 对齐解码帧，允许误差为 `0.45/fps`。遇到时间戳不匹配、视频截断或跨 episode 读取会报错；不能直接把全局行号用作视频帧号。

例如全局 episode 0 共 96 帧，`exec_start_idx=48`：第一个训练样本是局部 t=48，而它的历史允许覆盖 t=0…48 的演示和执行信息。t=0…47 不作为动作监督样本起点。

单样本返回接口如下，保持原 RoboMME 变换器的键名：

| 字段 | 变换前形状/内容 |
|---|---|
| `image`、`wrist_image` | 各 `[256,256,3]`、uint8 RGB，由原视频实时解码 |
| `state` | `[8]`，当前状态，供动作差分及归一化 |
| `actions` | `[20,8]`，episode 内裁边并重复尾动作 |
| `prompt` | 语言任务指令 |
| `static_image_emb` | `[512,2048]`，从 BF16 缓存解码到 float32 |
| `static_pos_emb` | `[512,768]`，时间 256 + 空间 512 |
| `static_state_emb` | `[512,8]`，历史状态归一化后复制到每帧的 16 个 token |
| `static_mask` | `[512]` bool，标记真实历史 token |
| `simple_subgoal`、`grounded_subgoal` | 当前适配器返回空字符串，非 symbolic 路径随后移除 |
| `recur_*` | 当前适配器均返回 `None` |

`static_state_emb` 虽然保留接口，但此配置 `use_state_emb=false`，不进入记忆特征编码。源数据有 subgoal，并不意味着当前适配器已经支持 symbolic 模型训练；扩展该方向必须补充真实 subgoal 读取，不能沿用空字符串。

### 5.3 自动分派与多进程

`src/mme_vla_suite/training/dataloader.py:create_data_loader` 检查：

```python
if (Path(dataset_path) / "meta/lerobot_framesamp.json").is_file():
    dataset_class = LeRobotFrameSampDataset
else:
    dataset_class = RoboMMEDataset  # 原 pickle 读取器
```

因此，将 `dataset_path` 错填成原始 LeRobot 根目录会走到不匹配的旧读取路径；决定分支的是 marker，不是 `repo_id`。`repo_id=robomme` 是当前训练的数据/归一化身份，不是触发在线下载的 Hugging Face 数据集地址。

DataLoader 仍复用上游的 `TorchDataLoader`，采用 spawn worker、shuffle、固定 seed、`drop_last=True`，然后将 numpy batch 放到 JAX 分片。适配器通过 `__getstate__/__setstate__` 清除并重开 mmap 和 decoder，避免 spawn 时复制完整数值索引；每个 worker 最多保留 8 个 episode 特征 mmap、4 个视频 decoder。它不实例化官方 `LeRobotDataset`。

## 6. 从原始数据到一次参数更新

```text
已有 LeRobot Parquet + 双相机 AV1 视频
  ├─ prepare_index → 数值 mmap 索引 + 执行段 sample_indices
  └─ front 全帧解码 → 冻结视觉编码器 → 4×4 mean pooling → BF16 特征缓存
                                    ↓
LeRobotFrameSampDataset：当前双相机图像 + 32 帧历史特征 + 20 步动作
  → RepackTransform
  → RoboMMEInputs + DeltaActions
  → 作者 norm_stats 的 quantile normalization
  → ResizeImages / prompt tokenization / state、action 补到 32 维
  → Torch DataLoader → JAX 四卡分片 → HistAugObservation
  → HistoryPi0：当前图像/语言 prefix + noisy-action suffix + 独立记忆序列
  → action expert 的 memory cross-attention → MemoryRMSNorm 调制
  → flow-matching MSE → 反向传播 → clip + AdamW → EMA → 定期 checkpoint
```

### 6.1 离线历史特征生成

入口为 `scripts/prepare_lerobot_framesamp.sh`，先建索引，再默认用 GPU 0–3 按 `episode_index % num_shards` 分四个进程。单个分片运行 `scripts/build_lerobot_framesamp.py`：

1. 校验固定视觉权重 SHA-256。
2. 遍历 episode 的全部 front 帧，包括演示帧。
3. 解码 RGB，在 CPU 上从 uint8 映射到 `[-1,1]`，再经 JAX resize 到 224×224。
4. 冻结视觉编码器输出每帧 16×16=256 个、2048 维的特征 token。
5. 用 `pool_tokens_to_size(...,target_size=16)` 的默认 **mean pooling** 变成 4×4=16 token。
6. 将 BF16 位模式写成 uint16 NPY，先落临时文件，完成后替换并写 episode JSON。
7. 已完成且身份匹配的 episode 会跳过；全部完成后运行全量结构验证及真实数据检查，最后写入项目 `runs/framesamp_preparation/COMPLETE`。

已有缓存全量完成。接手时不需要重跑四卡特征生成，尤其不要与当前占用 0–3 的训练同时运行该脚本。

### 6.2 沿用原输入变换，不能重复处理动作

`RoboMMEDataConfig` 保留原项目的 Repack → RoboMMEInputs → DeltaActions → Normalize → model transforms 顺序。

- `DeltaActions` 对 action 的前 7 维减去**样本当前 state 的对应 7 维**，gripper 维度不做这一差分。20 个未来动作都减同一个当前 state，不是对相邻 action 做差。
- 然后使用作者的 `norm_stats.json` 做 quantile normalization；不是 LeRobot `meta/stats.json`，也没有在这次数据适配中重新估计统计量。
- 当前图像变为模型要求的两路 224×224；训练前向仍使用原图像预处理/增强路径，历史特征缓存不随每次训练图像增强重算。
- prompt 最大长度 64，`discrete_state_input=false`；state/action 补齐到模型内部 32 维。pi05 分支也不额外添加 pi0 的 state token。
- 不应因“不使用 proprioception”删除 dataset 的 state：动作差分和反归一化仍依赖它。
- 当前观测仍在线经过冻结视觉主干；预计算的是历史 front 特征，不能将训练描述成所有视觉前向均被移除。

作者原文件与运行副本已经比对一致，其 SHA-256 为 `f332bbd34ace1b6837cdc415b44f680896070a41564f9ce39016f1ebf99d1be5`。

### 6.3 模型与张量尺寸

以全局 batch B=64 为例：

| 模型输入/中间量 | 尺寸 |
|---|---|
| 两路当前图像 | 各 `[64,224,224,3]` |
| 当前视觉 token | 每路 256 个、2048 维；两路共 512 个 |
| 语言 token | `[64,64]`，有 padding mask |
| 历史原始视觉特征 | `[64,512,2048]` |
| 历史位置编码 | `[64,512,768]` |
| memory mask | `[64,512]` |
| memory feature encoder 输出 | `[64,512,1024]` |
| 归一化并补齐后的监督动作 | `[64,20,32]` |
| 预测 velocity | `[64,20,32]` |

记忆 feature encoder 先对 768 维位置特征做可训练线性投影和 SiLU，再与 2048 维图像特征拼成 2816 维，经可训练 Linear 映射到 1024 维。当前不拼接历史 state 特征。

**Modul 的实际含义**：记忆不拼入 VLM prefix 充当额外上下文，也不是第三个 memory expert。`history_pi0.py` 将 memory sequence 单独传给 action expert；各 Transformer block 在 action 分支的 FFN 之前，让 action hidden states 作为 query 对记忆做 cross-attention，再用结果生成 RMSNorm 的 scale/shift 调制。该 memory attention 使用 4 个 query heads、1 个 KV head、head_dim=256、width=1024。

### 6.4 损失与参数更新

这是动作 flow matching，不是 subgoal 文本生成或 next-token classification：

```python
noise = Normal(0, 1)
t = Beta(1.5, 1.0) * 0.999 + 0.001
x_t = t * noise + (1 - t) * actions
target_velocity = noise - actions
loss = mean((predicted_velocity - target_velocity) ** 2)
```

实现先对动作最后一维求均值，再由训练步骤对 batch/horizon 求均值；当前代码没有另加一个仅监督物理 8 维的 loss mask。训练器执行 global-norm clip、AdamW 更新，再以 0.999 更新 EMA。

冻结由 `get_freeze_filter()` 和 `trainable_filter` 决定：当前无 LoRA，只冻结视觉 `img` 参数；VLM expert、action expert、新增记忆编码/调制参数参与训练。`dtype=bfloat16` 指计算配置，不能理解为所有参数、梯度、Adam 状态均以 BF16 保存。

## 7. 实际训练配置与启动机制

| 参数 | 本 run 的值 |
|---|---|
| 基座与模型 | π0.5 base；`gemma_2b` VLM + `gemma_300m` action expert |
| memory YAML | `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml` |
| 记忆 | perceptual / frame_sampling / modulation；front 一路；budget=512；每帧 16 token；mean pooling |
| 模型动作 | horizon=20，内部 action_dim=32；物理动作为 8 维 |
| GPU / 并行 | `CUDA_VISIBLE_DEVICES=0,1,2,3`，FSDP 4，单个 JAX 主进程 |
| batch / worker | 全局 64，各卡 16；4 个 DataLoader workers，无梯度累积 |
| 训练量 / seed | 80,000 步；seed=42，使用全部执行段样本，shuffle、drop_last |
| LR | warmup 10,000 步，随后恒定 5e-5 |
| schedule 实现 | 复用 `CosineDecaySchedule`，将 peak_lr 与 decay_lr 都设为 5e-5；step 0 约 5e-9，保留上游非零预热起点 |
| AdamW | beta1=0.9、beta2=0.95、eps=1e-8、weight_decay=0 |
| 梯度裁剪 / EMA | global norm=1.0；EMA=0.999 |
| 观测状态 | `discrete_state_input=false`、`use_state_emb=false` |
| 记录 / 保存 | 每 100 步记录；每 10,000 步保存并保留这些周期 checkpoint；另保存最终步 |
| 输出配置名 / 实验名 | `mme_vla_suite` / `framesamp-modul-lerobot-repro-s42` |
| 新 run 约束 | `resume=false`、`overwrite=false`、`resum_ckpt_id=null` |
| W&B | 关闭，使用控制台日志；上游 entity 为作者账号，不能直接打开复用 |
| 显存 / CPU | JAX preallocate=true，fraction=0.90；OMP 8，OpenBLAS 1；这不是驱动级硬隔离或实测峰值声明 |

`scripts/run_training_profile.py` 在导入 JAX/模型模块之前设置运行环境，然后读取 YAML，用 dataclasses 替换 `get_config("mme_vla_suite")` 的对应字段。`expected_memory` 只是对真实 memory YAML 的断言，不是覆盖实现的另一套配置。

执行模式会检查数据/权重/归一化路径、数据计数、四卡、输出目录和编译缓存绑定；之后只调用一次：

```python
train.main(config, tentative_run=False)
```

这是有意绕开原 `scripts/train.py` 的 `__main__`：原入口先执行 tentative pass，再执行正式 main，可能造成额外更新和目录冲突。当前 `scripts/finetune_framesamp_lerobot.sh` 也已转发到统一 profile 的正式执行入口。

编译缓存软链接为 `/root/.cache/jax_framesamp-modul-lerobot-repro-s42`，实际目标是第 2 节所列 CPFS 目录。审阅模式不创建该链接。

### 7.1 原训练 run 的身份与历史快照（2026-11-03）

- 启动：北京时间 **2026-11-02 20:07:06**。
- PID：`3671858`，记录在 `runs/train_profiles/framesamp_modul_repro.pid`。PID 是历史身份，后续使用前需核对进程命令和启动记录，避免 PID 被复用。
- `framesamp_modul_repro_launch.json` 保存启动时间、commit、PID、日志路径、配置 SHA-256、GPU 列表、权重 revision 和首次步骤验证。
- 首步已验证：step 0 的 loss=0.0970、grad_norm=1.5684，数值有限。
- 2026-11-03 11:31:58 核对时进程存活，进度约 **19,200/80,000**；最近已记录的 step 19,200 loss=0.0040、grad_norm=0.0385。该数值是训练日志，不是任务成功率。
- 当时已生成 checkpoint `10000/`，内含 `params/`、`assets/`；当时尚未完成全程训练和闭环评测。后续已完成三个最终 checkpoint 的评测，见第 12 节。
- `framesamp_modul_repro_execution.json` 是进入训练前的参数/路径检查快照。其中 `training_started=false` 是报告生成时的值，不是持续更新的存活状态；是否在训练以进程、日志和 checkpoint 为准。
- 旧 `framesamp_modul_repro_review.json` 及旧审阅说明可能记载“基座缺失、未启动”，那是准备阶段快照；本次基座已经补齐并训练中。

### 7.2 Checkpoint 并不包含完整续训状态

当前正式训练调用 `openpi.training.checkpoints.save_state`。其 `train_state` 保存项被注释掉，只保存 inference 参数与归一化 assets；EMA 开启时保存的参数来自 EMA。

所以目前 checkpoint **不包含完整 Adam optimizer/train_state**。不能仅改 `resume=true` 就宣称实现了严格断点续训。文件中虽有其他保存函数，也不能据其存在认定当前 main 调用了它们。

保存目录名采用循环索引：本配置最后三个 checkpoint 为 `60000`、`70000`、`79999`，分别对应 60001、70001、80000 次参数更新。周期为 10000 且保留周期 checkpoint，不能把 `max_to_keep=1` 单独理解成只剩最后一个目录。

## 8. 接手常用命令

### 8.1 只读检查当前训练

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
git log -1 --oneline
git status --short
cat runs/train_profiles/framesamp_modul_repro_launch.json
ps -p "$(cat runs/train_profiles/framesamp_modul_repro.pid)" -o pid,ppid,etime,stat,args
tail -n 30 261102_train_framesample_modual.log
rg 'Step [0-9]+:' 261102_train_framesample_modual.log | tail -5
nvidia-smi
```

持续追日志：

```bash
tail -f /mnt/cpfs/heping/projects/robomme_policy_learning/261102_train_framesample_modual.log
```

### 8.2 只解析配置，不启动训练

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
bash scripts/train_framesamp_modul_repro.sh
```

审阅模式用 CPU，不创建大模型或执行优化步骤。当前 run 目录已存在，因此报告中的执行 blocker 会包含“新训练输出目录已存在”；这不表示运行中的训练异常。

### 8.3 数据验证

轻量行为回归：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  .venv/bin/python -m unittest discover -s tests -p 'test_lerobot_framesamp.py' -v
```

真实视频、变换和双 worker 检查；需原数据/缓存/归一化文件就位，结果另写文件以保留准备阶段原记录：

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
PYTHONPATH=src:packages/openpi-client/src \
OPENPI_DATA_HOME=/mnt/cpfs/heping/checkpoints \
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/check_lerobot_framesamp.py \
  --cache /mnt/cpfs/heping/data/public/robomme_framesamp_cache \
  --output /tmp/robomme_framesamp_data_check.json
```

已有结果：6 项行为测试通过；8 个固定真实样本经过原始变换后形状和数值检查通过；实际训练 DataLoader 的双 spawn worker、随机 batch=4 检查通过。准备记录在 `runs/framesamp_preparation/data_check.json`。本轮文档整理未重新生成特征或重启训练。

### 8.4 新机器或未来新 run 的启动步骤

此部分是复现流程，不是让接手 agent 立即再启动当前实验。

1. 使用本次提交，准备 Python 3.12 独立 `.venv`；依赖范围见 `requirements-framesamp.txt`，已验证安装快照见 `requirements-framesamp-lock.txt`。当前训练不依赖激活旧 Conda 环境。
2. 放好原始 LeRobot 数据、完整缓存和两套权重；用固定 revision/哈希验证身份。迁移时处理缓存元信息内的原始绝对路径。
3. 运行资产就位：`mkdir -p runs/assets/mme_vla_suite/robomme`，再将 `assets/norm_stats.json` 复制到该目录。保持作者统计量，不能混用其他项目 norm 文件。
4. 缓存已迁移则直接验证使用；确实需要新建缓存且 GPU 空闲时，才运行 `bash scripts/prepare_lerobot_framesamp.sh`。该脚本会使用四张 GPU。
5. 为新实验复制 YAML，更改实验名与编译缓存目录等部署路径，检查输出目录不存在。默认 runner 固定 GPU 0–3、四卡和 fresh-run 约束；换卡、续训等需要同时审阅 runner 的显式校验，不只是改 YAML。
6. 对新 YAML 先运行 CPU 解析和必要的数据检查，核对无意外 blocker。
7. 确认要执行新 run 后，使用下面的后台模式，将输出写入该 run 独立日志；不要截断当前实验日志。

```bash
# 仅作新 run 示例：先准备独立配置，确认没有重复进程和目录冲突。
cd /mnt/cpfs/heping/projects/robomme_policy_learning
nohup env PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_training_profile.py \
  --config /absolute/path/to/new_run.yaml --execute \
  > /absolute/path/to/new_run.log 2>&1 < /dev/null &
```

启动后应验证后台进程在 SSH 断开后仍存活、四卡映射正确、权重成功恢复、首个 loss/梯度有限且进度继续前进。仅看到 PID 或显存预分配不等于首个参数更新已经成功。

## 9. 依赖环境与复现边界

当前独立环境的已验证核心版本：Python 3.12.15、JAX 0.5.3、Flax 0.10.2、Optax 0.2.4、Torch 2.7.1、NumPy 1.26.4、Transformers 4.53.2、Orbax Checkpoint 0.11.13、PyAV 17.1.0、PyArrow 25.0.1。准备时 `pip check` 通过。

旧 `simplememvla-robomme` 环境为 Python 3.10 / Torch 2.4.1，准备时缺少 JAX，不能直接作为本项目已验证运行环境。`pyproject.toml` 的 Python 上界已从 `<=3.12` 修正为 `<3.13`，允许 Python 3.12 的 patch 版本。

必须保留以下差异与限制：

1. **Pooling 差异**：论文 Appendix B.2 写 max pooling，发布配置及实际生成/在线路径使用 mean。本次沿用 mean。只修改 YAML 不能把旧缓存变成 max；若要严格改为 max，需要同步改生成和推理路径、重建独立缓存并验证。
2. **视频压缩差异**：本次来自 LeRobot AV1 解码图像，不是原始 HDF5 RGB。准备阶段一个样本的 recomputed/官方 4×4 特征平均 token cosine similarity 约 0.999405；一个动作样本的 float32 差异约 1.1e-7。这些是抽样诊断，不是全数据或最终成绩等价证明。
3. **缓存范围**：仅支持 perceptual FrameSamp、一条 front 历史相机、每帧 16 token。没有生成 TokenDrop/循环记忆使用的 8×8 历史缓存，也没有实现 symbolic subgoal 的真实读取。当前生成器有冻结特征，并不等于所有论文变体已完成数据适配。
4. **配置不是成绩**：44.51% 是论文主表参考；本 run 完整九组合计实测为 43.7778%。训练 loss 下降不能代替闭环成功率。
5. **优化器差异**：论文 weight decay=0，本 profile 明确采用 0；上游默认 1e-10，注释曾提示设 0 可能 OOM。本次真实训练已持续跑到约 1.9 万步，但这不构成所有设备、batch 的容量保证。
6. **seed 与评测**：训练 seed=42 是固定复现选择；论文未披露精确训练/评测 seed 值。YAML `evaluation` 是计划信息，不由训练 main 自动执行。
7. **训练与评测数据**：本地 LeRobot `train` 不是闭环 test 场景库。实际闭环评测使用官方 test 16 任务×每任务 50 episode、max_steps=1300、预测 20 步执行前 16 步，60000/70000/79999 各评测 seeds **`[0,42,7]`**。这取代早期训练 YAML 中 `[7,31,42]` 的计划值，不能标为作者原值。仿真 metadata 来自官方 benchmark revision `856bc3a189d4172f3f47dbee4424d585f8d78db3`，不是重新读取 LeRobot train。
8. **迁移与恢复**：运行资产被 Git 忽略；checkpoint 不含完整优化器状态；旧审阅快照和启动前 JSON 不能用作实时状态。接手时以代码、元数据、进程、日志及实际 checkpoint 为准。

## 10. 源码阅读路线与关键定位

建议按下表顺序读，优先跟踪当前 FrameSamp+Modul 分支，不必先通读所有变体。行号对应提交 `827c865`，以后修改后按符号名重新定位。

| 文件，相对项目根目录 | 重点符号/行号 | 用途 |
|---|---|---|
| `configs/train/robomme_framesamp_modul_repro.yaml` | 全文件 | 具体数据、权重、四卡与超参数 |
| `scripts/run_training_profile.py` | `configure_environment`:16；`build_config`:38；`review_config`:82；`main`:141 | YAML 如何变成真实 TrainConfig，何时才允许 execute |
| `src/mme_vla_suite/training/lerobot_framesamp.py` | `prepare_index`:36；`VideoReader`:107；`LeRobotFrameSampDataset`:159；`__getitem__`:205 | 此次 LeRobot 适配的核心 |
| `scripts/build_lerobot_framesamp.py` | `main`:15；`encode`:69；`verify_cache`:125 | 历史特征如何生成、落盘、验证 |
| `scripts/prepare_lerobot_framesamp.sh` | 全文件 | 四卡分片准备链路 |
| `src/mme_vla_suite/training/dataloader.py` | `create_data_loader`:34，分派在 51–54 行 | 新旧 dataset 自动选择 |
| `src/mme_vla_suite/training/config.py` | `RoboMMEDataConfig`:333；`trainable_filter`:545 | repack、delta action、归一化配置、可训练参数 |
| `src/mme_vla_suite/policies/robomme_policy.py` | `RoboMMEInputs`:31 | 图像、状态、记忆接口映射 |
| `src/openpi/transforms.py` | `DeltaActions`:204；`AbsoluteActions`:226 | 动作差分的具体语义 |
| `src/openpi/training/data_loader.py` | `transform_dataset`:约 170；`TorchDataLoader`:384 | 变换顺序、spawn worker 与 JAX 分片 |
| `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml` | 全文件 | 当前执行的记忆架构配置 |
| `src/mme_vla_suite/models/representation/mem_encoder.py` | `FeatureEncoder`:8；`_add_pos_emb`:66 | 位置投影、拼接与 memory feature projection |
| `src/mme_vla_suite/models/integration/history_pi0.py` | `get_freeze_filter`:203；`embed_memory`:395；`embed_prefix`:418；`compute_loss`:557 | 冻结、当前观测、历史与动作 loss |
| `src/mme_vla_suite/models/integration/history_gemma.py` | `MemoryRMSNorm`:28；`MemoryAttention`:45；block 内约 176 | cross-attention 如何调制 action FFN |
| `scripts/train.py` | `init_train_state`:128；`train_step`:212；`main`:316 | 基座恢复、梯度、AdamW、EMA 和保存循环 |
| `src/openpi/training/checkpoints.py` | `save_state`:65；`_split_params`:145 | 实际保存 EMA 参数/assets，未保存完整 train_state |
| `scripts/check_lerobot_framesamp.py`、`tests/test_lerobot_framesamp.py` | `main`、6 个行为测试 | 真实样本、spawn batch 和边界行为验证 |
| `src/mme_vla_suite/shared/mem_buffer.py`、`src/mme_vla_suite/shared/data_utils.py` | 在线历史、`pool_tokens_to_size`:135 | 后续评测需要与训练缓存一致的历史/池化路径 |

关键文件 SHA-256：

```text
src/mme_vla_suite/training/lerobot_framesamp.py
  aef99ee8ec481cb189a5ede684ca57e72ffdabd9676a2e5fbff460cb308aa155
scripts/run_training_profile.py
  27682dc9a3191b2b6d7e92d7a47bdf836f548c14646a182b37a6aaec1323de73
configs/train/robomme_framesamp_modul_repro.yaml
  a5dfba1cae1491f8e8d462f4e9e174ac69e3b5c200edf6af543900935ef5478f
```

## 11. 验证范围、来源与复习

本文依据远程实际文件与现有运行记录核对，不以 README 设计意图代替执行行为。11 月 3 日初稿记录训练与数据检查；11 月 8 日更新纳入已完成的三个 checkpoint、7,200 条官方 test 闭环评测及统一入口。此次入口迁移没有重启训练、改变超参数或重新提取特征；通过旧结果独立副本复算及 CPU 行为检查验收。跨格式全数据逐帧等价、完整优化器状态续训以及其他变体的 GPU 闭环仍未验证。

外部来源：

- **RoboMME 开源训练仓库**：<https://github.com/RoboMME/robomme_policy_learning>。
- **论文文本**：本机 `/Users/liam/Projects/release/deepseek_harness_wd/GraphResearch/data/papers/RoboMME/paper.md`；重点是主表、Appendix B.2、Table 6，以及 Table 16 的预算消融。
- **已有 LeRobot 数据来源**：<https://huggingface.co/datasets/pepijn223/robomme_data_lerobot_video>；身份来自本地 README 与下载元信息。
- **作者视觉编码器**：<https://huggingface.co/Yinpei/pi05_vision_encoder>。
- **π0.5 JAX 镜像**：<https://huggingface.co/beiyurobotics/pi05_base>；原始官方对象前缀为 `gs://openpi-assets/checkpoints/pi05_base/params`，本次下载文件已与官方对象 CRC32C 比对。

复习/接手自检：

1. 为什么训练 `dataset_path` 必须指向缓存目录，而迁移时还必须保留原始 LeRobot 视频？
2. 为什么 `is_demo=true` 的帧不能提供动作监督，却必须允许进入 FrameSamp 历史？为什么 uint16 不能直接转 float32？
3. 为什么 EMA 参数 checkpoint、训练 loss 下降、完整断点续训和闭环任务成功率是四种不同的验证结果？

## 12. 统一评测与完成结果（2026-11-08）

- 入口：`scripts/eval_robomme.py run --checkpoint /absolute/path/to/ckpt --seeds 0 42 7 --gpus 0 1 2 3 --runtime-config configs/eval/robomme_server.json --output-dir /absolute/path/to/new_run`。
- 多 checkpoint 时重复 `--checkpoint`；数字目录按 iter 倒序，整个 checkpoint 完成后再进入下一个。每张 GPU 默认两个 stream，每个策略的推理 batch=1。
- 变体由 checkpoint 父目录 `history_config.txt` 和原项目配置恢复；不固定 FrameSamp/Modul/512 token。符号记忆必须显式提供 subgoal 来源和所需预测器依赖。
- `status`、`report`、`run --resume` 均通过同一入口；`--resume` 固定原 manifest，不接受覆盖 seed、卡号或并发数。完整示例、协议、结果字段及验证范围见 [统一评测文档](UNIFIED_ROBOMME_EVALUATION.md)。
- 模型使用项目 `.venv`；仿真使用旧 Conda `simplememvla-robomme`。此 Conda 可供仿真使用，不代表能运行本项目 JAX 训练。
- benchmark 已从易失的旧临时目录复制到 `/root/robomme-runtime/robomme_benchmark-856bc3a189d4172f3f47dbee4424d585f8d78db3`，部署 profile 指向这个持久副本；每次 run 会记录源码与 test metadata 哈希。
- 原始结果、manifest、SQLite 保持在 `/root/robomme_framesamp_eval_20261105`；不要把新版入口覆盖到旧临时脚本后声称历史运行用了新版。迁移没有再发起一轮 GPU 评测。

| Checkpoint | Seeds | 成功数 / 总数 | 成功率 |
|---|---|---:|---:|
| 79999 | 0 / 42 / 7 | 1054 / 2400 | 43.9167% |
| 70000 | 0 / 42 / 7 | 1054 / 2400 | 43.9167% |
| 60000 | 0 / 42 / 7 | 1044 / 2400 | 43.5000% |
| 九组平均 / 合计 | 三 checkpoint × 三 seed | 3152 / 7200 | **43.7778%** |

与论文主表 44.51% 的差值是 **−0.7322 个百分点**。此处汇总来自实际记录复算，完整逐任务结果见旧 `REPORT.md`。
