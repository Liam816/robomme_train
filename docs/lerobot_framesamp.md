# RoboMME FrameSamp training from existing LeRobot v3 data

The LeRobot dataset stays at `/mnt/cpfs/heping/data/public/robomme`. Its 1,600 episodes contain 768,897 frames, including 476,857 execution frames. This adapter reads the existing AV1 videos with PyAV and indexes numeric Parquet columns. It does not create per-frame RGB pickle files.

The cache is `/mnt/cpfs/heping/data/public/robomme_framesamp_cache`. It contains about 50.4 GB of front-camera 4×4 SigLIP features (16 tokens × 2048 channels × BF16), a small memory-mapped numeric index, and reusable spatial/temporal position tables. Features are stored as uint16 arrays containing the unchanged BF16 bits. The original dataset is read-only.

## Environment and preparation

Use the independent project `.venv` (Python 3.12), not `simplememvla-robomme` (Python 3.10). Core versions match the repository: JAX 0.5.3, Flax 0.10.2, Optax 0.2.4, PyTorch 2.7.1, NumPy 1.26.4, Transformers 4.53.2. Data decoding uses PyAV 17.1.0 and PyArrow 25.0.1; PyArrow 26 requires NumPy 2 and is incompatible with this runtime.

`requirements-framesamp.txt` records this runtime. A full installed-package snapshot is saved at `/mnt/cpfs/heping/data/public/_download/robomme_policy_data_20261102/framesamp_environment_freeze.txt`. The Python package metadata upper bound is corrected to `<3.13` so Python 3.12 patch releases are accepted.

The frozen visual encoder is `/mnt/cpfs/heping/checkpoints/pi05_vision_encoder/siglip_params.pkl`, downloaded through HF-mirror from `Yinpei/pi05_vision_encoder`, revision `59bd9ff4d58ea0638064bda851fd7d477ee9708c`. Expected SHA-256: `f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293`.

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning
bash scripts/prepare_lerobot_framesamp.sh
```

Preparation defaults to the four GPUs 0–3. Override with `GPU_IDS="0 1"`. Episode outputs and completion records make interrupted runs resumable. Each shard writes a separate log in `runs/framesamp_preparation/`. Successful full verification creates `meta/features_complete.json` in the cache and `runs/framesamp_preparation/COMPLETE` in the project. The training loader rejects incomplete caches.

For a one-episode check, use:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/build_lerobot_framesamp.py \
  --source /mnt/cpfs/heping/data/public/robomme \
  --cache /mnt/cpfs/heping/data/public/robomme_framesamp_cache \
  --openpi-data-home /mnt/cpfs/heping/checkpoints --max-episodes 1
```

## Training

The model, flow-matching loss and existing data transforms are unchanged. The loader selects the adapter when `--dataset-path` contains `meta/lerobot_framesamp.json`; original pickle datasets keep their existing path.

```bash
bash scripts/finetune_framesamp_lerobot.sh
```

This uses batch size 64, four FSDP devices, action horizon 20, the original `perceptual-framesamp-modul.yaml`, and the author's normalization statistics under `runs/assets/mme_vla_suite/robomme/`.

Before training, copy the tracked `assets/norm_stats.json` to `runs/assets/mme_vla_suite/robomme/norm_stats.json` if that runtime asset is absent. Review the server profile in `configs/train/robomme_framesamp_modul_repro.yaml`; `bash scripts/train_framesamp_modul_repro.sh` prints it without starting training.

Full training additionally requires the JAX/Orbax π₀.₅ base checkpoint at `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params`. An existing PyTorch `model.safetensors` is not this checkpoint format. Feature generation needs only the visual encoder; it does not start policy training.

## Preserved behavior and limits

- Only execution frames are supervised; history sampling includes demonstration frames and the current frame.
- Histories use the original uniform frame selection, 16 tokens per frame, a 512-token budget, and right padding with the original validity masks.
- Twenty-step action chunks repeat the last action at episode boundaries. They never cross into the next episode.
- Memory features use the original frozen SigLIP, RGB normalization before resizing, and mean pooling. Position tables use the repository's `PosEmb3D` implementation.
- LeRobot stores actions as float32 and images as AV1 video. A checked official sample matched state and subgoal annotations; action differences were about 1.1e-7. Recomputed visual features use decoded video pixels, so inputs are not bit-identical to the original raw RGB/HDF5 feature cache. This is a LeRobot-based reproduction of the method, not a guarantee of identical paper scores.
- For one comparison at episode 0, timestep 48, the mean token cosine similarity between recomputed 4×4 features and the official raw-RGB cache was 0.999405. This is a one-frame diagnostic, not a dataset-wide equivalence result.
- This compact cache supports perceptual FrameSamp with one front-view memory camera and 4×4 tokens. It does not include the 8×8 cache required for TokenDrop or recurrent variants.

## Validation

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_lerobot_framesamp.py' -v
```

The checks cover demonstration exclusion, preservation of demonstration history, uniform sampling, padding masks, action episode boundaries, duplicate-row rejection, incomplete-cache rejection, and reopening memory maps in spawned data-loader workers. Real-video checks additionally inspect demonstration starts, execution starts and final frames across multiple episodes and both cameras.

After all shards finish, preparation also passes real samples through the original normalization, image transforms and tokenizer, checks action and history shapes and finite values, and reads a shuffled batch through the actual training loader with two spawned workers. It saves the result in `runs/framesamp_preparation/data_check.json`.

Completed on 2026-11-02: all 1,600 feature files and 768,897 frames were verified. Total cache size, including the numeric index and metadata, is 50,459,805,923 bytes (about 50.46 GB). The six behavior tests passed; eight fixed real samples and one four-sample shuffled batch through two spawned workers passed shape and finite-value checks. GPU devices 0–3 were released after generation. At the end of feature preparation, policy training had not been started and the JAX/Orbax π₀.₅ base parameters were not yet present. Weight preparation and training are separate subsequent steps.
