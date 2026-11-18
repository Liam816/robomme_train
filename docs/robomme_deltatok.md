# Train the EvoMem-VLA delta tokenizer on RoboMME

Stage 1 adapts the Kinetics-700 DeltaTok encoder and decoder to RoboMME. Frozen DINOv3 ViT-B/16 extracts 256 spatial patches from each 256×256 RGB frame. The encoder reads the ordered source and target patches and produces one 768-dimensional token. The decoder reconstructs the target patches from the source patches and that token. Training minimizes mean log-cosh in FP32 under BF16 autocast. [EvoMem-VLA Appendix A.2](https://arxiv.org/pdf/2610.05418v1#page=13)

## Weights and data

- Use the official `Amazon-FAR/deltatok-kinetics` **`pytorch_model.bin`** at revision `bd33ddfef720c5171f5adc0ff0d60ee9ae774de9`. It contains the 12-layer encoder and 12-layer decoder. This revision uses per-head Q/K normalization before 2D RoPE. RoPE allows the pretrained 512×512 tokenizer to run at 256×256 without resizing learned position weights. [Official checkpoint](https://huggingface.co/Amazon-FAR/deltatok-kinetics/tree/bd33ddfef720c5171f5adc0ff0d60ee9ae774de9), [official implementation](https://github.com/amazon-far/deltatok/tree/8cfafc97beda9649f5660536366a0da86821cfc4)
- Use the local timm DINOv3 ViT-B/16 checkpoint at `/mnt/cpfs/heping/checkpoints/dinov3_vitb16_pretrain_lvd1689m_timm.pth`. If copying the recipe to another server, download `pytorch_model.bin` from the corresponding timm model. [timm DINOv3 ViT-B/16 LVD-1689M](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/tree/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da)
- Use the existing `/mnt/cpfs/heping/data/public/robomme` LeRobot v3 video release. The sampler uses the release's contiguous 100-episode blocks for its 16 benchmark tasks. It selects source and target from the same episode and camera at indices `t` and `t+15`. The default includes the full observation timeline and samples from both `image` and `wrist_image`. Each global batch of 128 contains eight pairs per task. [RoboMME LeRobot v3 video release](https://huggingface.co/datasets/pepijn223/robomme_data_lerobot_video)

If the DeltaTok file needs to be downloaded, run the following with the Hugging Face CLI on a machine that can reach it.

```bash
hf download Amazon-FAR/deltatok-kinetics pytorch_model.bin \
  --revision bd33ddfef720c5171f5adc0ff0d60ee9ae774de9 \
  --local-dir /mnt/cpfs-standard/heping/projects/robomme_policy_learning_wd/weights/deltatok-kinetics-bd33ddf
```

The expected SHA-256 is `4fb478e81e70e743472ec59f959573476f5acea205aaa918c9934211126d35a9`.

## Run training

The server's `/usr/local/bin/python` has PyTorch 2.3.1, torchvision 0.18.1, timm 1.0.24, PyAV, NumPy, PyArrow and ml_dtypes. The command below uses the Table 4 batch size, learning rate, weight decay and gradient clipping. The implementation uses AdamW with betas (0.9, 0.999), a constant learning rate and decay on linear weights. Set `DELTA_TRAIN_STEPS` to choose the Stage 1 update budget. [EvoMem-VLA Appendix A.2](https://arxiv.org/pdf/2610.05418v1#page=13)

```bash
cd /mnt/cpfs/heping/projects/robomme_policy_learning_wd
export DELTA_TRAIN_STEPS=1000  # Example budget for an initial run.
PYTHONPATH=src /usr/local/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/train_robomme_deltatok.py \
  --dataset /mnt/cpfs/heping/data/public/robomme \
  --dino-checkpoint /mnt/cpfs/heping/checkpoints/dinov3_vitb16_pretrain_lvd1689m_timm.pth \
  --init-checkpoint /mnt/cpfs-standard/heping/projects/robomme_policy_learning_wd/weights/deltatok-kinetics-bd33ddf/pytorch_model.bin \
  --output-dir /mnt/cpfs-standard/heping/projects/robomme_policy_learning_wd/runs/deltatok-stage1 \
  --steps "$DELTA_TRAIN_STEPS"
```

For a short single-GPU check, set `CUDA_VISIBLE_DEVICES` to a free GPU, use `--nproc_per_node=1`, and pass `--steps 2 --batch-size 2 --workers 1` with a separate output directory.

- `last.pt` contains both tokenizer halves, AdamW state, the completed step, and the data and training settings. To resume, keep the same output directory, optimizer settings and rank layout, replace `--init-checkpoint ...` with `--resume`, and set `--steps` to the desired total.
- `encoder.pt` contains the encoder after the final step. Load its `encoder` state into `DeltaEncoder()`, set it to eval mode, and use it with the frozen DINOv3 backbone. It accepts two `[B, 256, 768]` patch tensors and returns `[B, 1, 768]`. [EvoMem-VLA Stage 2](https://arxiv.org/html/2610.05418v1#S3.SS4)
