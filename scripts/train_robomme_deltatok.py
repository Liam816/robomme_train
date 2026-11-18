"""Adapt the official DeltaTok checkpoint to RoboMME with frozen DINOv3."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mme_vla_suite.training.deltatok import IMAGE_SIZE
from mme_vla_suite.training.deltatok import DeltaTokenizer
from mme_vla_suite.training.deltatok import FrozenDINOv3
from mme_vla_suite.training.deltatok import reconstruction_loss
from mme_vla_suite.training.robomme_delta_data import CAMERAS
from mme_vla_suite.training.robomme_delta_data import TASKS
from mme_vla_suite.training.robomme_delta_data import RoboMMEDeltaPairs
from mme_vla_suite.training.robomme_delta_data import TaskBalancedBatches

CHECKPOINT_FORMAT = "robomme_deltatok_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="RoboMME LeRobot v3 video root")
    parser.add_argument("--dino-checkpoint", type=Path, required=True, help="Local timm DINOv3 ViT-B/16 .pth or .bin")
    init = parser.add_mutually_exclusive_group(required=True)
    init.add_argument("--init-checkpoint", type=Path, help="Official Amazon-FAR/deltatok-kinetics pytorch_model.bin")
    init.add_argument("--resume", action="store_true", help="Restore last.pt from --output-dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="Total optimizer updates, including resumed steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Pairs per GPU")
    parser.add_argument("--frame-gap", type=int, default=15)
    parser.add_argument("--cameras", choices=CAMERAS, nargs="+", default=CAMERAS)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1e-2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--workers", type=int, default=4, help="Video decoding workers per rank")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "batch_size", "frame_gap", "log_every", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.seed < 0:
        parser.error("--workers and --seed must be nonnegative")
    if not all(math.isfinite(value) for value in (args.learning_rate, args.weight_decay, args.grad_clip)):
        parser.error("Optimizer settings must be finite")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("Learning rate and gradient clipping must be positive and weight decay nonnegative")
    return args


def save_atomic(state: dict, path: Path) -> None:
    handle = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(state, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> None:
    dataset = RoboMMEDeltaPairs(args.dataset, args.frame_gap, tuple(args.cameras))
    output = args.output_dir.resolve()
    latest = output / "last.pt"
    encoder_path = output / "encoder.pt"
    dino_path = args.dino_checkpoint.resolve(strict=True)
    dino_stat = dino_path.stat()
    recipe = {
        "dataset_fingerprint": dataset.fingerprint,
        "dino_checkpoint": str(dino_path),
        "dino_file_size": dino_stat.st_size,
        "dino_file_mtime_ns": dino_stat.st_mtime_ns,
        "image_size": IMAGE_SIZE,
        "cameras": tuple(args.cameras),
        "frame_gap": args.frame_gap,
        "batch_size": args.batch_size,
        "world_size": world_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "precision": args.precision,
    }

    with ExitStack() as cleanup:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            lock = cleanup.enter_context((output / ".train.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another run is writing to {output}") from error
            if not args.resume and (latest.exists() or encoder_path.exists()):
                raise FileExistsError(f"{output} already has checkpoints. Use --resume or a new --output-dir.")

        state_path = latest if args.resume else args.init_checkpoint
        state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
        model = DeltaTokenizer()
        start_step = 0
        if args.resume:
            if state.get("format") != CHECKPOINT_FORMAT or state.get("recipe") != recipe:
                raise ValueError("Resume requires the same dataset, DINOv3 file, optimizer settings and batch layout")
            start_step = state["step"]
            if args.steps < start_step:
                raise ValueError(f"--steps={args.steps} precedes the saved step {start_step}")
            model.load_state_dict(state["tokenizer"], strict=True)
        else:
            model.load_state_dict(state, strict=True)
        model.to(device)

        # Match upstream DeltaTok's decay policy for linear weights.
        decay = [layer.weight for layer in model.modules() if isinstance(layer, nn.Linear)]
        decay_ids = {id(weight) for weight in decay}
        no_decay = [weight for weight in model.parameters() if id(weight) not in decay_ids]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.learning_rate,
            fused=True,
        )
        if args.resume:
            optimizer.load_state_dict(state["optimizer"])
        del state

        backbone = FrozenDINOv3(dino_path).to(device)
        network = model
        if world_size > 1:
            network = DistributedDataParallel(
                model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True
            )
        sampler = TaskBalancedBatches(
            dataset.task_offsets, args.batch_size, rank, world_size, args.seed, start_step, args.steps
        )
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
            multiprocessing_context="spawn" if args.workers else None,
        )
        if rank == 0:
            print(json.dumps({
                "frame_pairs_per_task": dict(zip(TASKS, np.diff(dataset.task_offsets).tolist(), strict=True)),
                "global_batch_size": sampler.global_batch_size,
                "start_step": start_step,
                "total_steps": args.steps,
                "output_dir": str(output),
            }), flush=True)

        generator = torch.Generator(device=device)
        for step, frames in enumerate(loader, start=start_step + 1):
            generator.manual_seed(args.seed + (step - 1) * world_size + rank)
            optimizer.zero_grad(set_to_none=True)
            frames = frames.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                source, target = backbone(frames).unbind(dim=1)
                reconstructed = network(source, target, generator=generator)
            loss = reconstruction_loss(reconstructed, target)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()

            if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                mean_loss = loss.detach()
                if world_size > 1:
                    dist.all_reduce(mean_loss)
                    mean_loss /= world_size
                if rank == 0:
                    print(
                        f"step={step}/{args.steps} loss={mean_loss.item():.6f} "
                        f"grad_norm_before_clip={grad_norm.item():.6f}",
                        flush=True,
                    )
            if step % args.save_every == 0 or step == args.steps:
                if rank == 0:
                    save_atomic({
                        "format": CHECKPOINT_FORMAT, "step": step, "recipe": recipe,
                        "tokenizer": model.state_dict(), "optimizer": optimizer.state_dict(),
                    }, latest)
                if world_size > 1:
                    dist.barrier()

        if rank == 0:
            encoder_state = {
                key: value for key, value in model.state_dict().items() if not key.startswith("decoder_blocks.")
            }
            save_atomic({
                "format": CHECKPOINT_FORMAT, "step": args.steps, "recipe": recipe, "encoder": encoder_state
            }, encoder_path)
            print(f"Saved {latest} and {encoder_path}", flush=True)


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Bfloat16 requires a compatible GPU. Use --precision fp32 to override.")
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    try:
        train(args, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
