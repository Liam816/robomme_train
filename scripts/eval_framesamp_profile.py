"""Evaluate the retained checkpoints from a FrameSamp training YAML on the official test split."""

import argparse
import os
from pathlib import Path
import sys

from train_framesamp_pytorch import DEFAULT
from train_framesamp_pytorch import ROOT
from train_framesamp_pytorch import ckpt_root
from train_framesamp_pytorch import resolve
from train_framesamp_pytorch import should_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, help="Use another run with the same checkpoint schedule")
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs/eval/robomme_server.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = resolve(args.config.resolve(), [])
    train, eval_cfg = cfg["train"], cfg["evaluation"]
    saved = [s for s in range(train["num_train_steps"]) if should_save(s, train)]
    retained = sorted(
        set(saved[-train["checkpoint_max_to_keep"] :])
        | {s for s in saved if train["keep_period"] and s % train["keep_period"] == 0}
    )
    count = eval_cfg["last_checkpoints"]
    if count < 1 or len(retained) < count:
        raise ValueError("Checkpoint retention cannot supply the requested number of evaluation checkpoints")
    if eval_cfg["tasks"] != 16 or not 1 <= eval_cfg["episodes_per_task"] <= 50:
        raise ValueError("This profile uses the 16 official RoboMME test tasks and episode IDs 0–49")
    root = (args.checkpoint_root or ckpt_root(cfg)).resolve()
    script = "eval_robomme.py" if args.backend == "jax" else "eval_framesamp_pytorch.py"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "run",
        "--runtime-config",
        str(args.runtime_config.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--seeds",
        *map(str, eval_cfg["seeds"]),
        "--episode-ids",
        *map(str, range(eval_cfg["episodes_per_task"])),
        "--gpus",
        *map(str, cfg["runtime"]["cuda_visible_devices"]),
        "--max-steps",
        str(eval_cfg["max_episode_steps"]),
        "--execute-steps",
        str(eval_cfg["execute_action_steps"]),
    ]
    for step in retained[-count:]:
        argv.extend(("--checkpoint", str(root / str(step))))
    if args.dry_run:
        argv.append("--dry-run")
    os.chdir(ROOT)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
