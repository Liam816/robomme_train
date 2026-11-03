"""Bootstrap offline tools without changing project dependencies or shell configuration."""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "packages/openpi-client/src"), str(PROJECT / "scripts")]


def parser(description):
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--execute", action="store_true", help="Perform work; default prints the requested operation only")
    value.add_argument("--devices", default="", help="Explicit CUDA IDs, e.g. 4; default is CPU")
    value.add_argument("--config", type=Path, default=PROJECT / "configs/train/robomme_percmem_v1_teacher_repro.yaml")
    value.add_argument("--cache", type=Path, default=Path("/mnt/cpfs/heping/data/public/robomme_percmem_v1_cache"))
    value.add_argument("--params", type=Path, help="Completed P0 teacher params directory")
    return value


def start(args):
    if not args.execute:
        print(json.dumps({"mode": "review_only", "work_started": False,
                          "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2))
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    os.environ["JAX_PLATFORMS"] = "cuda" if args.devices else "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.chdir(PROJECT)
    import yaml
    from mme_vla_suite.training.percmem_profile import build_config
    from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
    profile = yaml.safe_load(args.config.read_text())
    config, history = build_config(profile)
    return config, history, PercMemConfig.from_history(history)
