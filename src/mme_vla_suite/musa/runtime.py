"""Resolve host paths without changing b4432b1's experiment YAML or hyperparameters."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml


ALLOWED = {
    "runtime": {"openpi_data_home", "compilation_cache_dir"},
    "data": {"dataset_path", "norm_assets_dir"},
    "weights": {"params_path"},
    "train": {"assets_base_dir", "checkpoint_base_dir"},
}


def load_profile(path=None):
    path = Path(path or os.environ["ROBOMME_MUSA_CONFIG"]).resolve()
    profile = yaml.safe_load(path.read_text())
    runtime_path = os.environ.get("ROBOMME_MUSA_RUNTIME")
    if runtime_path:
        overlay_path = Path(runtime_path).resolve()
        overlay = yaml.safe_load(overlay_path.read_text())
        for group, values in overlay.items():
            if group == "musa":
                profile[group] = values
            elif group in ALLOWED and set(values) <= ALLOWED[group]:
                profile[group].update(values)
            else:
                raise ValueError("Host overlay may only change paths and MUSA execution settings: " + group)
        profile["musa"]["runtime_overlay"] = str(overlay_path)
        profile["musa"]["runtime_overlay_sha256"] = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    profile["musa"]["source_profile"] = str(path)
    profile["musa"]["source_profile_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return profile
