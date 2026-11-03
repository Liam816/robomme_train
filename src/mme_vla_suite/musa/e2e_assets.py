"""Initialize b4432b1's E2E modules with the original P0 seed and module order."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import numpy as np
from flax import nnx, traverse_util
from omegaconf import OmegaConf

from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.musa.prepare_assets import load_tree, save_tree
from mme_vla_suite.musa.model import flatten
from mme_vla_suite.musa.runtime import load_profile
from openpi.shared.array_typing import disable_typechecking


def main():
    root = Path(os.environ["ROBOMME_MUSA_ROOT"])
    repo = Path(os.environ["ROBOMME_MUSA_REPO"])
    profile = OmegaConf.create(load_profile())
    shared = Path("/data/heping/robomme_musa/assets")
    os.chdir(repo)
    cfg = OmegaConf.to_container(get_history_config(profile.model.history_config).perc_mem)
    assert cfg["stage"] == "e2e" and cfg["input_space"] == "native"
    assert cfg["e2e_temperature"] == 1.0 and not cfg["scorer_use_instr"]
    assert Path(profile.weights.params_path).resolve() == (shared / "pi05_base/params").resolve()
    assert json.loads((shared / "download_complete.json").read_text())["objects"] == 20
    assert json.loads((shared / "tokenizer_verification.json").read_text())["passed"]
    root.joinpath("assets").mkdir(parents=True, exist_ok=True)
    tokenizer = root / "assets/paligemma_tokenizer.model"
    if not tokenizer.exists():
        tokenizer.symlink_to(shared / "paligemma_tokenizer.model")
    _, init_rng = jax.random.split(jax.random.key(profile.train.seed))
    _, model_rng = jax.random.split(init_rng)
    with disable_typechecking():
        model = HistoryPi0Config(**OmegaConf.to_container(profile.model)).create(model_rng)
    state = nnx.state(model, nnx.Param).to_pure_dict()
    memory = {k: v for k, v in state.items() if k.startswith("mem_")}
    expected = flatten({k: v for k, v in state.items() if not k.startswith("mem_")})
    base_manifest = json.loads((Path(profile.musa.base_export) / "manifest.json").read_text())
    assert expected.keys() == base_manifest.keys()
    assert all(list(np.shape(expected[k])) == base_manifest[k]["shape"] for k in expected)
    # This is a seed-equivalence check, not a warm start from trained P0 weights.
    prior_initial = flatten(load_tree(shared / "memory_initial", verify=True))
    current = flatten(memory)
    assert current.keys() == prior_initial.keys()
    for key in current:
        if not np.array_equal(np.asarray(current[key]), prior_initial[key]):
            raise ValueError("Initialization changed from the original P0 seed: " + key)
    save_tree(Path(profile.musa.memory_initial), memory)
    source = json.loads((shared / "export_complete.json").read_text())
    result = {**source, "memory_config": cfg, "source": str(profile.weights.params_path),
              "initialization": "fresh original E2E HistoryPi0Config.create; same seed42/module order as P0",
              "initial_memory_matches_p0_seed": True,
              "memory_parameters": sum(np.size(v) for v in current.values()),
              "worktree": str(repo), "source_commit": "b4432b1ef2ab746b13eec32f25bcc5862c582164"}
    assert result["base_parameters"] + result["memory_parameters"] == 3386067729
    (root / "assets/export_complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
