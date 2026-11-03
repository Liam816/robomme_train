"""Review v1 profiles on CPU by default. --execute is required for training."""

import argparse
import json
import os
from pathlib import Path
import sys

import yaml
from run_training_profile import assert_subset


PROJECT = Path(__file__).resolve().parents[1]


def configure_environment(runtime, execute):
    devices = runtime["cuda_visible_devices"]
    if (not isinstance(devices, list) or len(devices) != 4
            or any(type(device) is not int or device < 0 for device in devices)
            or len(set(devices)) != 4):
        raise ValueError("Expected four distinct nonnegative physical CUDA device IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    os.environ["JAX_PLATFORMS"] = "cuda" if execute else "cpu"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(runtime["xla_python_client_mem_fraction"])
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(runtime["xla_python_client_preallocate"]).lower()
    os.environ["OMP_NUM_THREADS"] = str(runtime["omp_num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(runtime["openblas_num_threads"])
    os.environ["OPENPI_DATA_HOME"] = runtime["openpi_data_home"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    profile = yaml.safe_load(args.config.read_text())
    configure_environment(profile["runtime"], args.execute)
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "packages/openpi-client/src")]
    os.chdir(PROJECT)
    from mme_vla_suite.training.percmem_profile import build_config, review_config
    from omegaconf import OmegaConf
    config, memory = build_config(profile)
    assert_subset(profile["expected_memory"], OmegaConf.to_container(memory, resolve=True))
    report = review_config(profile, config, memory)
    report["execution_requested"] = args.execute
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not args.execute:
        return
    if report["execution_blockers"]:
        raise SystemExit("Execution blocked:\n" + "\n".join(report["execution_blockers"]))
    # Full cache and bank identity checks run before importing the training entry point.
    from mme_vla_suite.training.lerobot_percmem import LeRobotPercMemDataset
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = LeRobotPercMemDataset(config.dataset_path, data_config, memory, config.model.action_horizon)
    del dataset
    import jax
    if len(jax.devices("gpu")) != 4:
        raise RuntimeError(f"Expected four visible GPUs from physical IDs {profile['runtime']['cuda_visible_devices']}")
    binding = report["compilation_cache_binding"]
    target, link = Path(binding["symlink_target"]), Path(binding["upstream_path"])
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    import train
    train.main(config, tentative_run=False)


if __name__ == "__main__":
    main()
