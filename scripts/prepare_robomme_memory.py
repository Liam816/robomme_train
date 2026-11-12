"""Extract only the reference run's seeded, newly initialized memory parameters (CPU)."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, profile, output = (p.resolve() for p in (args.reference_repo, args.config, args.output))
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    sys.path[:0] = [str(repo / p) for p in ("scripts", "src", "packages/openpi-client/src")]
    os.chdir(repo)
    from flax import nnx
    from flax import traverse_util
    import jax
    import numpy as np
    from run_training_profile import build_config
    import yaml

    profile = yaml.safe_load(profile.read_text())
    config, _ = build_config(profile)

    def init(key):
        _, init_key = jax.random.split(key)
        _, model_key = jax.random.split(init_key)
        params = nnx.state(config.model.create(model_key)).to_pure_dict()
        return {k: v for k, v in traverse_util.flatten_dict(params, sep="/").items() if "mem" in k}

    params = jax.device_get(jax.jit(init)(jax.random.key(config.seed)))
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        tmp = Path(temp.name)
        try:
            np.savez(temp, seed=config.seed, **params)
            temp.close()
            os.link(tmp, output)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"Saved {len(params)} seed-{config.seed} memory arrays to {output}")


if __name__ == "__main__":
    main()
