"""P3 preparation: replay the identical streaming writer for every execution anchor."""

import dataclasses
from pathlib import Path
from common import parser, start


def main():
    cli = parser(__doc__)
    cli.add_argument("--projected", type=Path, required=True)
    cli.add_argument("--scorer", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--labels", type=Path)
    cli.add_argument("--evict", choices=["learned", "fifo", "uniform", "random", "oracle"], default="learned")
    args = cli.parse_args()
    prepared = start(args)
    if prepared is None:
        return
    from flax import nnx
    from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer
    from mme_vla_suite.shared.percmem_io import load_module
    from mme_vla_suite.training.percmem_stages import rollout_banks
    cfg = dataclasses.replace(prepared[2], evict=args.evict)
    scorer = EvictScorer(cfg, rngs=nnx.Rngs(0))
    metadata = load_module(scorer, args.scorer)
    rollout_banks(args.cache, args.projected, args.output, cfg, scorer, metadata, labels_dir=args.labels)


if __name__ == "__main__":
    main()
