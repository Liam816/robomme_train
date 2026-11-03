"""P2: train the scorer alone with the specified 64-trajectory ranking objective."""

from pathlib import Path
from common import parser, start


def main():
    cli = parser(__doc__)
    cli.add_argument("--projected", type=Path, required=True)
    cli.add_argument("--labels", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--steps", type=int, help="Original design leaves this unspecified; explicitly select the P2 budget")
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    prepared = start(args)
    if prepared is None:
        return
    if args.steps is None:
        raise ValueError("Specify --steps to define cosine decay and the 30% oracle schedule")
    from mme_vla_suite.training.percmem_stages import train_scorer
    train_scorer(args.cache, args.projected, args.labels, args.output, prepared[2], steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
