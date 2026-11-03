"""P1: project with the frozen P0 adapter, derive labels and check their causal usefulness."""

from pathlib import Path
from common import parser, start


def main():
    cli = parser(__doc__)
    cli.add_argument("--projected", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--validation-samples", type=int, default=200)
    cli.add_argument("--counterfactual", action="store_true", help="Direct single-frame deletion labels on a small episode subset")
    cli.add_argument("--max-episodes", type=int)
    cli.add_argument("--cf-scale", type=float, default=1.0, help="Explicit multiplicative scale for raw FM deletion deltas")
    args = cli.parse_args()
    prepared = start(args)
    if prepared is None:
        return
    config, history, cfg = prepared
    if args.params is None:
        raise ValueError("P1 requires a completed P0 --params directory")
    if cfg.stage != "teacher" or cfg.input_space != "native":
        raise ValueError("P1 must use the full-history native-input teacher configuration")
    if args.validation_samples < 2:
        raise ValueError("At least two deletion validation samples are required")
    from mme_vla_suite.shared.percmem_io import load_bundle
    from mme_vla_suite.training.percmem_stages import build_attention_labels, project_cache
    from mme_vla_suite.training.percmem_validation import counterfactual_labels, validate_attention_labels
    bundle, identity = load_bundle(args.params, cfg)
    project_cache(args.cache, args.projected, bundle, identity)
    if args.counterfactual:
        if args.max_episodes is None or args.max_episodes < 1 or args.cf_scale <= 0:
            raise ValueError("Counterfactual labeling requires positive --max-episodes and --cf-scale")
        counterfactual_labels(config, history, args.params, args.cache, args.projected, args.output,
                              args.max_episodes, args.cf_scale)
    else:
        build_attention_labels(args.cache, args.projected, args.output, bundle, identity)
        validate_attention_labels(config, history, args.params, args.cache, args.output,
                                   samples=args.validation_samples)


if __name__ == "__main__":
    main()
