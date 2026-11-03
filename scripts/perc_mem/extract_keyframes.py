"""Build native FP16 8x8 keyframes using the exact serving history encoder."""

from pathlib import Path
from common import parser, start


def main():
    cli = parser(__doc__)
    cli.add_argument("--index-cache", type=Path, default=Path("/mnt/cpfs/heping/data/public/robomme_framesamp_cache"))
    cli.add_argument("--base-params", type=Path, default=Path("/mnt/cpfs/heping/checkpoints/openpi-assets/checkpoints/pi05_base/params"))
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--index-only", action="store_true")
    cli.add_argument("--verify-only", action="store_true")
    args = cli.parse_args()
    prepared = start(args)
    if prepared is None:
        return
    _, _, cfg = prepared
    import hashlib
    import json
    import jax.numpy as jnp
    import numpy as np
    from mme_vla_suite.shared.percmem_io import restore_history_encoder
    from mme_vla_suite.training.lerobot_percmem import prepare_cache, validate_cache
    from mme_vla_suite.training.lerobot_framesamp import VideoReader, write_json
    from openpi.shared.nnx_utils import module_jit
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.verify_only:
        validate_cache(args.cache)
        print("Native cache validation passed")
        return
    meta = prepare_cache(args.index_cache, args.cache, cfg)
    if args.index_only:
        print(json.dumps(meta, indent=2))
        return
    encoder, vision_id = restore_history_encoder(args.base_params)
    encode = module_jit(encoder.__call__)
    source = json.loads((args.cache / "meta/source_index.json").read_text())
    reader = VideoReader(source)
    try:
        for episode in source["episodes"]:
            name = f"episode_{episode['episode_index']:06d}"
            output = args.cache / f"native/{name}.npy"
            marker = output.with_suffix(".json")
            if marker.exists():
                old = json.loads(marker.read_text())
                if old["cache_id"] != meta["cache_id"] or old["vision_id"] != vision_id:
                    raise ValueError("Refusing to mix vision encoders in one cache")
                continue
            with np.load(args.cache / f"keyframes/{name}.npz") as plan:
                rows = plan["row"]
            result = np.empty((len(rows), 64, 2048), np.float16)
            selected = set(rows.tolist())
            batch, count = [], 0
            for row, rgb in enumerate(reader.frames(episode)):
                if row not in selected:
                    continue
                batch.append(rgb)
                if len(batch) == args.batch_size or row == rows[-1]:
                    valid = len(batch)
                    while len(batch) < args.batch_size:
                        batch.append(batch[-1])
                    values = np.asarray(encode(jnp.asarray(np.stack(batch))))[:valid]
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite historical visual features")
                    result[count:count+valid] = values
                    count += valid
                    batch.clear()
            if count != len(rows):
                raise ValueError("Incomplete keyframe extraction")
            temp = output.with_name(output.name + ".tmp.npy")
            np.save(temp, result)
            temp.replace(output)
            write_json(marker, {"cache_id": meta["cache_id"], "vision_id": vision_id,
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            print(f"encoded {name}: {count} keyframes", flush=True)
    finally:
        reader.close()
    write_json(args.cache / "meta/native_complete.json", {"cache_id": meta["cache_id"], "vision_id": vision_id})
    validate_cache(args.cache)


if __name__ == "__main__":
    main()
