"""Cache only the frozen SigLIP 4x4 features needed by FrameSamp policies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from mme_vla_suite.training.lerobot_framesamp import VideoReader, prepare_index, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--openpi-data-home", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards or args.batch_size < 1:
        raise ValueError("Invalid shard or batch size")
    cache = Path(args.cache).resolve()
    metadata = prepare_index(args.source, cache)
    print(f"INDEX episodes={len(metadata['episodes'])} frames={metadata['total_frames']} execution_samples={metadata['execution_samples']}", flush=True)
    if args.index_only:
        return
    if args.verify_only:
        verify_cache(cache, metadata, require_complete=True)
        return

    os.environ["OPENPI_DATA_HOME"] = str(Path(args.openpi_data_home).resolve())
    weights = Path(os.environ["OPENPI_DATA_HOME"]) / "pi05_vision_encoder/siglip_params.pkl"
    digest = hashlib.sha256()
    with weights.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    weights_sha256 = digest.hexdigest()
    if weights_sha256 != "f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293":
        raise ValueError("Vision weights do not match the official published checksum")

    import jax
    import jax.numpy as jnp
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size
    from mme_vla_suite.shared.posemb_3d import PosEmb3D
    from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
    from openpi.shared import image_tools

    print(f"DEVICES {jax.devices()}", flush=True)
    posemb = PosEmb3D(dim=768)
    max_length = max(episode["length"] for episode in metadata["episodes"])
    for name, value in [("position_spatial", posemb.spatial_pe4x4),
                        ("position_temporal", posemb.compute_temporal_pe(max_length))]:
        path = cache / f"meta/{name}.npy"
        if not path.exists():
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
            np.save(temporary, np.asarray(value))
            temporary.replace(path)

    encoder = SigLipTokenizer(inference_batch_size=args.batch_size)
    vision = jax.jit(encoder.__call__)

    def encode(images):
        # Match MemoryBuffer.add_buffer: normalize on CPU before the JAX resize.
        normalized = images.astype(np.float32) / 255.0 * 2.0 - 1.0
        resized = image_tools.resize_with_pad(jnp.array(normalized), 224, 224)
        return pool_tokens_to_size(vision(resized[:, None]), target_size=16)[:, 0]

    reader = VideoReader(metadata)
    selected = [episode for episode in metadata["episodes"] if episode["episode_index"] % args.num_shards == args.shard_id]
    if args.max_episodes is not None:
        selected = selected[:args.max_episodes]
    (cache / "features").mkdir(exist_ok=True)
    started, processed = time.monotonic(), 0
    try:
        for episode in selected:
            episode_id, length = episode["episode_index"], episode["length"]
            output = cache / f"features/episode_{episode_id:06d}.npy"
            marker = output.with_suffix(".json")
            if marker.exists():
                old = json.loads(marker.read_text())
                if old["weights_sha256"] != weights_sha256 or old["source_info_sha256"] != metadata["source_info_sha256"]:
                    raise ValueError(f"Incompatible existing feature cache: {output}")
                if not output.exists() or np.load(output, mmap_mode="r").shape != (length, 16, 2048):
                    raise ValueError(f"Corrupt existing cache: {output}")
                print(f"SKIP episode={episode_id}", flush=True)
                continue
            temporary = output.with_name(output.name + ".incomplete")
            mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint16, shape=(length, 16, 2048))
            batch, offset = [], 0
            for image in reader.frames(episode):
                batch.append(image)
                if len(batch) == args.batch_size or offset + len(batch) == length:
                    valid = len(batch)
                    while len(batch) < args.batch_size:
                        batch.append(batch[-1])
                    features = np.asarray(jax.device_get(encode(np.stack(batch))))[:valid]
                    if str(features.dtype) != "bfloat16" or not np.isfinite(features.astype(np.float32)).all():
                        raise ValueError("Unexpected feature dtype or non-finite features")
                    mmap[offset:offset + valid] = features.view(np.uint16)
                    offset += valid
                    batch.clear()
            if offset != length:
                raise ValueError(f"Incomplete episode {episode_id}: {offset}/{length}")
            mmap.flush()
            del mmap
            temporary.replace(output)
            write_json(marker, {"episode_index": episode_id, "frames": length, "shape": [length, 16, 2048],
                "dtype": "uint16_bfloat16_bits", "weights_sha256": weights_sha256,
                "source_info_sha256": metadata["source_info_sha256"], "input": "LeRobot AV1 decoded RGB",
                "pooling": "official mean pooling from 16x16 to 4x4"})
            processed += length
            print(f"OK episode={episode_id} frames={length} processed={processed} fps={processed / (time.monotonic()-started):.2f}", flush=True)
    finally:
        reader.close()
    verify_cache(cache, metadata, require_complete=False)


def verify_cache(cache, metadata, require_complete):
    complete = []
    for episode in metadata["episodes"]:
        episode_id, length = episode["episode_index"], episode["length"]
        path = cache / f"features/episode_{episode_id:06d}.npy"
        marker = path.with_suffix(".json")
        if not marker.exists():
            continue
        record = json.loads(marker.read_text())
        array = np.load(path, mmap_mode="r")
        if (array.shape != (length, 16, 2048) or array.dtype != np.uint16
                or record["source_info_sha256"] != metadata["source_info_sha256"]
                or record["weights_sha256"] != "f16e9312f24760e6426ab82e42b606e80542ffbf351c9b40736bfb341d07f293"):
            raise ValueError(f"Invalid feature cache: {path}")
        complete.append(episode_id)
    print(f"CACHE completed_episodes={len(complete)}/{len(metadata['episodes'])}", flush=True)
    if len(complete) == len(metadata["episodes"]):
        write_json(cache / "meta/features_complete.json", {"episodes": len(complete), "frames": metadata["total_frames"],
                   "source_info_sha256": metadata["source_info_sha256"]})
    elif require_complete:
        raise RuntimeError("Feature cache is incomplete")


if __name__ == "__main__":
    main()
