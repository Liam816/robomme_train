"""Run RoboMME checkpoints through one resumable evaluation protocol."""

# Reports use Chinese punctuation; heavy imports belong to their respective runtime.
# ruff: noqa: RUF001, PLC0415

import argparse
from contextlib import closing
from datetime import datetime
from datetime import timezone
import fcntl
import hashlib
import json
import math
from multiprocessing.connection import Client
from multiprocessing.connection import Listener
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
RUNTIME_PATHS = (
    "model_python",
    "sim_python",
    "benchmark_dir",
    "maniskill_dir",
    "asset_dir",
    "graphics_dir",
    "work_dir",
    "openpi_data_home",
)


def now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - simulator runs Python 3.10


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(directory):
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    if not files:
        raise ValueError(f"No Python/JSON source found in {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def db_connect(root, *, readonly=False):
    if readonly:
        db = sqlite3.connect(f"file:{root / 'episodes.sqlite3'}?mode=ro", uri=True, timeout=60)
    else:
        db = sqlite3.connect(root / "episodes.sqlite3", timeout=60)
    db.row_factory = sqlite3.Row
    return db


def unique(values, name):
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and contain no duplicates")
    return values


def resolve_path(value, *, follow_symlinks=True):
    path = Path(value).expanduser()
    path = REPO / path if not path.is_absolute() else path
    # Resolving the interpreter symlink would bypass its virtual environment.
    return str(path.resolve() if follow_symlinks else path.absolute())


def runtime_options(args):
    runtime = {
        "model_python": str(REPO / ".venv/bin/python"),
        "sim_python": sys.executable,
        "benchmark_dir": str(REPO / "third_party/robomme_benchmark"),
        "maniskill_dir": None,
        "asset_dir": None,
        "graphics_dir": None,
        "work_dir": "/tmp/robomme_eval",
        "openpi_data_home": os.environ.get("OPENPI_DATA_HOME"),
    }
    if args.runtime_config:
        profile = json.loads(args.runtime_config.read_text())
        unknown = set(profile) - set(RUNTIME_PATHS)
        if unknown:
            raise ValueError(f"Unknown runtime settings: {sorted(unknown)}")
        runtime.update(profile)
    for key in RUNTIME_PATHS:
        supplied = getattr(args, key)
        if supplied is not None:
            runtime[key] = supplied
        if runtime[key] is not None:
            runtime[key] = resolve_path(runtime[key], follow_symlinks=key not in ("model_python", "sim_python"))
    return runtime


def checkpoint_description(value):
    label, separator, location = value.partition("=")
    path = Path(resolve_path(location if separator else value))
    label = label if separator else path.name
    if not label or any(character in label for character in "\n\r|"):
        raise ValueError(f"Invalid checkpoint label {label!r}")
    for name in ("params/_METADATA", "assets/robomme/norm_stats.json", "_CHECKPOINT_METADATA"):
        if not (path / name).is_file():
            raise ValueError(f"Missing checkpoint file: {path / name}")
    metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError(f"Checkpoint is not finalized: {path}")
    history = path.parent / "history_config.txt"
    return {
        "label": label,
        "path": str(path),
        "metadata_sha256": sha256(path / "_CHECKPOINT_METADATA"),
        "parameter_metadata_sha256": sha256(path / "params/_METADATA"),
        "normalization_sha256": sha256(path / "assets/robomme/norm_stats.json"),
        "history_config": history.read_text().strip() if history.exists() else None,
    }


def make_manifest(args):
    runtime = runtime_options(args)
    for name in ("model_python", "sim_python"):
        if not os.access(runtime[name], os.X_OK):
            raise ValueError(f"Executable not found: {runtime[name]}")
    bench = Path(runtime["benchmark_dir"]) / "src/robomme"
    if not (bench / "env_record_wrapper/episode_config_resolver.py").is_file():
        raise ValueError(f"Initialize the official benchmark or set --benchmark-dir: {bench}")
    for name in ("maniskill_dir", "asset_dir", "graphics_dir", "openpi_data_home"):
        if runtime[name] and not Path(runtime[name]).is_dir():
            raise ValueError(f"Runtime directory not found: {runtime[name]}")
    checkpoints = [checkpoint_description(value) for value in args.checkpoint]
    unique([item["label"] for item in checkpoints], "Checkpoint labels")
    unique([item["path"] for item in checkpoints], "Checkpoint paths")
    if all(item["label"].isdigit() for item in checkpoints):
        checkpoints.sort(key=lambda item: int(item["label"]), reverse=True)
    seeds = unique(args.seeds, "Seeds")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    tasks = unique(args.tasks, "Tasks")
    episodes = unique(args.episode_ids, "Episode IDs")
    if any(episode < 0 or episode >= 50 for episode in episodes):
        raise ValueError("Official test episode IDs must be in [0, 50)")
    gpus = unique(args.gpus, "GPUs")
    if any(gpu < 0 for gpu in gpus) or not 1 <= args.streams_per_gpu <= 8:
        raise ValueError("GPU IDs must be nonnegative and streams-per-gpu must be in [1,8]")
    if not 1 <= args.execute_steps <= 1300 or not 1 <= args.max_steps <= 10000:
        raise ValueError("Invalid action execution horizon or episode step limit")
    if not 1 <= args.denoising_steps <= 1000 or not 1 <= args.vision_batch_size <= 256:
        raise ValueError("Invalid denoising step count or vision batch size")
    if args.subgoal_keep_period < 1 or args.max_attempts < 1 or args.episode_timeout < 1:
        raise ValueError("Keep period, attempts, and timeouts must be positive")
    if args.subgoal_source in ("qwenvl", "memer") and not args.subgoal_adapter:
        raise ValueError("This subgoal source requires --subgoal-adapter")
    if args.subgoal_adapter and not Path(resolve_path(args.subgoal_adapter)).is_dir():
        raise ValueError("Subgoal adapter directory does not exist")
    if args.reference_percent is not None and (
        not math.isfinite(args.reference_percent) or not 0 <= args.reference_percent <= 100
    ):
        raise ValueError("reference-percent must be finite and in [0,100]")
    metadata = {}
    for task in tasks:
        path = bench / f"env_metadata/test/record_dataset_{task}_metadata.json"
        if not path.is_file():
            raise ValueError(f"Missing official test metadata: {path}")
        metadata[path.name] = sha256(path)
    source_files = [
        REPO / "src/mme_vla_suite/policies/policy.py",
        REPO / "src/mme_vla_suite/policies/policy_config.py",
        REPO / "src/mme_vla_suite/training/config.py",
        REPO / "src/mme_vla_suite/policies/robomme_policy.py",
        REPO / "src/openpi/transforms.py",
        REPO / "src/mme_vla_suite/shared/data_utils.py",
        REPO / "src/mme_vla_suite/models/integration/history_gemma.py",
        REPO / "src/mme_vla_suite/models/integration/history_pi0.py",
        REPO / "src/mme_vla_suite/shared/mem_buffer.py",
        REPO / "examples/robomme/env_runner.py",
        REPO / "examples/robomme/utils.py",
        REPO / "examples/robomme/subgoal_predictor.py",
    ]
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in source_files}
    config_dir = REPO / "src/mme_vla_suite/models/config/robomme"
    for checkpoint in checkpoints:
        if checkpoint["history_config"]:
            path = config_dir / checkpoint["history_config"]
            if not path.is_file():
                raise ValueError(f"History configuration not found: {path}")
            source_hashes[str(path.relative_to(REPO))] = sha256(path)
    return {
        "schema_version": 1,
        "created": now(),
        "run_id": secrets.token_hex(12),
        "variant": args.variant or Path(checkpoints[0]["path"]).parent.name,
        "checkpoints": checkpoints,
        "seeds": seeds,
        "tasks": tasks,
        "episode_ids": episodes,
        "split": "test",
        "policy_config": args.policy_config,
        "max_steps": args.max_steps,
        "execute_steps": args.execute_steps,
        "denoising_steps": args.denoising_steps,
        "vision_batch_size": args.vision_batch_size,
        "subgoal_source": args.subgoal_source,
        "subgoal_type": args.subgoal_type,
        "subgoal_adapter": resolve_path(args.subgoal_adapter) if args.subgoal_adapter else None,
        "subgoal_keep_period": args.subgoal_keep_period,
        "gemini_model": args.gemini_model,
        "gpus": gpus,
        "streams_per_gpu": args.streams_per_gpu,
        "max_attempts": args.max_attempts,
        "episode_timeout": args.episode_timeout,
        "runtime": runtime,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        "reference_percent": args.reference_percent,
        "test_metadata_sha256": metadata,
        "benchmark_source_sha256": source_identity(bench),
        "maniskill_source_sha256": source_identity(Path(runtime["maniskill_dir"]) / "mani_skill")
        if runtime["maniskill_dir"]
        else None,
        "runner_sha256": sha256(SCRIPT),
        "source_sha256": source_hashes,
    }


def load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if "schema_version" in manifest:
        return manifest
    # Read the completed 2026-11-05 run without migrating its database or manifest.
    return {
        **manifest,
        "schema_version": 0,
        "variant": "framesamp-modul-lerobot-repro-s42",
        "checkpoints": [
            {"label": str(value), "path": str(Path(manifest["checkpoint_root"]) / str(value))}
            for value in sorted(manifest["checkpoints"], reverse=True)
        ],
        "tasks": list(TASKS),
        "episode_ids": list(range(manifest["episodes_per_task"])),
        "max_steps": manifest["max_execution_steps"],
        "execute_steps": manifest["executed_action_chunk"],
        "reference_percent": 44.51,
    }


def expected_keys(manifest, labels=None):
    labels = labels or [item["label"] for item in manifest["checkpoints"]]
    return {
        (label, seed, task, episode)
        for label in labels
        for seed in manifest["seeds"]
        for task in manifest["tasks"]
        for episode in manifest["episode_ids"]
    }


def initialize_database(root, manifest):
    with closing(db_connect(root)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE episodes (
            checkpoint TEXT, seed INTEGER, task TEXT, episode INTEGER,
            priority INTEGER, seed_priority INTEGER, status TEXT DEFAULT 'pending',
            gpu INTEGER, stream INTEGER, attempts INTEGER DEFAULT 0, started REAL, finished REAL,
            outcome TEXT, steps INTEGER, env_seed INTEGER, difficulty TEXT, error TEXT,
            PRIMARY KEY(checkpoint,seed,task,episode))""")
        db.executemany(
            """INSERT INTO episodes(checkpoint,seed,task,episode,priority,seed_priority)
            VALUES (?,?,?,?,?,?)""",
            (
                (item["label"], seed, task, episode, priority, seed_priority)
                for priority, item in enumerate(manifest["checkpoints"])
                for seed_priority, seed in enumerate(manifest["seeds"])
                for task in manifest["tasks"]
                for episode in manifest["episode_ids"]
            ),
        )


def claim(db, gpu, stream):
    with db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT checkpoint,seed,task,episode FROM episodes WHERE status='pending'
            AND priority=(SELECT MIN(priority) FROM episodes WHERE status!='complete')
            ORDER BY seed_priority,episode,task LIMIT 1""").fetchone()
        if row is not None:
            db.execute(
                """UPDATE episodes SET status='running',gpu=?,stream=?,attempts=attempts+1,
                started=?,error=NULL WHERE (checkpoint,seed,task,episode)=(?,?,?,?)""",
                (gpu, stream, time.time(), *row),
            )
    return row


def release_interrupted(root, gpu, stream, max_attempts):
    with closing(db_connect(root)) as db, db:
        db.execute(
            """UPDATE episodes SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END,
            error=COALESCE(error,'Worker interrupted') WHERE status='running' AND gpu=? AND stream=?""",
            (max_attempts, gpu, stream),
        )


def rows_from_database(root):
    with closing(db_connect(root, readonly=True)) as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM episodes ORDER BY checkpoint,seed,task,episode")]
    for row in rows:
        row["checkpoint"] = str(row["checkpoint"])
    return rows


def validate_rows(rows, manifest, labels=None, *, require_complete=True):
    expected = expected_keys(manifest, labels)
    keys = [(row["checkpoint"], row["seed"], row["task"], row["episode"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Episode coverage differs from the manifest (missing, duplicate, or extra keys)")
    scene_seeds = {}
    for row in rows:
        if row["status"] != "complete":
            if require_complete:
                raise ValueError("Evaluation is incomplete; inspect status before generating the final report")
            continue
        if row["outcome"] not in ("success", "fail", "timeout") or row["env_seed"] is None:
            raise ValueError("A completed row has an invalid outcome or environment seed")
        if not 1 <= row["steps"] <= manifest["max_steps"]:
            raise ValueError("A completed row exceeds the configured episode horizon")
        scene = row["task"], row["episode"]
        if scene_seeds.setdefault(scene, row["env_seed"]) != row["env_seed"]:
            raise ValueError(f"Environment seed changed between runs: {scene}")


def status_snapshot(root):
    rows = rows_from_database(root)
    counts = {}
    for row in rows:
        key = row["checkpoint"], row["seed"], row["status"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "checked_at": now(),
        "completed": sum(row["status"] == "complete" for row in rows),
        "expected": len(rows),
        "counts": [
            {"checkpoint": key[0], "seed": key[1], "status": key[2], "n": n} for key, n in sorted(counts.items())
        ],
        "active": [
            {key: row.get(key) for key in ("checkpoint", "seed", "task", "episode", "gpu", "stream", "started")}
            for row in rows
            if row["status"] == "running"
        ],
        "failed_examples": [row for row in rows if row["status"] == "failed"][:10],
    }


def score(rows):
    successes = sum(row["outcome"] == "success" for row in rows)
    return {"successes": successes, "episodes": len(rows), "percent": 100 * successes / len(rows)}


def aggregate(rows, manifest):
    validate_rows(rows, manifest)
    by_checkpoint = {}
    runs = []
    for checkpoint in manifest["checkpoints"]:
        label = checkpoint["label"]
        selected = [row for row in rows if row["checkpoint"] == label]
        by_checkpoint[label] = {**score(selected), "tasks": {}, "seeds": {}}
        for task in manifest["tasks"]:
            by_checkpoint[label]["tasks"][task] = score([row for row in selected if row["task"] == task])
        for seed in manifest["seeds"]:
            seed_rows = [row for row in selected if row["seed"] == seed]
            item = {
                **score(seed_rows),
                "tasks": {task: score([row for row in seed_rows if row["task"] == task]) for task in manifest["tasks"]},
            }
            by_checkpoint[label]["seeds"][str(seed)] = item
            runs.append({"checkpoint": label, "seed": seed, **item})
    overall = score(rows)
    reference = manifest.get("reference_percent")
    return {
        "finished": now(),
        "variant": manifest["variant"],
        "overall": overall,
        "reference_percent": reference,
        "difference_pp": overall["percent"] - reference if reference is not None else None,
        "sample_std_pp": statistics.stdev(row["percent"] for row in runs) if len(runs) > 1 else None,
        "checkpoints": by_checkpoint,
        "runs": runs,
        "tasks": {task: score([row for row in rows if row["task"] == task]) for task in manifest["tasks"]},
    }


def checkpoint_reports(root, manifest, rows):
    ready = []
    results = {}
    for checkpoint in manifest["checkpoints"]:
        label = checkpoint["label"]
        selected = [row for row in rows if row["checkpoint"] == label]
        if selected and all(row["status"] == "complete" for row in selected):
            subset = {**manifest, "checkpoints": [checkpoint]}
            result = aggregate(selected, subset)
            ready.append(label)
            results[label] = result["checkpoints"][label]
    if not ready:
        return
    output = root / "checkpoint_results.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("completed_checkpoints") == ready and previous.get("checkpoints") == results:
            return
    lines = [
        "# 已完成 checkpoint 的 RoboMME 评测",
        "",
        "| 任务 | " + " | ".join(ready) + " |",
        "|---|" + "---:|" * len(ready),
    ]
    for task in ("总体", *manifest["tasks"]):
        values = [results[label] if task == "总体" else results[label]["tasks"][task] for label in ready]
        lines.append("| " + task + " | " + " | ".join(f"{value['percent']:.2f}%" for value in values) + " |")
    write_text(root / "CHECKPOINTS.md", "\n".join(lines) + "\n")
    write_json(output, {"completed_checkpoints": ready, "checkpoints": results})


def write_text(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def report(root, manifest):
    rows = rows_from_database(root)
    result = aggregate(rows, manifest)
    checkpoint_reports(root, manifest, rows)
    labels = [item["label"] for item in manifest["checkpoints"]]
    overall = result["overall"]
    lines = [
        "# RoboMME 评测汇总",
        "",
        f"{manifest['variant']} 完成 {overall['episodes']} 条，"
        f"成功 {overall['successes']} 条，总体成功率 **{overall['percent']:.2f}%**。",
    ]
    if result["reference_percent"] is not None:
        lines += [
            "",
            f"参照成功率为 {result['reference_percent']:.2f}%，差值为 {result['difference_pp']:+.2f} 个百分点。",
        ]
    lines += ["", "| 任务 | " + " | ".join(labels) + " | 全部组合 |", "|---|" + "---:|" * (len(labels) + 1)]
    for task in ("总体", *manifest["tasks"]):
        values = [
            result["checkpoints"][label] if task == "总体" else result["checkpoints"][label]["tasks"][task]
            for label in labels
        ]
        values.append(overall if task == "总体" else result["tasks"][task])
        lines.append("| " + task + " | " + " | ".join(f"{value['percent']:.2f}%" for value in values) + " |")
    lines += ["", "| Checkpoint | Seed | 成功/总数 | 成功率 |", "|---|---:|---:|---:|"]
    for run in result["runs"]:
        lines.append(
            f"| {run['checkpoint']} | {run['seed']} | {run['successes']}/{run['episodes']} | {run['percent']:.2f}% |"
        )
    lines += [
        "",
        f"场景 split 为 test，包含 {len(manifest['tasks'])} 个任务，各 {len(manifest['episode_ids'])} 条，"
        f"每回合执行上限 {manifest['max_steps']} 步，每次执行动作块前 {manifest['execute_steps']} 步。",
    ]
    write_text(root / "REPORT.md", "\n".join(lines) + "\n")
    write_text(root / "episodes.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(root / "results.json", result)
    return result


def policy_process(root, checkpoint, address):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi_client import msgpack_numpy

    from mme_vla_suite.policies.policy_config import create_trained_policy
    from mme_vla_suite.training.config import get_config

    manifest = load_manifest(root)
    item = next(item for item in manifest["checkpoints"] if item["label"] == checkpoint)
    if len(jax.devices("gpu")) != 1:
        raise ValueError("A policy process requires exactly one visible GPU")
    policy = create_trained_policy(
        get_config(manifest["policy_config"]),
        Path(item["path"]),
        seed=0,
        sample_kwargs={"num_steps": manifest["denoising_steps"]},
    )
    horizon = policy._model.action_horizon  # noqa: SLF001 - original policy has no public horizon accessor
    if manifest["execute_steps"] > horizon:
        raise ValueError("execute-steps exceeds the model action horizon")
    memory = policy.config
    representation = memory.representation_type if memory is not None else None
    symbolic_type = memory.symbolic_memory.type if representation == "symbolic" else None
    if (representation == "symbolic") != (manifest["subgoal_source"] != "none"):
        raise ValueError("Symbolic checkpoints require an explicit subgoal source; other checkpoints use none")
    if manifest["subgoal_type"] and manifest["subgoal_type"] != symbolic_type:
        raise ValueError("subgoal-type differs from the checkpoint history configuration")
    encode = policy.mem_buffer.vision_enc if policy.mem_buffer is not None else None
    vision_batch = manifest["vision_batch_size"]

    def encode_frames(images):
        if images.shape[0] in (1, vision_batch):
            return encode(images)
        outputs = []
        for start in range(0, len(images), vision_batch):
            chunk = images[start : start + vision_batch]
            count = len(chunk)
            if count < vision_batch:
                chunk = jnp.concatenate((chunk, jnp.repeat(chunk[-1:], vision_batch - count, axis=0)))
            outputs.append(encode(chunk)[:count])
        return jnp.concatenate(outputs, axis=0)

    token = bytes.fromhex(os.environ["ROBOMME_EVAL_TOKEN"])
    with Listener(address, family="AF_UNIX", authkey=token) as listener, listener.accept() as connection:
        connection.send_bytes(
            msgpack_numpy.packb({"horizon": horizon, "representation": representation, "symbolic_type": symbolic_type})
        )
        while True:
            try:
                request = msgpack_numpy.unpackb(connection.recv_bytes())
            except EOFError:
                return
            try:
                if "reset" in request:
                    policy._seed = int(request["reset"])  # noqa: SLF001 - preserve original per-episode reset protocol
                    policy.reset()
                    if encode is not None:
                        policy.mem_buffer.vision_enc = encode_frames
                    result = {"reset": True}
                else:
                    policy.add_buffer(request["history"])
                    actions = policy.infer(request["observation"])["actions"]
                    if actions.shape != (horizon, 8) or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid actions with shape {actions.shape}")
                    result = {"actions": actions}
                connection.send_bytes(msgpack_numpy.packb(result))
            except Exception:
                connection.send_bytes(msgpack_numpy.packb({"error": traceback.format_exc()}))
                raise


class PolicyClient:
    def __init__(self, root, manifest, checkpoint, gpu, stream):
        runtime = manifest["runtime"]
        work = Path(runtime["work_dir"])
        self.address = str(work / f"{manifest['run_id'][:12]}-{gpu}-{stream}.sock")
        if len(self.address.encode()) >= 104:
            raise ValueError("work-dir is too long for a Unix socket path")
        Path(self.address).unlink(missing_ok=True)
        self.connection = None
        token = secrets.token_bytes(32)
        env = dict(
            os.environ,
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            JAX_COMPILATION_CACHE_DIR=str(work / "jax_cache"),
            ROBOMME_EVAL_TOKEN=token.hex(),
            PYTHONPATH=f"{REPO}/src:{REPO}/packages/openpi-client/src",
            HF_ENDPOINT=manifest["hf_endpoint"],
        )
        if runtime["openpi_data_home"]:
            env["OPENPI_DATA_HOME"] = runtime["openpi_data_home"]
        log = root / f"gpu{gpu}-stream{stream}-policy.log"
        with log.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                [
                    runtime["model_python"],
                    "-u",
                    str(SCRIPT),
                    "_policy",
                    "--output-dir",
                    str(root),
                    "--checkpoint-label",
                    checkpoint,
                    "--socket",
                    self.address,
                ],
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        try:
            deadline = time.monotonic() + 600
            while not Path(self.address).exists():
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Policy startup failed; see {log}")
                time.sleep(0.5)
            self.connection = Client(self.address, family="AF_UNIX", authkey=token)
            self.metadata = self.receive()
        except BaseException:
            self.close()
            raise

    def receive(self):
        from openpi_client import msgpack_numpy

        if not self.connection.poll(600):
            raise TimeoutError("Policy response exceeded 600 seconds")
        result = msgpack_numpy.unpackb(self.connection.recv_bytes())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def call(self, request):
        from openpi_client import msgpack_numpy

        self.connection.send_bytes(msgpack_numpy.packb(request))
        return self.receive()

    def close(self):
        if self.connection is not None:
            self.connection.close()
        try:
            self.process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        Path(self.address).unlink(missing_ok=True)


def predictor_for(manifest, metadata, root):
    from subgoal_predictor import build_subgoal_predictor

    source = manifest["subgoal_source"]
    adapter = manifest["subgoal_adapter"]
    args = SimpleNamespace(
        use_oracle=source == "oracle",
        use_qwenvl=source == "qwenvl",
        use_memer=source == "memer",
        use_gemini=source == "gemini",
        subgoal_type=metadata["symbolic_type"],
        gemini_model_name=manifest["gemini_model"],
        qwenvl_simpleSG_adapter_path=adapter,
        qwenvl_groundSG_adapter_path=adapter,
        memer_adapter_path=adapter,
    )
    return build_subgoal_predictor(args, root / "subgoals")


def rollout(client, runner, predictor, seed, episode, manifest):
    from unittest.mock import patch

    import gymnasium as gym
    from utils import EpisodeState
    from utils import pack_buffer

    if not client.call({"reset": seed}).get("reset"):
        raise RuntimeError("Policy reset was not acknowledged")
    make = gym.make

    def make_on_local_gpu(*args, **kwargs):
        return make(*args, **kwargs, render_backend="cuda:0")

    try:
        with patch.object(gym, "make", make_on_local_gpu):
            runner.make_env(episode)
        initial = runner.get_init_obs()
        state = EpisodeState()
        state.image_buffer.extend(initial["images"])
        state.wrist_image_buffer.extend(initial["wrist_images"])
        state.state_buffer.extend(initial["states"])
        state.exec_start_idx = len(state.image_buffer) - 1
        predictor.start_episode(state, runner)
        subgoal = last_subgoal = None
        outcome = "timeout"
        for count in range(1, manifest["max_steps"] + 1):  # noqa: B007 - final count is returned after the loop
            predictor.step(state)
            if not state.action_plan:
                if state.count % manifest["subgoal_keep_period"] == 0 or last_subgoal is None:
                    subgoal, api_error = predictor.get_subgoal(state.count, subgoal, last_subgoal)
                    if api_error:
                        raise RuntimeError("Subgoal provider returned an API error")
                else:
                    subgoal = last_subgoal
                front, wrist, joints = state.get_current_obs()
                observation = {
                    "observation/image": front,
                    "observation/wrist_image": wrist,
                    "observation/state": joints,
                    "prompt": initial["task_goal"],
                }
                if subgoal is not None:
                    observation.update(simple_subgoal=subgoal, grounded_subgoal=subgoal)
                response = client.call(
                    {
                        "history": pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx),
                        "observation": observation,
                    }
                )
                state.action_plan.extend(response["actions"][: manifest["execute_steps"]])
                state.clear_buffers()
                last_subgoal = subgoal
            obs, stop, outcome = runner.step(state.action_plan.popleft())
            state.count += 1
            if outcome == "error" or (stop and outcome not in ("success", "fail", "timeout")):
                raise RuntimeError(f"Invalid simulator outcome {outcome!r}, stop={stop}")
            if stop:
                break
            state.add_observation(*obs)
        else:
            outcome = "timeout"
        predictor.end_episode(state, outcome)
        return outcome, count, str(runner.difficulty)
    finally:
        runner.close_env()


def worker(root, gpu, stream):
    from env_runner import EnvRunner
    import torch

    manifest = load_manifest(root)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise ValueError("A rollout worker requires exactly one visible GPU")
    runners = {task: EnvRunner(task, str(root), max_steps=manifest["max_steps"]) for task in manifest["tasks"]}
    for task, runner in runners.items():
        if runner.num_episodes != 50:
            raise ValueError(f"Official test task must contain 50 episodes: {task}")
        for episode in manifest["episode_ids"]:
            if runner.env_builder.resolve_episode(episode)[0] is None:
                raise ValueError(f"Missing official scene seed: {task}/{episode}")
    client = None
    loaded = None
    try:
        with closing(db_connect(root)) as db:
            while (job := claim(db, gpu, stream)) is not None:
                checkpoint, seed, task, episode = job
                try:
                    if checkpoint != loaded:
                        if client is not None:
                            client.close()
                        client = PolicyClient(root, manifest, checkpoint, gpu, stream)
                        predictor = predictor_for(manifest, client.metadata, root / f"gpu{gpu}-stream{stream}")
                        loaded = checkpoint
                    runner = runners[task]
                    environment_seed, _ = runner.env_builder.resolve_episode(episode)
                    outcome, steps, difficulty = rollout(client, runner, predictor, seed, episode, manifest)
                    with db:
                        db.execute(
                            """UPDATE episodes SET status='complete',finished=?,outcome=?,steps=?,env_seed=?,difficulty=?
                            WHERE (checkpoint,seed,task,episode)=(?,?,?,?)""",
                            (time.time(), outcome, steps, environment_seed, difficulty, *job),
                        )
                    print(now(), "EPISODE", gpu, stream, *job, outcome, steps, flush=True)
                except Exception:
                    with db:
                        db.execute(
                            "UPDATE episodes SET error=? WHERE (checkpoint,seed,task,episode)=(?,?,?,?)",
                            (traceback.format_exc()[-6000:], *job),
                        )
                    raise
    finally:
        if client is not None:
            client.close()


def launch_worker(root, manifest, gpu, stream):
    runtime = manifest["runtime"]
    work = Path(runtime["work_dir"])
    for directory in (work, work / "sapien_cache", work / "tmp", work / "xdg_cache"):
        directory.mkdir(parents=True, exist_ok=True)
    paths = [
        str(Path(runtime["benchmark_dir"]) / "src"),
        str(REPO / "examples/robomme"),
        str(REPO / "packages/openpi-client/src"),
    ]
    if runtime["maniskill_dir"]:
        paths.append(runtime["maniskill_dir"])
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONPATH=os.pathsep.join(paths),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_ENDPOINT=manifest["hf_endpoint"],
        SAPIEN_CACHE_DIR=str(work / "sapien_cache"),
        TMPDIR=str(work / "tmp"),
        XDG_CACHE_HOME=str(work / "xdg_cache"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if runtime["asset_dir"]:
        env["MS_ASSET_DIR"] = runtime["asset_dir"]
    if runtime["graphics_dir"]:
        graphics = Path(runtime["graphics_dir"])
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(graphics / "10_nvidia.json")
        env["VK_ICD_FILENAMES"] = str(graphics / "nvidia_icd.json")
    with (root / f"gpu{gpu}-stream{stream}-worker.log").open("ab", buffering=0) as output:
        return subprocess.Popen(
            [
                runtime["sim_python"],
                "-u",
                str(SCRIPT),
                "_worker",
                "--output-dir",
                str(root),
                "--gpu",
                str(gpu),
                "--stream",
                str(stream),
            ],
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_worker(process):
    # Also reap the policy subprocess when its rollout worker has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def supervise(root, manifest):
    slots = [(gpu, stream) for stream in range(manifest["streams_per_gpu"]) for gpu in manifest["gpus"]]
    workers = {}
    failures = dict.fromkeys(slots, 0)
    for gpu, stream in slots:
        release_interrupted(root, gpu, stream, manifest["max_attempts"])
    try:
        while True:
            rows = rows_from_database(root)
            validate_rows(rows, manifest, require_complete=False)
            checkpoint_reports(root, manifest, rows)
            unfinished = [row for row in rows if row["status"] != "complete"]
            order = [item["label"] for item in manifest["checkpoints"]]
            stage = next((label for label in order if any(row["checkpoint"] == label for row in unfinished)), None)
            pending = any(row["status"] == "pending" and row["checkpoint"] == stage for row in unfinished)
            for slot in slots:
                process = workers.get(slot)
                active = [
                    row for row in unfinished if (row["gpu"], row["stream"]) == slot and row["status"] == "running"
                ]
                if process is not None:
                    if active and time.time() - active[0]["started"] > manifest["episode_timeout"]:
                        print(now(), "EPISODE_TIMEOUT", slot, flush=True)
                        stop_worker(process)
                    if process.poll() is not None:
                        stop_worker(process)
                        if process.returncode:
                            failures[slot] += 1
                        release_interrupted(root, *slot, manifest["max_attempts"])
                        del workers[slot]
                if slot not in workers and pending and failures[slot] < 5:
                    workers[slot] = launch_worker(root, manifest, *slot)
            status = status_snapshot(root)
            status.update(
                state="running",
                supervisor_pid=os.getpid(),
                active_checkpoint=stage,
                model_batch_size=1,
                streams_per_gpu=manifest["streams_per_gpu"],
                worker_pids={f"{gpu}:{stream}": process.pid for (gpu, stream), process in workers.items()},
                worker_failures={f"{gpu}:{stream}": value for (gpu, stream), value in failures.items()},
            )
            write_json(root / "status.json", status)
            if not workers:
                refreshed = rows_from_database(root)
                remaining = [row for row in refreshed if row["status"] != "complete"]
                stage = next((label for label in order if any(row["checkpoint"] == label for row in remaining)), None)
                if not any(row["status"] == "pending" and row["checkpoint"] == stage for row in remaining):
                    break
                if all(value >= 5 for value in failures.values()):
                    break
            time.sleep(5)
        status = status_snapshot(root)
        complete = status["completed"] == status["expected"]
        if complete:
            report(root, manifest)
        write_json(root / "status.json", {**status, "state": "complete" if complete else "needs_attention"})
        return 0 if complete else 1
    finally:
        for process in workers.values():
            stop_worker(process)


def verify_resume(manifest):
    if manifest["schema_version"] != 1:
        raise ValueError("Legacy runs support status/report; use a new output directory for unified evaluation")
    if sha256(SCRIPT) != manifest["runner_sha256"]:
        raise ValueError("Evaluation runner changed since this run was created")
    for name, digest in manifest["source_sha256"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Evaluation source changed since this run was created: {name}")
    for checkpoint in manifest["checkpoints"]:
        value = checkpoint_description(f"{checkpoint['label']}={checkpoint['path']}")
        if value != checkpoint:
            raise ValueError(f"Checkpoint identity changed: {checkpoint['label']}")
    runtime = manifest["runtime"]
    if source_identity(Path(runtime["benchmark_dir"]) / "src/robomme") != manifest["benchmark_source_sha256"]:
        raise ValueError("Benchmark source changed since this run was created")
    if runtime["maniskill_dir"] and (
        source_identity(Path(runtime["maniskill_dir"]) / "mani_skill") != manifest["maniskill_source_sha256"]
    ):
        raise ValueError("ManiSkill source changed since this run was created")


def run(args):
    root = args.output_dir.resolve()
    if args.resume:
        if args.checkpoint or args.runtime_config:
            raise ValueError("Resume reads the original manifest; supply only --output-dir and --resume")
        manifest = load_manifest(root)
        verify_resume(manifest)
    else:
        if not args.checkpoint:
            raise ValueError("At least one --checkpoint is required for a new run")
        manifest = make_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            # Retired child processes must finish before claiming their interrupted rows.
            previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
            for pid in previous.get("worker_pids", {}).values():
                command = Path(f"/proc/{pid}/cmdline")
                if command.exists() and str(root).encode() in command.read_bytes():
                    raise ValueError(f"A prior rollout worker is still alive: {pid}")
        else:
            if (root / "manifest.json").exists() or (root / "episodes.sqlite3").exists():
                raise ValueError("Output directory contains a run; use --resume or choose a new directory")
            initialize_database(root, manifest)
            write_json(root / "manifest.json", manifest)
        with (root / "scheduler_history.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "started": now(),
                        "pid": os.getpid(),
                        "resuming": args.resume,
                        "gpus": manifest["gpus"],
                        "streams_per_gpu": manifest["streams_per_gpu"],
                    }
                )
                + "\n"
            )
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        return supervise(root, manifest)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run", help="Run a new checkpoint/seed matrix or resume it")
    start.add_argument(
        "--checkpoint", "--ckpt", action="append", help="Checkpoint directory, optionally LABEL=PATH; repeatable"
    )
    start.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0, 42, 7])
    start.add_argument("--variant", help="Name shown in the report")
    start.add_argument("--policy-config", default="mme_vla_suite")
    start.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    start.add_argument("--streams-per-gpu", type=int, default=2)
    start.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    start.add_argument("--episode-ids", type=int, nargs="+", default=list(range(50)))
    start.add_argument("--max-steps", type=int, default=1300)
    start.add_argument("--execute-steps", type=int, default=16)
    start.add_argument("--denoising-steps", type=int, default=10)
    start.add_argument("--vision-batch-size", type=int, default=16)
    start.add_argument("--subgoal-source", choices=("none", "oracle", "qwenvl", "memer", "gemini"), default="none")
    start.add_argument("--subgoal-type", choices=("simple_subgoal", "grounded_subgoal"))
    start.add_argument("--subgoal-adapter")
    start.add_argument("--subgoal-keep-period", type=int, default=1)
    start.add_argument("--gemini-model", default="gemini-2.5-pro")
    start.add_argument("--max-attempts", type=int, default=3)
    start.add_argument("--episode-timeout", type=int, default=900)
    start.add_argument("--reference-percent", type=float)
    start.add_argument("--runtime-config", type=Path)
    for name in RUNTIME_PATHS:
        start.add_argument("--" + name.replace("_", "-"))
    start.add_argument("--resume", action="store_true")
    start.add_argument(
        "--dry-run", action="store_true", help="Validate and print the resolved manifest without starting workers"
    )
    commands.add_parser("status", help="Read progress without changing a run")
    commands.add_parser("report", help="Validate coverage and regenerate reports")
    internal_worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    internal_worker.add_argument("--gpu", type=int, required=True)
    internal_worker.add_argument("--stream", type=int, required=True)
    internal_policy = commands.add_parser("_policy", help=argparse.SUPPRESS)
    internal_policy.add_argument("--checkpoint-label", required=True)
    internal_policy.add_argument("--socket", required=True)
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    return cli


def main(argv=None):
    cli = parser()
    argv = sys.argv[1:] if argv is None else argv
    args = cli.parse_args(argv)
    if args.command == "run" and args.resume:
        supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        overrides = supplied - {"--resume", "--output-dir", "--dry-run"}
        if overrides:
            cli.error("Resume uses the saved manifest; remove these overrides: " + ", ".join(sorted(overrides)))
    root = args.output_dir.resolve()
    try:
        if args.command == "run":
            return run(args)
        if args.command == "status":
            manifest = load_manifest(root)
            rows = rows_from_database(root)
            validate_rows(rows, manifest, require_complete=False)
            print(json.dumps(status_snapshot(root), indent=2, ensure_ascii=False))
        elif args.command == "report":
            result = report(root, load_manifest(root))
            print(json.dumps(result["overall"], indent=2))
        elif args.command == "_worker":
            worker(root, args.gpu, args.stream)
        else:
            policy_process(root, args.checkpoint_label, args.socket)
    except (ValueError, FileNotFoundError) as error:
        cli.error(str(error))
    return 0


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
