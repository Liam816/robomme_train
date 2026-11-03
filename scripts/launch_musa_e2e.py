"""Detached preflight followed by the authorized 80k E2E training run."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("ROBOMME_MUSA_ROOT", "/data/heping/robomme_musa/e2e"))
    cli = argparse.ArgumentParser()
    cli.add_argument("--run-name", default="e2e_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    args = cli.parse_args()
    root.mkdir(exist_ok=True, parents=True)
    lock = (root / "training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool = set(range(64, 96)) & os.sched_getaffinity(0)
    if len(pool) != 32:
        raise RuntimeError("Expected the 32 CPU cores local to physical GPUs 4-7")
    os.sched_setaffinity(0, pool)
    launcher = repo / "scripts/run_musa_percmem_e2e.sh"
    source = repo / "configs/train/robomme_percmem_v1_e2e_repro.yaml"
    git = lambda *a: subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()
    original = subprocess.check_output(["git", "-C", str(repo), "show", "b4432b1ef2ab746b13eec32f25bcc5862c582164:" + str(source.relative_to(repo))])
    if source.read_bytes() != original:
        raise RuntimeError("The requested experiment profile changed from b4432b1")
    load = lambda name: json.loads((root / name).read_text())
    for name in ["numerical_cpu_validation.json", "e2e_validation_cpu.json", "e2e_validation_musa.json", "production_validation.json"]:
        if not load(name).get("passed"):
            raise RuntimeError("Validation has not passed: " + name)
    if not load('precision_diagnostic.json').get('passed'):
        raise RuntimeError('Original production FP16 warmup behavior must be checked')
    for name in ['e2e_validation_cpu.json','e2e_validation_musa.json']:
        if load(name)['controlled_optimizer']['scorer_parameter_delta_l2']<=0:
            raise RuntimeError('Controlled E2E scorer update validation failed')
    if not load("production_validation.json")["sample"]["overflow_exercised"]:
        raise RuntimeError("Production validation must exercise real evictions")
    if load("assets/export_complete.json")["memory_config"]["stage"] != "e2e":
        raise RuntimeError("E2E asset initialization is required")
    pids = [2531277, 2531278, 3238230, 3238231]
    original_alive = {str(p): Path(f"/proc/{p}").exists() for p in pids}
    logs = root / "logs" / args.run_name
    logs.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "process_group": os.getpgrp(), "worktree": str(repo), "branch": git("branch", "--show-current"),
             "commit": git("rev-parse", "HEAD"), "source_commit": "b4432b1ef2ab746b13eec32f25bcc5862c582164",
             "source_profile": str(source), "source_profile_sha256": hashlib.sha256(original).hexdigest(),
             "run_name": args.run_name, "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "physical_gpus": [4, 5, 6, 7], "target_optimizer_updates": 80000, "original_task_pids": pids,
             "original_tasks_alive_at_launch": original_alive,
             "train_log": str(logs / "train.log"), "preflight_log": str(logs / "preflight.log")}
    def update(**values):
        state.update(values)
        tmp = root / "training.status.json.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(root / "training.status.json")
    def run(stage, flags, path):
        command = ["bash", str(launcher), "train", *flags]
        with path.open("a") as log:
            log.write("\nLAUNCH " + datetime.now(ZoneInfo("Asia/Shanghai")).isoformat() + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid, command=command)
            code = child.wait()
            if code:
                raise RuntimeError(stage + " exited " + str(code))
    try:
        update(stage="preflight_starting")
        preflight = args.run_name + "_preflight"
        run("preflight", ["--steps", "3", "--run-name", preflight], logs / "preflight.log")
        folder = root / "runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42"
        result = json.loads((folder / preflight / "status.json").read_text())
        if result["stage"] != "complete" or not result["checkpoint_reload_fixed_input_identical"] or not result["frozen_hashes_unchanged"]:
            raise RuntimeError("Four-card preflight/checkpoint validation failed")
        update(preflight_result=result, train_status=str(folder / args.run_name / "status.json"),
               precision_diagnostic=str(root/'precision_diagnostic.json'),
               warmup_note='Original zero reader initialization and FP16 storage preserved; initial scorer gradients can be zero.')
        # Begin the full experiment from the requested fresh official-base seed.
        run("training", ["--steps", "80000", "--run-name", args.run_name], logs / "train.log")
        update(stage="complete", finished_at=time.time())
    except BaseException as error:
        update(stage="failed", error=repr(error), finished_at=time.time())
        raise


if __name__ == "__main__":
    main()
