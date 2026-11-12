"""Run PyTorch policy inference through the existing RoboMME controller and simulator."""

from multiprocessing.connection import Listener
import os
from pathlib import Path
import signal
import sys
import traceback

import eval_robomme as evaluation


def pytorch_policy(root, checkpoint, address):
    # JAX is used only for the released CPU-side history/position transforms and seeded noise.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    from openpi_client import msgpack_numpy
    import torch

    from mme_vla_suite.policies.framesamp_pytorch import create_eval_policy
    from mme_vla_suite.training.config import get_config

    if torch.cuda.device_count() != 1:
        raise ValueError("A PyTorch policy worker requires exactly one visible GPU")
    manifest = evaluation.load_manifest(root)
    item = next(c for c in manifest["checkpoints"] if c["label"] == checkpoint)
    policy = create_eval_policy(
        get_config(manifest["policy_config"]),
        Path(item["path"]),
        0,
        manifest["denoising_steps"],
        manifest["vision_batch_size"],
    )
    horizon = policy._model.action_horizon
    with Listener(address, family="AF_UNIX", authkey=bytes.fromhex(os.environ["ROBOMME_EVAL_TOKEN"])) as listener:
        with listener.accept() as connection:
            connection.send_bytes(
                msgpack_numpy.packb({"horizon": horizon, "representation": "perceptual", "symbolic_type": None})
            )
            while True:
                try:
                    request = msgpack_numpy.unpackb(connection.recv_bytes())
                except EOFError:
                    return
                try:
                    if "reset" in request:
                        policy._seed = int(request["reset"])
                        policy.reset()
                        response = {"reset": True}
                    else:
                        policy.add_buffer(request["history"])
                        actions = policy.infer(request["observation"])["actions"]
                        if actions.shape != (horizon, 8) or not np.isfinite(actions).all():
                            raise ValueError("Invalid PyTorch policy actions")
                        response = {"actions": actions}
                    connection.send_bytes(msgpack_numpy.packb(response))
                except Exception:
                    connection.send_bytes(msgpack_numpy.packb({"error": traceback.format_exc()}))
                    raise


make_reference_manifest = evaluation.make_manifest


def pytorch_manifest(args):
    if not args.model_python:
        args.model_python = sys.executable
    manifest = make_reference_manifest(args)
    if (
        any(c["history_config"] != "perceptual-framesamp-modul.yaml" for c in manifest["checkpoints"])
        or manifest["execute_steps"] > 20
        or manifest["subgoal_source"] != "none"
        or manifest["subgoal_type"] is not None
    ):
        raise ValueError("Expected RoboMME FrameSamp+Modul, no subgoal source, and at most 20 actions per plan")
    manifest["policy_backend"] = "openpi_pytorch"
    for path in (
        "scripts/eval_robomme.py",
        "src/mme_vla_suite/models/framesamp_pytorch.py",
        "src/mme_vla_suite/training/framesamp_weights.py",
        "src/mme_vla_suite/policies/framesamp_pytorch.py",
        "src/openpi/models_pytorch/gemma_pytorch.py",
    ):
        manifest["source_sha256"][path] = evaluation.sha256(evaluation.REPO / path)
    return manifest


if __name__ == "__main__":
    # The controller starts all subprocesses via SCRIPT. Give this model backend its own
    # runner identity, so existing JAX manifests and resume checks stay valid.
    evaluation.SCRIPT = Path(__file__).resolve()
    evaluation.policy_process = pytorch_policy
    evaluation.make_manifest = pytorch_manifest

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(evaluation.main())
