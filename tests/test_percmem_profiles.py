"""The P0 budget matches the active reproduction; review never constructs a model."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from etils import epath
from flax import nnx
import jax.numpy as jnp
import optax
import yaml

from mme_vla_suite.training.percmem_profile import build_config, review_config
from openpi.training import checkpoints
from openpi.training.utils import TrainState


ROOT = Path(__file__).resolve().parents[1]


class ProfilesTest(unittest.TestCase):
    def test_teacher_matches_reference_training_budget(self):
        reference = yaml.safe_load((ROOT / "configs/train/robomme_framesamp_modul_repro.yaml").read_text())
        teacher = yaml.safe_load((ROOT / "configs/train/robomme_percmem_v1_teacher_repro.yaml").read_text())
        for key in ["learning_rate", "optimizer", "evaluation"]:
            self.assertEqual(teacher[key], reference[key])
        for key in ["seed", "batch_size", "num_workers", "fsdp_devices", "num_train_steps", "ema_decay",
                    "log_interval", "save_interval", "keep_period", "overwrite", "resume", "wandb_enabled"]:
            self.assertEqual(teacher["train"][key], reference["train"][key])
        for key in reference["model"]:
            if key != "history_config":
                self.assertEqual(teacher["model"][key], reference["model"][key])
        config, memory = build_config(teacher)
        with patch("mme_vla_suite.models.integration.history_pi0.HistoryPi0Config.create",
                   side_effect=AssertionError("Review must not construct a model")):
            report = review_config(teacher, config, memory)
        self.assertFalse(report["training_started"])
        self.assertFalse(report["capacity_tested"])

    def test_student_uses_ten_percent_budget_and_frozen_feature_space(self):
        teacher = yaml.safe_load((ROOT / "configs/train/robomme_percmem_v1_teacher_repro.yaml").read_text())
        student = yaml.safe_load((ROOT / "configs/train/robomme_percmem_v1_student.yaml").read_text())
        self.assertEqual(student["train"]["num_train_steps"], teacher["train"]["num_train_steps"] // 10)
        self.assertAlmostEqual(student["learning_rate"]["value"], teacher["learning_rate"]["value"] / 10)
        _, memory = build_config(student)
        self.assertEqual(memory.perc_mem.input_space, "projected")
        self.assertEqual(memory.perc_mem.distill_weight, 1.0)

    def test_full_train_state_is_actually_written_for_new_runs(self):
        class Small(nnx.Module):
            def __init__(self):
                self.weight = nnx.Param(jnp.ones((2, 2)))
        model = Small()
        params = nnx.state(model)
        tx = optax.adam(1e-3)
        state = TrainState(step=1, params=params, model_def=nnx.graphdef(model), tx=tx,
                           opt_state=tx.init(params), ema_decay=.999, ema_params=params)
        loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
        with TemporaryDirectory() as temporary:
            directory = epath.Path(temporary) / "checkpoint"
            manager, _ = checkpoints.initialize_checkpoint_dir(directory, keep_period=1, overwrite=False,
                                                                resume=False, save_train_state=True)
            try:
                checkpoints.save_state(manager, state, loader, 1, save_train_state=True)
                manager.wait_until_finished()
                self.assertTrue((directory / "1/train_state/_METADATA").exists())
                self.assertTrue((directory / "1/params/_METADATA").exists())
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
