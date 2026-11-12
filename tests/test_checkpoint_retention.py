"""Exercise real Orbax retention on tiny CPU checkpoints."""

# Retain the existing standard-library unittest runner used by the training tests.
# ruff: noqa: PT009, PT027

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from openpi.training.checkpoints import initialize_checkpoint_dir

PROJECT = Path(__file__).resolve().parents[1]
SAVE_STEPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 79999]


class CheckpointRetentionTest(unittest.TestCase):
    def check_policy(self, keep_period, expected, **options):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoints"
            manager, resuming = initialize_checkpoint_dir(
                root,
                keep_period=keep_period,
                overwrite=False,
                resume=False,
                **options,
            )
            self.assertFalse(resuming)
            try:
                for index, step in enumerate(SAVE_STEPS):
                    saved = manager.save(
                        step,
                        {
                            "assets": lambda directory: (directory / "marker.json").write_text("{}"),
                            "params": {"params": np.array([step], dtype=np.int32)},
                        },
                    )
                    self.assertTrue(saved)
                    manager.wait_until_finished()
                    if keep_period is None:
                        self.assertEqual(sorted(manager.all_steps()), SAVE_STEPS[: index + 1][-3:])
                self.assertEqual(sorted(manager.all_steps()), expected)
            finally:
                manager.close()
            self.assertEqual(sorted(int(p.name) for p in root.iterdir() if p.name.isdigit()), expected)

    def test_batch_ablation_rotates_to_final_three_real_checkpoints(self):
        self.check_policy(None, [60000, 70000, 79999], max_to_keep=3)

    def test_original_periodic_retention_remains_unchanged(self):
        self.check_policy(10000, SAVE_STEPS)

    def test_invalid_retention_is_rejected(self):
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "positive"):
            initialize_checkpoint_dir(
                Path(temporary) / "checkpoints",
                keep_period=None,
                overwrite=False,
                resume=False,
                max_to_keep=0,
            )

    def test_new_profile_keeps_training_hyperparameters_except_batch(self):
        directory = PROJECT / "configs/train"
        baseline = yaml.safe_load((directory / "robomme_framesamp_modul_repro.yaml").read_text())
        ablation = yaml.safe_load((directory / "robomme_framesamp_modul_bs32.yaml").read_text())
        for section in ("model", "expected_memory", "freeze", "learning_rate", "optimizer", "weights"):
            self.assertEqual(ablation[section], baseline[section], section)
        allowed = {"project_name", "exp_name", "batch_size", "keep_period", "assets_base_dir", "checkpoint_base_dir"}
        for key, value in baseline["train"].items():
            if key not in allowed:
                self.assertEqual(ablation["train"][key], value, key)
        self.assertEqual(ablation["data"]["dataset_path"], baseline["data"]["dataset_path"])
        self.assertEqual(ablation["train"]["batch_size"], baseline["train"]["batch_size"] // 2)
        self.assertEqual(ablation["train"]["checkpoint_max_to_keep"], 3)
        self.assertIsNone(ablation["train"]["keep_period"])
        self.assertEqual(ablation["train"]["save_interval"], 10000)
        self.assertEqual(ablation["train"]["num_train_steps"], 80000)
        for value in (
            ablation["runtime"]["compilation_cache_dir"],
            ablation["runtime"]["openpi_data_home"],
            ablation["train"]["assets_base_dir"],
            ablation["train"]["checkpoint_base_dir"],
        ):
            self.assertTrue(value.startswith("/mnt/cpfs-standard/"), value)


class ProfileRuntimeTest(unittest.TestCase):
    @staticmethod
    def runner():
        spec = importlib.util.spec_from_file_location(
            "training_profile_runtime", PROJECT / "scripts/run_training_profile.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def runtime():
        profile = yaml.safe_load((PROJECT / "configs/train/robomme_framesamp_modul_bs32.yaml").read_text())
        return profile["runtime"]

    def test_review_supports_both_four_gpu_groups_without_enabling_jax_gpu(self):
        for devices in ([0, 1, 2, 3], [4, 5, 6, 7]):
            runtime = {**self.runtime(), "cuda_visible_devices": devices}
            with self.subTest(devices=devices), patch.dict(os.environ, {}, clear=True):
                self.runner().configure_environment(runtime, execute=False)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], ",".join(map(str, devices)))
                self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")

    def test_execute_limits_visibility_to_the_last_four_gpus(self):
        with patch.dict(os.environ, {}, clear=True):
            self.runner().configure_environment(self.runtime(), execute=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda")

    def test_invalid_gpu_lists_are_rejected(self):
        for devices in ([], [4, 5, 6], [4, 5, 6, 6], [4, 5, 6, -1], [4, 5, 6, "7"], [4, 5, 6, True]):
            runtime = {**self.runtime(), "cuda_visible_devices": devices}
            with self.subTest(devices=devices), self.assertRaisesRegex(ValueError, "four distinct"):
                self.runner().configure_environment(runtime, execute=False)


if __name__ == "__main__":
    unittest.main()
