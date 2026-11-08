"""CPU-only protocol checks; no model, simulator, or GPU allocation."""

# Keep the standard-library runner usable in both Python environments without pytest.
# ruff: noqa: PT009, PT027

from collections import deque
from contextlib import closing
from contextlib import redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("eval_robomme", Path(__file__).parents[1] / "scripts/eval_robomme.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            "schema_version": 1,
            "variant": "another-variant",
            "checkpoints": [{"label": "79999", "path": "/weights/79999"}, {"label": "60000", "path": "/weights/60000"}],
            "seeds": [31, 7],
            "tasks": ["BinFill", "MoveCube"],
            "episode_ids": [0, 2],
            "max_steps": 1300,
            "execute_steps": 16,
            "reference_percent": None,
            "gpus": [0],
            "streams_per_gpu": 1,
            "max_attempts": 2,
            "episode_timeout": 900,
        }

    def database(self):
        evaluation.initialize_database(self.root, self.manifest)

    def completed(self):
        return [
            {
                "checkpoint": label,
                "seed": seed,
                "task": task,
                "episode": episode,
                "status": "complete",
                "outcome": "success" if episode == 0 else "timeout",
                "env_seed": episode + 100,
                "steps": 1 if episode == 0 else 1300,
            }
            for label, seed, task, episode in sorted(evaluation.expected_keys(self.manifest))
        ]

    def test_arbitrary_checkpoint_paths_seeds_and_gpu_arguments(self):
        args = evaluation.parser().parse_args(
            [
                "run",
                "--output-dir",
                str(self.root),
                "--checkpoint",
                "token=/a/79999",
                "--checkpoint",
                "/b/69999",
                "--seeds",
                "123",
                "456",
                "--gpus",
                "0",
                "1",
                "2",
                "3",
            ]
        )
        self.assertEqual(args.checkpoint, ["token=/a/79999", "/b/69999"])
        self.assertEqual(args.seeds, [123, 456])
        self.assertEqual(args.gpus, [0, 1, 2, 3])

    def test_checkpoint_requires_finalized_orbax_and_tracks_history(self):
        checkpoint = self.root / "79999"
        for name in ("params/_METADATA", "assets/robomme/norm_stats.json", "_CHECKPOINT_METADATA"):
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "not finalized"):
            evaluation.checkpoint_description(str(checkpoint))
        (checkpoint / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 123}')
        (self.root / "history_config.txt").write_text("recurrent-rmt-modul.yaml")
        value = evaluation.checkpoint_description(f"rmt={checkpoint}")
        self.assertEqual(value["label"], "rmt")
        self.assertEqual(value["history_config"], "recurrent-rmt-modul.yaml")

    def test_python_symlink_preserves_virtual_environment(self):
        interpreter = self.root / "venv-python"
        interpreter.symlink_to("/usr/bin/python3")
        args = evaluation.parser().parse_args(
            ["run", "--output-dir", str(self.root), "--model-python", str(interpreter)]
        )
        self.assertEqual(evaluation.runtime_options(args)["model_python"], str(interpreter))

    def test_checkpoint_and_seed_order_are_strict(self):
        self.database()
        claimed = []
        with closing(evaluation.db_connect(self.root)) as db:
            while (row := evaluation.claim(db, 0, 0)) is not None:
                claimed.append(tuple(row))
                with db:
                    db.execute(
                        "UPDATE episodes SET status='complete' WHERE (checkpoint,seed,task,episode)=(?,?,?,?)", row
                    )
        self.assertEqual([row[0] for row in claimed[:8]], ["79999"] * 8)
        self.assertEqual([row[0] for row in claimed[8:]], ["60000"] * 8)
        self.assertEqual([row[1] for row in claimed[:4]], [31] * 4)

    def test_next_checkpoint_waits_for_all_inflight_episodes(self):
        self.database()
        with closing(evaluation.db_connect(self.root)) as db:
            for _ in range(8):
                self.assertIsNotNone(evaluation.claim(db, 0, 0))
            self.assertIsNone(evaluation.claim(db, 0, 1))

    def test_resume_requeues_interrupted_without_double_counting(self):
        self.database()
        with closing(evaluation.db_connect(self.root)) as db:
            original = tuple(evaluation.claim(db, 0, 0))
            evaluation.release_interrupted(self.root, 0, 0, 2)
            self.assertEqual(tuple(evaluation.claim(db, 0, 0)), original)
        self.assertEqual(len(evaluation.rows_from_database(self.root)), 16)

    def test_attempt_limit_blocks_failed_checkpoint(self):
        self.database()
        with closing(evaluation.db_connect(self.root)) as db:
            for _ in range(2):
                evaluation.claim(db, 0, 0)
                evaluation.release_interrupted(self.root, 0, 0, 2)
            rows = evaluation.rows_from_database(self.root)
            self.assertEqual(sum(row["status"] == "failed" for row in rows), 1)
            with db:
                db.execute("UPDATE episodes SET status='complete' WHERE status='pending' AND checkpoint='79999'")
            self.assertIsNone(evaluation.claim(db, 0, 0))

    def test_dynamic_aggregation_and_reference(self):
        self.manifest["reference_percent"] = 44.51
        result = evaluation.aggregate(self.completed(), self.manifest)
        self.assertEqual(result["overall"], {"successes": 8, "episodes": 16, "percent": 50.0})
        self.assertAlmostEqual(result["difference_pp"], 5.49)
        self.assertEqual(len(result["runs"]), 4)

    def test_single_checkpoint_single_seed(self):
        self.manifest["checkpoints"] = self.manifest["checkpoints"][:1]
        self.manifest["seeds"] = [123]
        result = evaluation.aggregate(self.completed(), self.manifest)
        self.assertEqual(result["overall"]["episodes"], 4)
        self.assertIsNone(result["sample_std_pp"])
        self.assertIsNone(result["difference_pp"])

    def test_missing_extra_duplicate_rows_are_rejected(self):
        rows = self.completed()
        for invalid in (rows[:-1], [*rows, rows[0]], [*rows[:-1], {**rows[-1], "seed": 999}]):
            with self.subTest(invalid=invalid[-1]), self.assertRaisesRegex(ValueError, "coverage"):
                evaluation.aggregate(invalid, self.manifest)

    def test_environment_scene_seed_must_match_between_runs(self):
        rows = self.completed()
        rows[-1]["env_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Environment seed changed"):
            evaluation.aggregate(rows, self.manifest)

    def test_invalid_outcomes_horizons_and_incomplete_rows(self):
        for replacement in ({"outcome": "error"}, {"env_seed": None}, {"steps": 1301}, {"status": "running"}):
            rows = self.completed()
            rows[0].update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                evaluation.aggregate(rows, self.manifest)

    def test_incremental_report_contains_only_completed_checkpoints(self):
        rows = self.completed()
        for row in rows:
            if row["checkpoint"] == "60000":
                row["status"] = "pending"
        evaluation.checkpoint_reports(self.root, self.manifest, rows)
        result = json.loads((self.root / "checkpoint_results.json").read_text())
        self.assertEqual(result["completed_checkpoints"], ["79999"])
        self.assertNotIn("60000", (self.root / "CHECKPOINTS.md").read_text())

    def test_resume_never_silently_ignores_protocol_overrides(self):
        for options in (["--seeds", "9"], ["--gpus=2"], ["--execute-steps", "8"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    evaluation.main(["run", "--resume", "--output-dir", str(self.root), *options])
                self.assertEqual(raised.exception.code, 2)

    def test_dry_run_does_not_create_output(self):
        args = evaluation.parser().parse_args(
            ["run", "--checkpoint", "/weights/79999", "--output-dir", str(self.root / "absent"), "--dry-run"]
        )
        with patch.object(evaluation, "make_manifest", return_value=self.manifest), patch("builtins.print"):
            self.assertEqual(evaluation.run(args), 0)
        self.assertFalse(args.output_dir.exists())

    def test_supervisor_completes_matrix_and_generates_reports(self):
        self.database()
        order = []

        def launch(root, manifest, gpu, stream):
            with closing(evaluation.db_connect(root)) as db:
                while (job := evaluation.claim(db, gpu, stream)) is not None:
                    order.append(job[0])
                    with db:
                        db.execute(
                            """UPDATE episodes SET status='complete',outcome='success',steps=1,env_seed=?
                            WHERE (checkpoint,seed,task,episode)=(?,?,?,?)""",
                            (100 + job[3], *job),
                        )
            return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

        with (
            patch.object(evaluation, "launch_worker", side_effect=launch),
            patch.object(evaluation, "stop_worker"),
            patch.object(evaluation.time, "sleep"),
        ):
            self.assertEqual(evaluation.supervise(self.root, self.manifest), 0)
        self.assertEqual(order, ["79999"] * 8 + ["60000"] * 8)
        self.assertEqual(json.loads((self.root / "status.json").read_text())["state"], "complete")
        self.assertEqual(json.loads((self.root / "results.json").read_text())["overall"]["episodes"], 16)

    def test_rollout_preserves_history_action_order_and_resets_seed(self):
        actions = []
        requests = []
        closed = []

        class State:
            def __init__(self):
                self.image_buffer, self.wrist_image_buffer, self.state_buffer = [], [], []
                self.action_plan, self.count = deque(), 0

            def get_current_obs(self):
                return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]

            def clear_buffers(self):
                self.image_buffer, self.wrist_image_buffer, self.state_buffer = [], [], []
                self.exec_start_idx = 0

            def add_observation(self, image, wrist, state):
                self.image_buffer.append(image)
                self.wrist_image_buffer.append(wrist)
                self.state_buffer.append(state)

        def call(request):
            requests.append(copy.deepcopy(request))
            return {"reset": True} if "reset" in request else {"actions": list(range(20))}

        def step(action):
            actions.append(action)
            return (("next", "wrist", "state"), len(actions) == 18, "success" if len(actions) == 18 else "unknown")

        runner = SimpleNamespace(
            make_env=lambda ep: None,
            difficulty="easy",
            step=step,
            close_env=lambda: closed.append(True),
            get_init_obs=lambda: {
                "images": ["demo", "initial"],
                "wrist_images": ["demo-w", "w"],
                "states": ["demo-s", "s"],
                "task_goal": "test",
            },
        )
        predictor = SimpleNamespace(
            start_episode=lambda *a: None,
            step=lambda *a: None,
            get_subgoal=lambda *a: (None, False),
            end_episode=lambda *a: None,
        )
        modules = {
            "gymnasium": SimpleNamespace(make=lambda *a, **kw: None),
            "utils": SimpleNamespace(EpisodeState=State, pack_buffer=lambda *a: copy.deepcopy(a)),
        }
        manifest = {**self.manifest, "subgoal_keep_period": 1}
        with patch.dict("sys.modules", modules):
            result = evaluation.rollout(SimpleNamespace(call=call), runner, predictor, 42, 2, manifest)
        self.assertEqual(result, ("success", 18, "easy"))
        self.assertEqual(actions, [*list(range(16)), 0, 1])
        self.assertEqual(requests[0], {"reset": 42})
        self.assertEqual(requests[1]["history"][0], ["demo", "initial"])
        self.assertEqual(requests[1]["history"][2], 1)
        self.assertEqual(len(requests[2]["history"][0]), 16)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
