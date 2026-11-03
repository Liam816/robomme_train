"""E2E hard-write parity, full-history gradients and P0 experiment alignment."""

import dataclasses
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import yaml

from mme_vla_suite.models.representation.perc_memory_e2e import hard_eviction, replay_memory
from mme_vla_suite.models.representation.perc_memory_v1 import EvictScorer
from mme_vla_suite.shared.perc_mem_bank import PercMemBank
from mme_vla_suite.shared.percmem_io import tree_hash
from test_percmem_v1 import TinyVision, tiny_config


ROOT = Path(__file__).resolve().parents[1]


def small_config(**kwargs):
    values = dict(stage="e2e", recent_frames=2, queue_frames=3, max_frames=12)
    values.update(kwargs)
    return dataclasses.replace(tiny_config(), **values)


class QueueTest(unittest.TestCase):
    def test_online_writer_scores_fp16_storage_in_fp32(self):
        from mme_vla_suite.policies.percmem_policy import PercMemPolicy
        from mme_vla_suite.shared.perc_mem_bank import Frame
        policy = object.__new__(PercMemPolicy)
        seen = []
        def score(tokens, ticks, valid, now, new):
            seen.append(tokens.dtype)
            return jnp.zeros(ticks.shape)
        policy._score = score
        policy._score_candidates([Frame(0, 0, np.ones((16, 40), np.float16))], 0)
        self.assertEqual(seen, [jnp.float32])

    def test_credit_reaches_early_writes_without_history_truncation(self):
        cfg = small_config()
        class ScoreTable(nnx.Module):
            def __init__(self):
                self.table = nnx.Param(jnp.zeros((12, 4)).at[:, -1].set(-.25))
            def encode_frames(self, x):
                return x[:, :, 0, :1]
            def score_frames(self, x, ticks, valid, now, is_new, **kwargs):
                return self.table.value[now // 10]
        scorer = ScoreTable()
        x = jnp.broadcast_to(jnp.arange(12)[None, :, None, None], (1, 12, 16, 40)).astype(jnp.float16)
        def objective(module):
            selected, _, _, _ = replay_memory(module, x, jnp.arange(12)[None] * 10,
                                               jnp.ones((1, 12), bool), jnp.array([115]), cfg)
            return selected.astype(jnp.float32).sum()
        gradient = nnx.grad(objective)(scorer).table.value
        np.testing.assert_array_equal(gradient[:5], 0)  # W/Q warm-up has no eviction decision.
        self.assertGreater(float(jnp.linalg.norm(gradient[5])), 0)  # First overflow, six writes before the loss.
        self.assertGreater(float(jnp.linalg.norm(gradient[11])), 0)

    def test_exact_hard_forward_has_nonzero_soft_derivative(self):
        scores = jnp.array([[1., -2., .5]])
        np.testing.assert_array_equal(hard_eviction(scores), [[0., 1., 0.]])
        gradient = jax.grad(lambda s: (hard_eviction(s) * jnp.array([[2., 5., -3.]])).sum())(scores)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0)
        self.assertAlmostEqual(float(gradient.sum()), 0., places=6)

    def test_replay_matches_online_bank_with_padding_holes_and_future(self):
        cfg = small_config()
        scorer = EvictScorer(cfg, rngs=nnx.Rngs(8))
        feats = jax.random.normal(jax.random.key(4), (3, 12, 16, 40)).astype(jnp.float16)
        ticks = jnp.broadcast_to(jnp.arange(12)[None] * 10, (3, 12))
        valid = jnp.array([[True] * 12, [True, False, True, True, True, False, True, True, True, True, True, True],
                           [False] * 12])
        now = jnp.array([115, 85, 115])
        tokens, actual_ticks, actual_valid, ids = jax.jit(
            lambda x, t, v, z: replay_memory(scorer, x, t, v, z, cfg))(feats, ticks, valid, now)
        for batch in range(3):
            def score(candidates, event_tick):
                x = jnp.asarray(np.stack([f.tokens for f in candidates]))[None].astype(jnp.float32)
                t = jnp.array([[f.tick for f in candidates]])
                mask = jnp.ones(t.shape, bool)
                new = jnp.arange(t.shape[1])[None] == t.shape[1] - 1
                return np.asarray(scorer(x, t, mask, jnp.array([event_tick]), new))[0]
            bank = PercMemBank(cfg.recent_frames, cfg.queue_frames, score_fn=score)
            for i in range(12):
                if bool(valid[batch, i]) and int(ticks[batch, i]) <= int(now[batch]):
                    bank.write(i, int(ticks[batch, i]), np.asarray(feats[batch, i]))
            expected, expected_ticks, expected_valid, expected_ids = bank.arrays(cfg.capacity, 40, 16)
            np.testing.assert_array_equal(ids[batch], expected_ids)
            np.testing.assert_array_equal(actual_ticks[batch], expected_ticks)
            np.testing.assert_array_equal(actual_valid[batch], expected_valid)
            np.testing.assert_array_equal(tokens[batch], expected)

    def test_candidate_rejection_and_mandatory_recent_window(self):
        cfg = small_config()
        class RejectCandidate:
            def encode_frames(self, x):
                return x[:, :, 0, :1]
            def score_frames(self, x, ticks, valid, now, is_new, **kwargs):
                return -100. * is_new.astype(jnp.float32)
        x = jnp.broadcast_to(jnp.arange(12)[None, :, None, None], (1, 12, 16, 40)).astype(jnp.float16)
        _, _, valid, ids = replay_memory(RejectCandidate(), x, jnp.arange(12)[None] * 10,
                                         jnp.ones((1, 12), bool), jnp.array([115]), cfg)
        np.testing.assert_array_equal(ids, [[0, 1, 2, 10, 11]])
        self.assertEqual(int(valid.sum()), 5)


class ExperimentTest(unittest.TestCase):
    def test_weight_loader_initializes_memory_and_rejects_p0_memory_weights(self):
        from mme_vla_suite.training.percmem_profile import PercMemWeightLoader
        reference = {"base": {"kernel": jax.ShapeDtypeStruct((2, 2), jnp.float32)},
                     "mem_reader": {"kernel": jax.ShapeDtypeStruct((2, 2), jnp.float32)}}
        loader = PercMemWeightLoader("unused", "e2e")
        base = {"base": {"kernel": np.ones((2, 2), np.float32)}}
        with patch("mme_vla_suite.training.percmem_profile.restore_params", return_value=base):
            loaded = loader.load(reference)
        self.assertIsInstance(loaded["mem_reader"]["kernel"], jax.ShapeDtypeStruct)
        pretrained = {**base, "mem_reader": {"kernel": np.ones((2, 2), np.float32)}}
        with patch("mme_vla_suite.training.percmem_profile.restore_params", return_value=pretrained):
            with self.assertRaisesRegex(ValueError, "fresh memory initialization"):
                loader.load(reference)

    def test_profile_only_changes_memory_policy_and_run_identity(self):
        from mme_vla_suite.training.percmem_profile import build_config, review_config
        p0 = yaml.safe_load((ROOT / "configs/train/robomme_percmem_v1_teacher_repro.yaml").read_text())
        e2e = yaml.safe_load((ROOT / "configs/train/robomme_percmem_v1_e2e_repro.yaml").read_text())
        for key in ["data", "weights", "freeze", "learning_rate", "optimizer", "evaluation"]:
            self.assertEqual(p0[key], e2e[key])
        for key in p0["train"]:
            if key != "exp_name":
                self.assertEqual(p0["train"][key], e2e["train"][key])
        for key in p0["runtime"]:
            if key != "compilation_cache_dir":
                self.assertEqual(p0["runtime"][key], e2e["runtime"][key])
        for key in p0["model"]:
            if key != "history_config":
                self.assertEqual(p0["model"][key], e2e["model"][key])
        config, memory = build_config(e2e)
        self.assertTrue(config.trainable_filter(("mem_scorer", "output", "kernel"), nnx.VariableState(nnx.Param, jnp.ones(1))))
        self.assertTrue(config.trainable_filter(("mem_reader", "query"), nnx.VariableState(nnx.Param, jnp.ones(1))))
        report = review_config(e2e, config, memory)
        self.assertEqual(report["memory_training"]["reader_capacity_frames"], 32)
        self.assertEqual(report["memory_training"]["history_gradient"], "full_unroll_no_truncation")
        self.assertFalse(report["capacity_tested"])
        with self.assertRaisesRegex(ValueError, "native keyframes"):
            changed = dataclasses.replace(config.model, history_config=OmegaConf.create({
                **OmegaConf.to_container(memory), "perc_mem": {**dict(memory.perc_mem), "input_space": "projected"}}))
            profile = {**e2e, "model": {**e2e["model"], "history_config": changed.history_config}}
            build_config(profile)

    def test_p0_e2e_initial_weights_identical_and_fm_updates_scorer(self):
        from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
        from mme_vla_suite.training.config import TrainConfig
        from openpi.training.optimizer import CosineDecaySchedule
        from openpi.training.sharding import DATA_AXIS, make_mesh
        from openpi.training.weight_loaders import NoOpWeightLoader

        cfg = small_config(recent_frames=1, queue_frames=2, max_frames=6)
        history = OmegaConf.create({"representation_type": "perceptual", "integration_type": "perc_adarms",
                                    "perc_mem": dataclasses.asdict(cfg)})
        mc = HistoryPi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
            action_dim=8, action_horizon=3, max_token_len=4, use_history=True, history_config=history)
        h0 = OmegaConf.create(OmegaConf.to_container(history))
        h0.perc_mem.stage = "teacher"
        with patch("mme_vla_suite.models.integration.history_pi0._siglip.Module", TinyVision):
            teacher = dataclasses.replace(mc, history_config=h0).create(jax.random.key(9))
            model = mc.create(jax.random.key(9))
        self.assertEqual(tree_hash(nnx.state(teacher).to_pure_dict()), tree_hash(nnx.state(model).to_pure_dict()))

        spec = importlib.util.spec_from_file_location("e2e_test_train", ROOT / "scripts/train.py")
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        config = TrainConfig(name="e2e_test", exp_name="e2e_test", model=mc, freeze_filter=mc.get_freeze_filter(),
            fsdp_devices=1, weight_loader=NoOpWeightLoader(),
            lr_schedule=CosineDecaySchedule(warmup_steps=1, peak_lr=1e-3, decay_steps=10, decay_lr=1e-3))
        mesh = make_mesh(1)
        with patch("mme_vla_suite.models.integration.history_pi0._siglip.Module", TinyVision):
            state, shards = training.init_train_state(config, jax.random.key(9), mesh, resume=False)
        before_scorer = tree_hash(state.params.mem_scorer.to_pure_dict())
        before_reader = tree_hash(state.params.mem_reader.to_pure_dict())
        frozen = tree_hash(state.params.filter(config.freeze_filter).to_pure_dict())
        obs = mc.fake_obs()
        obs = dataclasses.replace(obs,
            perc_mem_feats=jax.random.normal(jax.random.key(31), (1, 6, 16, 32)).astype(jnp.float16),
            perc_mem_frame_tick=jnp.arange(6)[None] * 10, perc_mem_frame_mask=jnp.ones((1, 6), bool),
            perc_mem_now_tick=jnp.array([60]))
        rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(DATA_AXIS))
        update = training.compile_train_step(config, shards, data, rep)
        scorer_gradients = []
        for _ in range(3):
            state, info, _ = update(jax.random.key(41), state, (obs, mc.fake_act()), None)
            jax.block_until_ready(state)
            self.assertTrue(np.isfinite(float(info["loss"])))
            scorer_gradients.append(float(info["scorer_grad_norm"]))
            self.assertEqual(tree_hash(state.params.filter(config.freeze_filter).to_pure_dict()), frozen)
            self.assertEqual(tree_hash(state.ema_params.filter(config.freeze_filter).to_pure_dict()), frozen)
        self.assertEqual(scorer_gradients[0], 0.)  # P0-compatible zero reader output initialization.
        self.assertGreater(max(scorer_gradients[1:]), 0.)
        self.assertNotEqual(tree_hash(state.params.mem_scorer.to_pure_dict()), before_scorer)
        self.assertNotEqual(tree_hash(state.params.mem_reader.to_pure_dict()), before_reader)

        # Native full-history replay and projected online W/Q must feed the same reader.
        model = nnx.merge(state.model_def, state.params)
        replay_condition = model.memory_condition(obs)
        projected = np.asarray(model.project_memory(obs.perc_mem_feats))[0]
        def score(candidates, tick):
            values = jnp.asarray(np.stack([f.tokens for f in candidates]))[None].astype(jnp.float32)
            ticks = jnp.array([[f.tick for f in candidates]])
            new = jnp.arange(ticks.shape[1])[None] == ticks.shape[1] - 1
            return np.asarray(model.mem_scorer(values, ticks, jnp.ones(ticks.shape, bool), jnp.array([tick]), new))[0]
        bank = PercMemBank(cfg.recent_frames, cfg.queue_frames, score_fn=score)
        for i in range(6):
            bank.write(i, i * 10, projected[i])
        values, ticks, valid, _ = bank.arrays(cfg.capacity, cfg.feature_dim, cfg.store_grid ** 2)
        model.memory_cfg = dataclasses.replace(cfg, input_space="projected")
        online = dataclasses.replace(obs, perc_mem_feats=jnp.asarray(values)[None],
                                    perc_mem_frame_tick=jnp.asarray(ticks)[None],
                                    perc_mem_frame_mask=jnp.asarray(valid)[None])
        np.testing.assert_allclose(model.memory_condition(online), replay_condition, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
