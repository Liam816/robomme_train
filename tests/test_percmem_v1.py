"""Behavior, gradients and current-frame equivalence on CPU; no pretrained weights."""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from flax import linen, nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from mme_vla_suite.models.representation.perc_memory_v1 import (
    EvictScorer, MemReader, PercMemConfig, pairwise_ranking_loss, pool_tokens,
)
from mme_vla_suite.shared.perc_mem_bank import MemoryClock, PercMemBank, eviction_index
from mme_vla_suite.shared.percmem_io import load_module, save_module, tree_hash, module_params


def tiny_config(**kwargs):
    return dataclasses.replace(PercMemConfig(), native_dim=32, feature_dim=40, output_dim=32,
                               store_grid=4, read_grid=2, score_grid=2, age_dim=8,
                               scorer_dim=16, scorer_ffn=64, reader_dim=16, reader_ffn=64,
                               reader_queries=2, max_frames=32, **kwargs)


class BankTest(unittest.TestCase):
    def test_fifo_capacity_2000_and_reset(self):
        bank = PercMemBank(mode="fifo")
        for i in range(2000):
            bank.write(i, i * 10)
            self.assertLessEqual(len(bank.W), 4)
            self.assertLessEqual(len(bank.Q), 28)
        self.assertEqual([f.index for f in bank.frames()], list(range(1968, 2000)))
        other = PercMemBank(mode="fifo")
        bank.reset()
        self.assertEqual(len(bank.frames()) + len(other.frames()), 0)

    def test_new_candidate_can_evict_itself(self):
        bank = PercMemBank(score_fn=lambda c, now: np.r_[np.ones(len(c)-1), -1.0])
        for i in range(33):
            bank.write(i, i * 10)
        self.assertNotIn(28, [f.index for f in bank.frames()])
        self.assertEqual([f.index for f in bank.W], [29, 30, 31, 32])

    def test_uniform_and_random(self):
        self.assertEqual(eviction_index([0, 10, 30, 20], "uniform"), 1)
        a, b = PercMemBank(mode="random"), PercMemBank(mode="random")
        for i in range(80):
            a.write(i, i * 10)
            b.write(i, i * 10)
        self.assertEqual([f.index for f in a.frames()], [f.index for f in b.frames()])

    def test_clock_mapping_and_missing_duplicate_observations(self):
        outputs = []
        for hz in [10, 20]:
            clock = MemoryClock()
            outputs.append([tick for i in range(2 * hz) for key, tick in [clock.observe(i / hz)] if key])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], [0, 10, 20, 30])
        clock = MemoryClock()
        self.assertEqual(clock.observe(10), (True, 0))
        self.assertEqual(clock.observe(10), (False, 0))
        self.assertEqual(clock.observe(11.2), (True, 24))
        with self.assertRaises(ValueError):
            clock.observe(11.1)


class NetworkTest(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config()
        self.x = jax.random.normal(jax.random.key(3), (1, 3, 16, 40))
        self.ticks = jnp.array([[0, 10, 20]])
        self.mask = jnp.array([[True, False, True]])
        self.now = jnp.array([30])

    def test_pool_spatial_order(self):
        x = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)
        np.testing.assert_array_equal(pool_tokens(x, 2)[0, :, 0], [2.5, 4.5, 10.5, 12.5])

    def test_reader_zero_initialization_padding_empty_and_gradients(self):
        reader = MemReader(self.cfg, rngs=nnx.Rngs(1))
        out, attention = reader(self.x, self.ticks, self.mask, self.now, return_attention=True)
        np.testing.assert_array_equal(out, np.zeros((1, 32)))
        np.testing.assert_array_equal(attention[..., 1, :], 0)

        def loss(module):
            return module(self.x, self.ticks, self.mask, self.now).sum()

        grads = nnx.grad(loss)(reader)
        self.assertGreater(float(jnp.linalg.norm(grads.output_final.kernel.value)), 0)
        reader.output_final.kernel.value -= 0.001 * grads.output_final.kernel.value
        later = nnx.grad(loss)(reader)
        self.assertGreater(float(jnp.linalg.norm(later.proj.kernel.value)), 0)
        empty, weights = reader(self.x, self.ticks, jnp.zeros_like(self.mask), self.now, return_attention=True)
        self.assertTrue(np.isfinite(empty).all())
        np.testing.assert_array_equal(empty, 0)
        np.testing.assert_array_equal(weights, 0)

    def test_scorer_eval_and_checkpoint_roundtrip(self):
        scorer = EvictScorer(self.cfg, rngs=nnx.Rngs(4))
        new = jnp.array([[False, False, True]])
        a = scorer(self.x, self.ticks, self.mask, self.now, new)
        b = scorer(self.x, self.ticks, self.mask, self.now, new)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isinf(np.asarray(a)[0, 1]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.npz"
            save_module(path, scorer, {"teacher_id": "test"})
            loaded = EvictScorer(self.cfg, rngs=nnx.Rngs(8))
            meta = load_module(loaded, path)
            self.assertEqual(meta["teacher_id"], "test")
            self.assertEqual(tree_hash(module_params(scorer)), tree_hash(module_params(loaded)))
            np.testing.assert_array_equal(a, loaded(self.x, self.ticks, self.mask, self.now, new))
            incompatible = EvictScorer(dataclasses.replace(self.cfg, norm_eps=1e-3), rngs=nnx.Rngs(8))
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_module(incompatible, path)

    def test_pairwise_direction_and_no_valid_pairs(self):
        key, valid = jax.random.key(1), jnp.ones(3, bool)
        utility = jnp.array([1., 0., .5])
        good = pairwise_ranking_loss(jnp.array([2., -2., 0.]), utility, valid, key)
        bad = pairwise_ranking_loss(jnp.array([-2., 2., 0.]), utility, valid, key)
        self.assertLess(float(good), float(bad))
        self.assertEqual(float(pairwise_ranking_loss(jnp.zeros(3), jnp.zeros(3), valid, key)), 0)


class TinyVision(linen.Module):
    num_classes: int
    variant: str = ""
    pool_type: str = ""
    scan: bool = True
    dtype_mm: str = ""

    @linen.compact
    def __call__(self, images, train=False):
        del train
        vector = linen.Dense(self.num_classes)(images.mean(axis=(1, 2)))
        return jnp.broadcast_to(vector[:, None], (images.shape[0], 4, self.num_classes)), None


class IntegrationTest(unittest.TestCase):
    def test_actual_train_state_initialization_matches_shape_graph(self):
        import importlib.util
        from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
        from mme_vla_suite.training.config import TrainConfig
        from openpi.training.sharding import DATA_AXIS, make_mesh
        from openpi.training.weight_loaders import NoOpWeightLoader

        path = Path(__file__).resolve().parents[1] / "scripts/train.py"
        spec = importlib.util.spec_from_file_location("percmem_init_test_train", path)
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        history = OmegaConf.create({"integration_type": "perc_adarms", "representation_type": "perceptual",
                                    "perc_mem": dataclasses.asdict(tiny_config())})
        model_config = HistoryPi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
            action_dim=8, action_horizon=3, max_token_len=4, use_history=True, history_config=history)
        config = TrainConfig(name="cpu_init_test", exp_name="cpu_init_test", model=model_config,
            freeze_filter=model_config.get_freeze_filter(), weight_loader=NoOpWeightLoader(), fsdp_devices=1)
        mesh = make_mesh(1)
        with patch("mme_vla_suite.models.integration.history_pi0._siglip.Module", TinyVision):
            state, state_sharding = training.init_train_state(config, jax.random.key(9), mesh, resume=False)
        jax.block_until_ready(state)
        self.assertEqual(int(state.step), 0)
        np.testing.assert_array_equal(state.params.mem_feature_adapter.kernel.value, np.eye(32, 40))
        self.assertGreater(float(jnp.linalg.norm(state.params.mem_cond_adapter.kernel.value)), 0)
        frozen_hash = tree_hash(state.params.filter(config.freeze_filter).to_pure_dict())
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(DATA_AXIS))
        update = training.compile_train_step(config, state_sharding, data_sharding, replicated)
        obs = model_config.fake_obs()
        obs = dataclasses.replace(obs, perc_mem_frame_mask=jnp.ones((1, 32), bool))
        for _ in range(3):
            state, info, _ = update(jax.random.key(10), state, (obs, model_config.fake_act()), None)
            jax.block_until_ready(state)
            self.assertTrue(np.isfinite(float(info["loss"])))
            self.assertEqual(tree_hash(state.params.filter(config.freeze_filter).to_pure_dict()), frozen_hash)
            self.assertEqual(tree_hash(state.ema_params.filter(config.freeze_filter).to_pure_dict()), frozen_hash)
        self.assertEqual(int(state.step), 3)

    def test_p3_optimizer_step_keeps_teacher_and_frozen_modules_unchanged(self):
        import importlib.util
        import optax
        from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
        from mme_vla_suite.training.config import TrainConfig
        from openpi.training.utils import TrainState

        path = Path(__file__).resolve().parents[1] / "scripts/train.py"
        spec = importlib.util.spec_from_file_location("percmem_test_train", path)
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        cfg = dataclasses.replace(tiny_config(), stage="student", input_space="projected")
        history = OmegaConf.create({"integration_type": "perc_adarms", "representation_type": "perceptual",
                                    "perc_mem": dataclasses.asdict(cfg)})
        model_config = HistoryPi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
            action_dim=8, action_horizon=3, max_token_len=4, use_history=True, history_config=history)
        with patch("mme_vla_suite.models.integration.history_pi0._siglip.Module", TinyVision):
            student = model_config.create(jax.random.key(9))
            teacher = model_config.create(jax.random.key(9))
        config = TrainConfig(name="cpu_test", exp_name="cpu_test", model=model_config,
                             freeze_filter=model_config.get_freeze_filter())
        params = nnx.state(student)
        tx = optax.adam(1e-3)
        state = TrainState(step=0, params=params, model_def=nnx.graphdef(student), tx=tx,
                           opt_state=tx.init(params.filter(config.trainable_filter)), ema_decay=.999, ema_params=params)
        teacher_params = nnx.state(teacher)
        teacher_hash = tree_hash(teacher_params.to_pure_dict())
        frozen_before = tree_hash(params.filter(config.freeze_filter).to_pure_dict())
        train_before = tree_hash(params.filter(config.trainable_filter).to_pure_dict())
        obs = model_config.fake_obs()
        obs = dataclasses.replace(obs, perc_mem_frame_mask=jnp.arange(32)[None] < 2,
                                   perc_teacher_mask=jnp.arange(32)[None] < 5)
        new_state, info, _ = training.train_step(config, jax.random.key(10), state,
                                                (obs, model_config.fake_act()), teacher_params)
        self.assertTrue(np.isfinite(float(info["loss"])))
        self.assertEqual(float(info["distill_loss"]), 0)
        self.assertEqual(tree_hash(teacher_params.to_pure_dict()), teacher_hash)
        self.assertEqual(tree_hash(new_state.params.filter(config.freeze_filter).to_pure_dict()), frozen_before)
        self.assertEqual(tree_hash(new_state.ema_params.filter(config.freeze_filter).to_pure_dict()), frozen_before)
        self.assertNotEqual(tree_hash(new_state.params.filter(config.trainable_filter).to_pure_dict()), train_before)

    def test_actual_gemma_train_and_sample_equal_at_zero_memory(self):
        from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
        from mme_vla_suite.models.integration.history_observation import HistAugObservation, preprocess_observation

        cfg = tiny_config()
        history = OmegaConf.create({"integration_type": "perc_adarms", "representation_type": "perceptual",
                                    "perc_mem": dataclasses.asdict(cfg)})
        config = HistoryPi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                  action_dim=8, action_horizon=3, max_token_len=4,
                                  use_history=True, history_config=history)
        with patch("mme_vla_suite.models.integration.history_pi0._siglip.Module", TinyVision):
            model = config.create(jax.random.key(9))
            baseline = dataclasses.replace(config, use_history=False, history_config=None).create(jax.random.key(9))
        obs = config.fake_obs()
        obs = dataclasses.replace(obs, perc_mem_frame_mask=jnp.ones((1, 32), bool),
                                   perc_mem_frame_tick=jnp.zeros((1, 32), jnp.int32))
        normalized = preprocess_observation(None, obs, train=False)
        self.assertEqual(normalized.perc_mem_feats.shape, obs.perc_mem_feats.shape)
        self.assertEqual(HistAugObservation.from_dict(obs.to_dict()).perc_mem_feats.shape, obs.perc_mem_feats.shape)
        actions = config.fake_act()
        key = jax.random.key(123)
        a, _ = model.compute_loss(key, obs, actions)
        b, _ = baseline.compute_loss(key, obs, actions)
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        noise = jax.random.normal(key, actions.shape)
        original = model.memory_condition
        with patch.object(type(model), "memory_condition", side_effect=original) as reader:
            sample = model.sample_actions(key, obs, noise=noise, num_steps=2)
            self.assertEqual(reader.call_count, 1)
        reference = baseline.sample_actions(key, obs, noise=noise, num_steps=2)
        np.testing.assert_allclose(sample, reference, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
