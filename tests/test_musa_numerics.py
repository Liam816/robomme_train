"""CPU checks for objective-preserving backward scaling and resume restrictions."""
import copy
import unittest
import torch
from mme_vla_suite.musa.numerics import loss_scale, unscale_gradients, resume_profiles_compatible

class NumericExecutionTests(unittest.TestCase):
    def test_scaling_preserves_fp32_gradient_optimizer_and_ema(self):
        torch.manual_seed(42)
        first = torch.nn.Linear(7, 4); second = copy.deepcopy(first)
        one = torch.optim.AdamW(first.parameters(), lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=0)
        two = torch.optim.AdamW(second.parameters(), lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=0)
        x, target = torch.randn(8, 7), torch.randn(8, 4)
        a = (first(x) - target).square().mean(); b = (second(x) - target).square().mean()
        self.assertTrue(torch.equal(a, b))
        a.backward(); (b * 2**20).backward(); unscale_gradients(second, 2**20)
        for p, q in zip(first.parameters(), second.parameters()): self.assertTrue(torch.equal(p.grad, q.grad))
        self.assertTrue(torch.equal(torch.nn.utils.clip_grad_norm_(first.parameters(), 1.), torch.nn.utils.clip_grad_norm_(second.parameters(), 1.)))
        old = [p.detach().clone() for p in first.parameters()]
        one.step(); two.step()
        for initial, p, q in zip(old, first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(p, q))
            self.assertTrue(torch.equal(initial * .999 + p.detach() * .001, initial * .999 + q.detach() * .001))
        for left, right in zip(one.state.values(), two.state.values()):
            for key in left: self.assertTrue(torch.equal(left[key], right[key]))

    def test_recovers_fp16_cast_underflow_without_changing_forward(self):
        first = torch.nn.Parameter(torch.ones(())); second = torch.nn.Parameter(torch.ones(()))
        a = first.half().float() * 1e-10; b = second.half().float() * 1e-10
        self.assertTrue(torch.equal(a, b)); a.backward(); (b * 2**20).backward()
        holder = torch.nn.ParameterList([second]); unscale_gradients(holder, 2**20)
        self.assertEqual(float(first.grad), 0.)
        self.assertGreater(float(second.grad), 0.)
        self.assertLess(abs(float(second.grad) / 1e-10 - 1), .001)

    def test_resume_keeps_model_optimizer_data_and_schedule_strict(self):
        old = {'train': {'batch_size': 64, 'fsdp_devices': 4, 'num_train_steps': 80000},
               'runtime': {'cuda_visible_devices': [4, 5, 6, 7]},
               'musa': {'micro_batch_size': 2, 'gradient_accumulation_steps': 8,
                        'runtime_overlay': '/old', 'runtime_overlay_sha256': 'old', 'root': '/runtime'},
               'model': {'history_config': 'same'}, 'optimizer': {'eps': 1e-8},
               'data': {'dataset_path': '/same'}, 'learning_rate': {'value': 5e-5}}
        new = copy.deepcopy(old); new['musa'].update(micro_batch_size=16, gradient_accumulation_steps=1,
            gradient_loss_scale=2**20, runtime_overlay='/new', runtime_overlay_sha256='new')
        self.assertTrue(resume_profiles_compatible(old, new))
        for group, key, value in [('model', 'history_config', 'different'), ('optimizer', 'eps', 1e-7),
            ('data', 'dataset_path', '/other'), ('learning_rate', 'value', 1e-3),
            ('train', 'num_train_steps', 80001), ('runtime', 'cuda_visible_devices', [0, 1, 2, 3])]:
            bad = copy.deepcopy(new); bad[group][key] = value
            self.assertFalse(resume_profiles_compatible(old, bad), (group, key))
        self.assertEqual(loss_scale(new), 2**20)
        for value in [0, .5, 3, float('inf'), 2**25]:
            bad = copy.deepcopy(new); bad['musa']['gradient_loss_scale'] = value
            with self.assertRaises(ValueError): loss_scale(bad)

if __name__ == '__main__': unittest.main()
