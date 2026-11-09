import copy
import torch
import pytest

from mme_vla_suite.musa.numerics import backward_with_backoff


class HalfBoundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.25, .5, .75, 1.]))
        self.dropout = torch.nn.Dropout(.5)

    def forward(self):
        return self.dropout(self.weight.half().float()).sum()


def finite_norm(model):
    norm = torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite gradient norm')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    return [float(norm)]


def test_retry_restores_dropout_rng_and_matches_safe_update():
    model = HalfBoundary()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=5e-5)
    observed = []
    def backward(scale):
        value = model()
        observed.append(value.detach().clone())
        (value * scale).backward()
        return value.detach()
    torch.manual_seed(42)
    rng = torch.get_rng_state()
    events = []
    loss, norms, scale, retries = backward_with_backoff(
        model, backward, lambda: finite_norm(model), 2**20,
        torch.device('cpu'), report=lambda *event: events.append(event))
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    expected = reference()
    expected.backward()
    expected_norms = finite_norm(reference)
    assert retries > 0 and scale < 2**20 and len(events) == retries
    assert all(torch.equal(loss, value) for value in observed)
    assert torch.equal(loss, expected.detach()) and norms == expected_norms
    assert torch.equal(final_rng, torch.get_rng_state())
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.equal(a.grad, b.grad)
    optimizer.step()
    ref_optimizer.step()
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.equal(a, b)
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            assert torch.equal(optimizer.state[a][key], ref_optimizer.state[b][key])


def test_irrecoverable_gradient_fails_before_optimizer_update():
    model = HalfBoundary()
    original = model.weight.detach().clone()
    def backward(scale):
        value = model().mul(float('inf'))
        value.backward()
        return value.detach()
    with pytest.raises(FloatingPointError):
        backward_with_backoff(model, backward, lambda: finite_norm(model), 2, torch.device('cpu'))
    assert torch.equal(original, model.weight)


def test_finite_backward_keeps_scale_and_does_not_retry():
    model = HalfBoundary()
    def backward(scale):
        value = model()
        (value * scale).backward()
        return value.detach()
    _, _, scale, retries = backward_with_backoff(
        model, backward, lambda: finite_norm(model), 16, torch.device('cpu'))
    assert scale == 16 and retries == 0
