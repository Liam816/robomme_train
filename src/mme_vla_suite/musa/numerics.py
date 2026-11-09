"""FP16-intermediate loss scaling and narrowly scoped resume compatibility.

Scaling affects the backward execution only. Gradients are restored to their
original units before clipping, AdamW, EMA or logging; forward/storage are intact.
"""
import copy
import math
import torch

EXECUTION_KEYS = ('micro_batch_size', 'gradient_accumulation_steps',
                  'gradient_loss_scale', 'runtime_overlay', 'runtime_overlay_sha256')

def loss_scale(profile):
    value = float(profile['musa'].get('gradient_loss_scale', 1.))
    if not math.isfinite(value) or value < 1 or value > 2**24 or math.frexp(value)[0] != .5:
        raise ValueError('gradient_loss_scale must be an exact power of two in [1, 2**24]')
    return value

@torch.no_grad()
def unscale_gradients(model, scale):
    if scale == 1: return
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if parameter.grad.dtype != torch.float32:
                raise TypeError('This backend requires FP32 trainable master gradients')
            parameter.grad.div_(scale)


def backward_with_backoff(model, backward, gradient_norms, scale, device, report=None):
    """Retry the same batch and dropout draws before any optimizer/EMA update.

    gradient_norms must reduce across ranks before rejecting a non-finite norm,
    so every rank chooses the same scale. Scale only decreases, and stays >=1.
    """
    cpu_rng = torch.get_rng_state()
    device_rng = torch.musa.get_rng_state(device) if device.type == 'musa' else None
    retries = 0
    while True:
        model.zero_grad(set_to_none=True)
        if retries:
            torch.set_rng_state(cpu_rng)
            if device_rng is not None:
                torch.musa.set_rng_state(device_rng, device)
        loss = backward(scale)
        unscale_gradients(model, scale)
        try:
            norms = gradient_norms()
        except FloatingPointError:
            if scale <= 1:
                raise
            previous = scale
            scale /= 2
            retries += 1
            if report is not None:
                report(previous, scale, retries)
        else:
            return loss, norms, scale, retries

def resume_profiles_compatible(saved, current):
    """Same experiment and assets; only batch splitting/numeric execution may vary."""
    saved, current = copy.deepcopy(saved), copy.deepcopy(current)
    for profile in (saved, current):
        train, runtime = profile['train'], profile['runtime']
        if train['batch_size'] != 64 or train['fsdp_devices'] != 4:
            return False
        if runtime['cuda_visible_devices'] != [4, 5, 6, 7]:
            return False
        if profile['musa']['micro_batch_size'] * profile['musa']['gradient_accumulation_steps'] * 4 != 64:
            return False
        loss_scale(profile)
        for key in EXECUTION_KEYS: profile['musa'].pop(key, None)
    return saved == current
